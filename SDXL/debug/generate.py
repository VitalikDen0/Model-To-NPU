#!/usr/bin/env python3
"""
Quick SDXL Lightning generator — play with prompts on phone NPU.
All 4 components (CLIP-L, CLIP-G, UNet, VAE) run on Qualcomm NPU.

Usage:
  python NPU/generate.py "1girl, cherry blossoms, masterpiece"
  python NPU/generate.py "landscape, mountains, sunset" --seed 123
  python NPU/generate.py "cat sitting on windowsill" --seed 777 --no-stretch
  python NPU/generate.py "dark fantasy castle" --name castle
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PLATFORM_TOOLS = ROOT.parent
DEFAULT_SDXL_NPU = EXTERNAL_PLATFORM_TOOLS / "sdxl_npu"
SDXL_NPU = DEFAULT_SDXL_NPU if DEFAULT_SDXL_NPU.exists() else (ROOT / "sdxl_npu")
sys.path.insert(0, str(SDXL_NPU))

from export_sdxl_to_onnx import (
    collect_unet_resnet_conditioning_modules,
    compute_external_resnet_biases,
    infer_unet_resnet_spatial_shapes,
)

DIFFUSERS_DIR = SDXL_NPU / "diffusers_pipeline"
MERGED_UNET_DIR = SDXL_NPU / "unet_lightning8step_merged"
ADB_CANDIDATES = [
    ROOT / "adb.exe",
    EXTERNAL_PLATFORM_TOOLS / "adb.exe",
    Path(r"D:\platform-tools\adb.exe"),
]
ADB = str(next((p for p in ADB_CANDIDATES if p.exists()), ADB_CANDIDATES[0]))
DR = os.environ.get("SDXL_QNN_BASE", "/sdcard/Download/sdxl_qnn")
OUTPUT_DIR = ROOT / "NPU" / "outputs"
WORK = ROOT / "NPU" / "runtime_work_gen"

# ── globals loaded once ──
_unet = None
_resnet_modules = None
_spatial_shapes = None
_scheduler = None
_tok_l = None
_tok_g = None


def adb(*a, check=True, cap=False):
    cmd = [ADB] + list(a)
    if cap:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if check and r.returncode:
            raise RuntimeError(r.stderr)
        return r.stdout.strip()
    subprocess.run(cmd, check=check, timeout=600)


def push(local, remote):
    adb("push", str(local), remote)


def pull(remote, local):
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    adb("pull", remote, str(local))


def pull_first_remote_raw(remote_result_dir: str, local_raw: Path, candidates: tuple[str, ...]) -> str:
    listing = adb("shell", f"ls {remote_result_dir}", check=False, cap=True)
    files = set(listing.split()) if listing else set()
    for name in candidates:
        if name in files:
            pull(f"{remote_result_dir}/{name}", local_raw)
            return name
    raise FileNotFoundError(
        f"No expected raw output in {remote_result_dir}. "
        f"Expected one of: {', '.join(candidates)}. ls output: {listing!r}"
    )


def nhwc_f32(arr):
    return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32).tobytes()


def qnn_run(ctx, ilist, odir, native=False):
    nf = "--use_native_output_files " if native else ""
    cmd = (
        f"cd {DR} && "
        f"export LD_LIBRARY_PATH={DR}/lib:{DR}/bin:{DR}/model && "
        f"export ADSP_LIBRARY_PATH='{DR}/lib;/vendor/lib64/rfs/dsp;"
        f"/vendor/lib/rfsa/adsp;/vendor/dsp' && "
        f"mkdir -p {odir} && "
        f"{DR}/bin/qnn-net-run "
        f"--retrieve_context {ctx} "
        f"--backend {DR}/lib/libQnnHtp.so "
        f"--input_list {ilist} "
        f"--output_dir {odir} "
        f"{nf}--perf_profile burst --log_level warn"
    )
    t0 = time.time()
    adb("shell", cmd)
    return (time.time() - t0) * 1000


def init():
    global _unet, _resnet_modules, _spatial_shapes, _scheduler, _tok_l, _tok_g
    if _unet is not None:
        return

    from transformers import CLIPTokenizer
    from diffusers import UNet2DConditionModel, EulerDiscreteScheduler

    print("[init] Loading tokenizers...")
    _tok_l = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer"))
    _tok_g = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer_2"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] Loading Lightning UNet on {dev}...")
    _unet = UNet2DConditionModel.from_pretrained(
        str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
    ).to(dev).eval()

    _resnet_modules = collect_unet_resnet_conditioning_modules(_unet)
    _spatial_shapes = infer_unet_resnet_spatial_shapes(_unet, 128, 128)

    cfg = json.loads((DIFFUSERS_DIR / "scheduler" / "scheduler_config.json").read_text())
    _scheduler = EulerDiscreteScheduler.from_config(cfg, timestep_spacing="trailing")
    print("[init] Ready!\n")


def generate(prompt, seed=42, steps=8, stretch=True, name=None):
    init()
    dev = next(_unet.parameters()).device
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    WORK.mkdir(parents=True, exist_ok=True)
    dw = f"{DR}/runtime_work_gen"

    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}, Steps: {steps}")
    t_total = time.time()

    # ── 1. Tokenize ──
    ids_l = _tok_l(prompt, padding="max_length", max_length=77,
                   truncation=True, return_tensors="np")["input_ids"].astype(np.float32)
    ids_g = _tok_g(prompt, padding="max_length", max_length=77,
                   truncation=True, return_tensors="np")["input_ids"].astype(np.float32)

    cd = WORK / "clip"
    cd.mkdir(parents=True, exist_ok=True)
    ids_l.tofile(str(cd / "ids_l.raw"))
    ids_g.tofile(str(cd / "ids_g.raw"))
    dc = f"{dw}/clip"

    for fname in ["ids_l.raw", "ids_g.raw"]:
        path = cd / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file after write: {path}")
    with open(cd / "il_l.txt", "w") as f:
        f.write(f"{dc}/ids_l.raw\n")
    with open(cd / "il_g.txt", "w") as f:
        f.write(f"{dc}/ids_g.raw\n")

    adb("shell", f"mkdir -p {dc}")
    for f in ["ids_l.raw", "ids_g.raw", "il_l.txt", "il_g.txt"]:
        push(cd / f, f"{dc}/{f}")

    # ── 2. CLIP ──
    print("[CLIP-L]", end=" ", flush=True)
    ms_l = qnn_run(f"{DR}/context/clip_l.serialized.bin.bin",
                   f"{dc}/il_l.txt", f"{dc}/out_l")
    print(f"{ms_l:.0f}ms", end="  ", flush=True)

    print("[CLIP-G]", end=" ", flush=True)
    ms_g = qnn_run(f"{DR}/context/clip_g.serialized.bin.bin",
                   f"{dc}/il_g.txt", f"{dc}/out_g")
    print(f"{ms_g:.0f}ms")

    co = cd / "out"
    co.mkdir(parents=True, exist_ok=True)
    pull(f"{dc}/out_l/Result_0/penultimate_hidden.raw", co / "cl.raw")
    pull(f"{dc}/out_g/Result_0/penultimate_hidden.raw", co / "cg.raw")
    pull(f"{dc}/out_g/Result_0/text_embeds.raw", co / "te.raw")

    cl = np.fromfile(str(co / "cl.raw"), np.float32).reshape(1, 77, 768)
    cg = np.fromfile(str(co / "cg.raw"), np.float32).reshape(1, 77, 1280)
    te = np.fromfile(str(co / "te.raw"), np.float32).reshape(1, 1280)
    pe = np.concatenate([cl, cg], axis=-1)  # [1,77,2048]

    # ── 3. Biases ──
    print("[Biases]", end=" ", flush=True)
    t0 = time.time()
    pe_t = torch.from_numpy(pe).to(dev, torch.float32)
    te_t = torch.from_numpy(te).to(dev, torch.float32)
    tid = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32, device=dev)

    _scheduler.set_timesteps(steps, device=dev)
    timesteps = _scheduler.timesteps
    dummy = torch.zeros(1, 4, 128, 128, device=dev, dtype=torch.float32)

    all_biases = {}
    for si, t in enumerate(timesteps):
        with torch.no_grad():
            bts = compute_external_resnet_biases(
                _unet, dummy,
                torch.tensor([t.item()], dtype=torch.float32, device=dev),
                pe_t, te_t, tid,
            )
        biases = []
        for (nm, _), bt in zip(_resnet_modules, bts):
            h, w = _spatial_shapes[nm]
            biases.append(bt.expand(-1, -1, h, w).detach().cpu().numpy())
        all_biases[si] = biases
    print(f"{(time.time()-t0)*1000:.0f}ms")

    # ── 4. Push biases + enc_hs ──
    print("[Push]", end=" ", flush=True)
    t0 = time.time()
    enc = np.transpose(pe, (0, 2, 1)).astype(np.float32)
    enc.tofile(str(WORK / "enc.raw"))
    push(WORK / "enc.raw", f"{dw}/enc.raw")

    for si in range(steps):
        sd = WORK / f"s{si:02d}"
        sd.mkdir(parents=True, exist_ok=True)
        dd = f"{dw}/s{si:02d}"
        adb("shell", f"mkdir -p {dd}")
        for i, b in enumerate(all_biases[si]):
            fn = f"b{i:02d}.raw"
            np.transpose(b, (0, 2, 3, 1)).astype(np.float32).tofile(str(sd / fn))
            push(sd / fn, f"{dd}/{fn}")
    print(f"{(time.time()-t0)*1000:.0f}ms")

    # ── 5. Denoise ──
    gen = torch.Generator(device=dev).manual_seed(seed)
    latents = torch.randn((1, 4, 128, 128), generator=gen, device=dev, dtype=dtype)
    latents = latents * _scheduler.init_noise_sigma
    total_unet = 0

    for si, t in enumerate(timesteps):
        sd = WORK / f"s{si:02d}"
        dd = f"{dw}/s{si:02d}"

        lat_in = _scheduler.scale_model_input(latents, t)
        smp = lat_in.detach().cpu().float().numpy()
        with open(sd / "smp.raw", "wb") as f:
            f.write(nhwc_f32(smp))
        push(sd / "smp.raw", f"{dd}/smp.raw")

        bfs = [f"b{i:02d}.raw" for i in range(len(all_biases[si]))]
        entries = [f"{dd}/smp.raw", f"{dw}/enc.raw"] + [f"{dd}/{bf}" for bf in bfs]
        with open(sd / "il.txt", "w") as f:
            f.write(" ".join(entries) + "\n")
        push(sd / "il.txt", f"{dd}/il.txt")

        print(f"  [UNet {si+1}/{steps}]", end=" ", flush=True)
        ms = qnn_run(f"{DR}/context/unet_lightning8step.serialized.bin.bin",
                     f"{dd}/il.txt", f"{dd}/out")
        total_unet += ms
        print(f"{ms:.0f}ms", end="", flush=True)

        ro = sd / "out" / "Result_0"
        ro.mkdir(parents=True, exist_ok=True)
        pulled_unet = pull_first_remote_raw(
            f"{dd}/out/Result_0",
            ro / "np.raw",
            ("np.raw", "noise_pred.raw", "noise_pred_native.raw"),
        )
        d = np.fromfile(str(ro / "np.raw"), np.float32)
        expected_elems = 1 * 128 * 128 * 4
        assert d.size == expected_elems, (
            f"Unexpected size for {ro / 'np.raw'}: expected {expected_elems} float32 elements, got {d.size}"
        )

        d = np.fromfile(str(ro / "np.raw"), np.float32)
        np_arr = np.transpose(d.reshape(1, 128, 128, 4), (0, 3, 1, 2))
        print(f" [{np_arr.min():.2f}..{np_arr.max():.2f}] ({pulled_unet})")

        npt = torch.from_numpy(np_arr).to(device=dev, dtype=dtype)
        latents = _scheduler.step(npt, t, latents).prev_sample

    print(f"  UNet total: {total_unet:.0f}ms ({total_unet/steps:.0f}ms/step)")

    # ── 6. VAE ──
    print("[VAE]", end=" ", flush=True)
    vae_cfg = json.loads((DIFFUSERS_DIR / "vae" / "config.json").read_text()) \
        if (DIFFUSERS_DIR / "vae" / "config.json").exists() else {"scaling_factor": 0.13025}
    sf = vae_cfg.get("scaling_factor", 0.13025)
    lat_sc = latents.float().cpu().numpy() / sf

    vd = WORK / "vae"
    vd.mkdir(parents=True, exist_ok=True)
    with open(vd / "lat.raw", "wb") as f:
        f.write(nhwc_f32(lat_sc))
    dv = f"{dw}/vae"
    adb("shell", f"mkdir -p {dv}")
    push(vd / "lat.raw", f"{dv}/lat.raw")
    with open(vd / "il.txt", "w") as f:
        f.write(f"{dv}/lat.raw\n")
    push(vd / "il.txt", f"{dv}/il.txt")

    ms_vae = qnn_run(f"{DR}/context/vae_decoder.serialized.bin.bin",
                     f"{dv}/il.txt", f"{dv}/out", native=True)
    print(f"{ms_vae:.0f}ms")

    vo = vd / "out" / "Result_0"
    vo.mkdir(parents=True, exist_ok=True)
    pulled_vae = pull_first_remote_raw(
        f"{dv}/out/Result_0",
        vo / "img.raw",
        ("img.raw", "image.raw", "image_native.raw"),
    )
    raw = np.fromfile(str(vo / "img.raw"), np.float16).astype(np.float32)
    expected_elems = 1024 * 1024 * 3
    if raw.size != expected_elems:
        raise ValueError(
            f"VAE output size mismatch for {vo / 'img.raw'}: "
            f"expected {expected_elems} elements for shape (1024, 1024, 3), got {raw.size}"
        )

    raw = np.fromfile(str(vo / "img.raw"), np.float16).astype(np.float32)
    img = raw.reshape(1024, 1024, 3)
    img = np.clip(img / 2 + 0.5, 0, 1)

    if stretch:
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi - lo > 0.05:
            img = np.clip((img - lo) / (hi - lo), 0, 1)

    img_u8 = (img * 255).astype(np.uint8)

    from PIL import Image
    tag = name or f"gen_s{seed}"
    out_path = OUTPUT_DIR / f"{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(str(out_path))

    elapsed = time.time() - t_total
    print(f"\n{'='*40}")
    print(f"Saved: {out_path}")
    print(f"VAE source raw: {pulled_vae}")
    print(f"CLIP: {ms_l+ms_g:.0f}ms | UNet: {total_unet:.0f}ms | VAE: {ms_vae:.0f}ms")
    print(f"Total: {elapsed:.1f}s")
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL Lightning on Phone NPU")
    ap.add_argument("prompt", type=str, help="Text prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no-stretch", action="store_true", help="Disable contrast stretch")
    ap.add_argument("--name", type=str, default=None, help="Output filename (without .png)")
    a = ap.parse_args()
    generate(a.prompt, a.seed, a.steps, stretch=not a.no_stretch, name=a.name)
