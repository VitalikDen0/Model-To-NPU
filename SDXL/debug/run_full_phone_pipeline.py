#!/usr/bin/env python3
"""
Full phone SDXL Lightning pipeline: CLIP-L + CLIP-G + UNet + VAE all on NPU.
PC acts as orchestrator (tokenization, scheduling, bias computation).

Usage:
  python NPU/run_full_phone_pipeline.py --prompt "1girl, upper body, masterpiece"
  python NPU/run_full_phone_pipeline.py --prompt "landscape" --steps 8 --seed 123
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SDXL_NPU = ROOT / "sdxl_npu"
if str(SDXL_NPU) not in sys.path:
    sys.path.insert(0, str(SDXL_NPU))

from export_sdxl_to_onnx import (
    collect_unet_resnet_conditioning_modules,
    compute_external_resnet_biases,
    infer_unet_resnet_spatial_shapes,
)

DIFFUSERS_DIR = SDXL_NPU / "diffusers_pipeline"
MERGED_UNET_DIR = SDXL_NPU / "unet_lightning8step_merged"
ADB = str(ROOT / "adb.exe")
DEVICE_ROOT = os.environ.get("SDXL_QNN_BASE", "/sdcard/Download/sdxl_qnn")
OUTPUT_DIR = ROOT / "NPU" / "outputs"

# Context binaries on phone
CTX_CLIP_L = f"{DEVICE_ROOT}/context/clip_l.serialized.bin.bin"
CTX_CLIP_G = f"{DEVICE_ROOT}/context/clip_g.serialized.bin.bin"
CTX_UNET = f"{DEVICE_ROOT}/context/unet_lightning8step.serialized.bin.bin"
CTX_VAE = f"{DEVICE_ROOT}/context/vae_decoder.serialized.bin.bin"
HTP_EXT = f"{DEVICE_ROOT}/htp_backend_extensions_lightning.json"

ENV_CMD = (
    f"cd {DEVICE_ROOT} && "
    f"export LD_LIBRARY_PATH={DEVICE_ROOT}/lib:{DEVICE_ROOT}/bin:{DEVICE_ROOT}/model && "
    f"export ADSP_LIBRARY_PATH='{DEVICE_ROOT}/lib;/vendor/lib64/rfs/dsp;"
    f"/vendor/lib/rfsa/adsp;/vendor/dsp' && "
)


def adb(*args, check=True, capture=False):
    cmd = [ADB] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if check and r.returncode != 0:
            raise RuntimeError(f"adb failed: {r.stderr}")
        return r.stdout.strip()
    else:
        subprocess.run(cmd, check=check, timeout=600)


def push_file(local_path, device_path):
    adb("push", str(local_path), device_path)


def pull_file(device_path, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    adb("pull", device_path, str(local_path))


def nchw_to_nhwc_f32(arr):
    """NCHW → NHWC float32 bytes."""
    return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32).tobytes()


def run_qnn(context_bin, input_list_device, output_dir_device,
            use_native=False, perf="burst"):
    """Run qnn-net-run on phone with context binary."""
    native_flag = "--use_native_output_files " if use_native else ""
    cmd = (
        f"{ENV_CMD}"
        f"mkdir -p {output_dir_device} && "
        f"{DEVICE_ROOT}/bin/qnn-net-run "
        f"--retrieve_context {context_bin} "
        f"--backend {DEVICE_ROOT}/lib/libQnnHtp.so "
        f"--input_list {input_list_device} "
        f"--output_dir {output_dir_device} "
        f"{native_flag}"
        f"--perf_profile {perf} --log_level warn"
    )
    t0 = time.time()
    adb("shell", cmd)
    return (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str,
                    default="1girl, upper body, looking at viewer, masterpiece, best quality")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    latent_h, latent_w = args.height // 8, args.width // 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    work = ROOT / "NPU" / "runtime_work_full"
    work.mkdir(parents=True, exist_ok=True)
    device_work = f"{DEVICE_ROOT}/runtime_work_full"
    timings = {}

    # ═══════════════════════════════════════════════
    # 1. Tokenize on PC
    # ═══════════════════════════════════════════════
    print("[1] Tokenizing prompt...")
    from transformers import CLIPTokenizer
    tok_l = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer"))
    tok_g = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer_2"))

    ids_l = tok_l(args.prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="np")["input_ids"]  # [1,77] int64
    ids_g = tok_g(args.prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="np")["input_ids"]  # [1,77] int64

    # QNN expects float32 values for token IDs
    ids_l_f32 = ids_l.astype(np.float32)
    ids_g_f32 = ids_g.astype(np.float32)

    clip_dir = work / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    ids_l_f32.tofile(str(clip_dir / "input_ids_l.raw"))
    ids_g_f32.tofile(str(clip_dir / "input_ids_g.raw"))

    # Write input lists
    dev_clip = f"{device_work}/clip"
    with open(clip_dir / "input_list_l.txt", "w") as f:
        f.write(f"{dev_clip}/input_ids_l.raw\n")
    with open(clip_dir / "input_list_g.txt", "w") as f:
        f.write(f"{dev_clip}/input_ids_g.raw\n")

    print(f"  CLIP-L tokens: {ids_l[0,:5]}... CLIP-G tokens: {ids_g[0,:5]}...")

    # ═══════════════════════════════════════════════
    # 2. Push token files & run CLIP on phone NPU
    # ═══════════════════════════════════════════════
    print("[2] Running CLIP-L on phone NPU...")
    adb("shell", f"mkdir -p {dev_clip}")
    for f in ["input_ids_l.raw", "input_ids_g.raw", "input_list_l.txt", "input_list_g.txt"]:
        push_file(clip_dir / f, f"{dev_clip}/{f}")

    ms = run_qnn(CTX_CLIP_L, f"{dev_clip}/input_list_l.txt",
                 f"{dev_clip}/output_l")
    timings["clip_l"] = ms
    print(f"  CLIP-L: {ms:.0f}ms")

    print("[2] Running CLIP-G on phone NPU...")
    ms = run_qnn(CTX_CLIP_G, f"{dev_clip}/input_list_g.txt",
                 f"{dev_clip}/output_g")
    timings["clip_g"] = ms
    print(f"  CLIP-G: {ms:.0f}ms")

    # Pull CLIP outputs
    clip_out = clip_dir / "output"
    clip_out.mkdir(parents=True, exist_ok=True)
    pull_file(f"{dev_clip}/output_l/Result_0/penultimate_hidden.raw",
              clip_out / "clip_l_hidden.raw")
    pull_file(f"{dev_clip}/output_g/Result_0/penultimate_hidden.raw",
              clip_out / "clip_g_hidden.raw")
    pull_file(f"{dev_clip}/output_g/Result_0/text_embeds.raw",
              clip_out / "text_embeds.raw")

    # Parse CLIP outputs
    clip_l_hidden = np.fromfile(str(clip_out / "clip_l_hidden.raw"), dtype=np.float32)
    clip_l_hidden = clip_l_hidden.reshape(1, 77, 768)
    clip_g_hidden = np.fromfile(str(clip_out / "clip_g_hidden.raw"), dtype=np.float32)
    clip_g_hidden = clip_g_hidden.reshape(1, 77, 1280)
    text_embeds_np = np.fromfile(str(clip_out / "text_embeds.raw"), dtype=np.float32)
    text_embeds_np = text_embeds_np.reshape(1, 1280)

    # Concatenate to form prompt_embeds [1, 77, 2048]
    prompt_embeds = np.concatenate([clip_l_hidden, clip_g_hidden], axis=-1)
    print(f"  prompt_embeds: {prompt_embeds.shape}, text_embeds: {text_embeds_np.shape}")
    print(f"  prompt_embeds range: [{prompt_embeds.min():.4f}, {prompt_embeds.max():.4f}]")

    # ═══════════════════════════════════════════════
    # 3. Compute extmaps biases on PC
    # ═══════════════════════════════════════════════
    print("[3] Computing extmaps biases on PC...")
    from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
    lightning_unet = UNet2DConditionModel.from_pretrained(
        str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    lightning_unet.eval()

    # Scheduler
    sched_config = json.loads((DIFFUSERS_DIR / "scheduler" / "scheduler_config.json").read_text())
    scheduler = EulerDiscreteScheduler.from_config(sched_config, timestep_spacing="trailing")
    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps
    print(f"  {args.steps} steps, timesteps: {timesteps.tolist()}")

    # Collect resnet modules
    resnet_modules = collect_unet_resnet_conditioning_modules(lightning_unet)
    spatial_shapes = infer_unet_resnet_spatial_shapes(lightning_unet, latent_h, latent_w)
    print(f"  {len(resnet_modules)} resnet bias modules")

    # Prepare fixed conditioning
    prompt_embeds_t = torch.from_numpy(prompt_embeds).to(device, torch.float32)
    text_embeds_t = torch.from_numpy(text_embeds_np).to(device, torch.float32)
    time_ids = torch.tensor(
        [[args.height, args.width, 0, 0, args.height, args.width]],
        dtype=torch.float32, device=device
    )

    # Pre-compute biases for ALL timesteps
    t0 = time.time()
    all_biases = {}
    for step_idx, t in enumerate(timesteps):
        dummy_sample = torch.zeros(1, 4, latent_h, latent_w, device=device, dtype=torch.float32)
        with torch.no_grad():
            bias_tensors = compute_external_resnet_biases(
                lightning_unet, dummy_sample, torch.tensor([t.item()], dtype=torch.float32, device=device),
                prompt_embeds_t, text_embeds_t, time_ids,
            )
        biases = []
        for (name, _), bt in zip(resnet_modules, bias_tensors):
            h, w = spatial_shapes[name]
            biases.append(bt.expand(-1, -1, h, w).detach().cpu().numpy())
        all_biases[step_idx] = biases
    bias_ms = (time.time() - t0) * 1000
    timings["bias_compute"] = bias_ms
    print(f"  Biases computed: {bias_ms:.0f}ms for all {args.steps} steps")

    # Free UNet from memory
    del lightning_unet
    if device == "cuda":
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════
    # 4. Push pre-computed data to phone
    # ═══════════════════════════════════════════════
    print("[4] Pushing pre-computed data to phone...")
    t0 = time.time()

    # encoder_hidden_states: [1,77,2048] → QNN NFC [1,2048,77]
    enc_hs_qnn = np.transpose(prompt_embeds, (0, 2, 1)).astype(np.float32)
    enc_hs_path = work / "encoder_hidden_states.qnn.raw"
    enc_hs_qnn.tofile(str(enc_hs_path))
    push_file(enc_hs_path, f"{device_work}/encoder_hidden_states.qnn.raw")

    # Push biases for all steps
    for step_idx in range(args.steps):
        step_dev = f"{device_work}/step_{step_idx:03d}"
        step_local = work / f"step_{step_idx:03d}"
        step_local.mkdir(parents=True, exist_ok=True)
        adb("shell", f"mkdir -p {step_dev}")

        for i, bias_np in enumerate(all_biases[step_idx]):
            fname = f"resnet_bias_{i:02d}.qnn.raw"
            bias_nhwc = np.transpose(bias_np, (0, 2, 3, 1)).astype(np.float32)
            bias_nhwc.tofile(str(step_local / fname))
            push_file(step_local / fname, f"{step_dev}/{fname}")

    push_ms = (time.time() - t0) * 1000
    timings["push_biases"] = push_ms
    print(f"  Push complete: {push_ms:.0f}ms")

    # ═══════════════════════════════════════════════
    # 5. Denoise loop: UNet on phone NPU
    # ═══════════════════════════════════════════════
    print("[5] Starting denoising loop...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator,
                          device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    total_unet_ms = 0
    for step_idx, t in enumerate(timesteps):
        step_local = work / f"step_{step_idx:03d}"
        step_dev = f"{device_work}/step_{step_idx:03d}"

        print(f"  [step {step_idx}/{args.steps}] t={t.item():.1f}")

        # Scale input
        latents_in = scheduler.scale_model_input(latents, t)
        sample_np = latents_in.detach().cpu().float().numpy()

        # Write and push sample (NCHW → NHWC)
        sample_path = step_local / "sample.qnn.raw"
        sample_bytes = nchw_to_nhwc_f32(sample_np)
        with open(sample_path, "wb") as f:
            f.write(sample_bytes)
        push_file(sample_path, f"{step_dev}/sample.qnn.raw")

        # Build input_list for this step
        bias_files = [f"resnet_bias_{i:02d}.qnn.raw" for i in range(len(all_biases[step_idx]))]
        input_entries = [
            f"{step_dev}/sample.qnn.raw",
            f"{device_work}/encoder_hidden_states.qnn.raw",
        ] + [f"{step_dev}/{bf}" for bf in bias_files]

        input_list_path = step_local / "input_list.txt"
        with open(input_list_path, "w") as f:
            f.write(" ".join(input_entries) + "\n")
        push_file(input_list_path, f"{step_dev}/input_list.txt")

        # Run UNet
        out_dev = f"{step_dev}/output"
        ms = run_qnn(CTX_UNET, f"{step_dev}/input_list.txt", out_dev,
                     perf="burst")
        total_unet_ms += ms
        print(f"    UNet NPU: {ms:.0f}ms")

        # Pull & decode output
        result_local = step_local / "output" / "Result_0"
        result_local.mkdir(parents=True, exist_ok=True)
        pull_file(f"{out_dev}/Result_0/noise_pred.raw",
                  result_local / "noise_pred.raw")

        data = np.fromfile(str(result_local / "noise_pred.raw"), dtype=np.float32)
        expected = latent_h * latent_w * 4
        if data.size != expected:
            print(f"    [ERROR] Unexpected output: {data.size} vs {expected}")
            sys.exit(1)
        noise_pred = np.transpose(data.reshape(1, latent_h, latent_w, 4), (0, 3, 1, 2))
        print(f"    noise_pred: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]")

        # Scheduler step
        noise_pred_t = torch.from_numpy(noise_pred).to(device=device, dtype=dtype)
        latents = scheduler.step(noise_pred_t, t, latents).prev_sample
        print(f"    latents: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

    timings["unet_total"] = total_unet_ms
    timings["unet_avg"] = total_unet_ms / args.steps
    print(f"  UNet total: {total_unet_ms:.0f}ms ({total_unet_ms/args.steps:.0f}ms/step)")

    # ═══════════════════════════════════════════════
    # 6. VAE decode on phone NPU
    # ═══════════════════════════════════════════════
    print("[6] Running VAE decode on phone NPU...")

    # Scale latents
    vae_config = json.loads((DIFFUSERS_DIR / "vae" / "config.json").read_text()
                            if (DIFFUSERS_DIR / "vae" / "config.json").exists()
                            else '{"scaling_factor": 0.13025}')
    scaling_factor = vae_config.get("scaling_factor", 0.13025)
    latents_scaled = latents.float().cpu().numpy() / scaling_factor

    # Write VAE input (NCHW → NHWC float32)
    vae_dir = work / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae_input = nchw_to_nhwc_f32(latents_scaled)
    with open(vae_dir / "latent.raw", "wb") as f:
        f.write(vae_input)

    dev_vae = f"{device_work}/vae"
    adb("shell", f"mkdir -p {dev_vae}")
    push_file(vae_dir / "latent.raw", f"{dev_vae}/latent.raw")

    with open(vae_dir / "input_list.txt", "w") as f:
        f.write(f"{dev_vae}/latent.raw\n")
    push_file(vae_dir / "input_list.txt", f"{dev_vae}/input_list.txt")

    ms = run_qnn(CTX_VAE, f"{dev_vae}/input_list.txt", f"{dev_vae}/output",
                 use_native=True, perf="burst")
    timings["vae"] = ms
    print(f"  VAE NPU: {ms:.0f}ms")

    # Pull VAE output
    vae_out = vae_dir / "output" / "Result_0"
    vae_out.mkdir(parents=True, exist_ok=True)

    # Check what output files exist
    ls_out = adb("shell", f"ls {dev_vae}/output/Result_0/", capture=True)
    print(f"  VAE output files: {ls_out}")

    # Try native output first, then float
    native_path = None
    for candidate in ["image_native.raw", "image.raw"]:
        check = adb("shell", f"ls -la {dev_vae}/output/Result_0/{candidate} 2>/dev/null",
                     capture=True, check=False)
        if check and "No such file" not in check:
            native_path = candidate
            break

    if native_path:
        pull_file(f"{dev_vae}/output/Result_0/{native_path}",
                  vae_out / native_path)
        raw_data = np.fromfile(str(vae_out / native_path), dtype=np.uint8)
        raw_bytes = len(raw_data)
        print(f"  VAE raw output: {raw_bytes} bytes ({native_path})")

        # Decode based on size
        pixels = args.height * args.width * 3
        if raw_bytes == pixels * 2:
            # FP16 NHWC [1, H, W, 3]
            image_fp16 = np.frombuffer(raw_data.tobytes(), dtype=np.float16)
            image_f32 = image_fp16.astype(np.float32).reshape(1, args.height, args.width, 3)
        elif raw_bytes == pixels * 4:
            # FP32 NHWC
            image_f32 = np.frombuffer(raw_data.tobytes(), dtype=np.float32)
            image_f32 = image_f32.reshape(1, args.height, args.width, 3)
        else:
            # Try to read sidecar JSON
            json_candidates = [native_path.replace(".raw", ".json"),
                               native_path.replace("_native.raw", ".json")]
            for jc in json_candidates:
                jp = f"{dev_vae}/output/Result_0/{jc}"
                try:
                    jdata = adb("shell", f"cat {jp}", capture=True, check=False)
                    if jdata:
                        meta = json.loads(jdata)
                        print(f"  VAE sidecar: {meta}")
                        break
                except Exception:
                    pass
            print(f"  [WARN] Unexpected VAE output size: {raw_bytes}")
            # Attempt FP16 interpretation
            image_fp16 = np.frombuffer(raw_data.tobytes(), dtype=np.float16)
            total_pixels = image_fp16.size
            side = int((total_pixels / 3) ** 0.5)
            image_f32 = image_fp16.astype(np.float32).reshape(1, side, side, 3)

        # Convert to uint8 image
        img = image_f32[0]
        # SDXL VAE output is in [-1, 1], rescale to [0, 1]
        img = (img / 2 + 0.5)
        img = np.clip(img, 0, 1)
        # FP16 VAE may compress dynamic range — apply contrast stretching
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi - lo > 0.05:
            img = (img - lo) / (hi - lo)
            img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
    else:
        print("  [ERROR] No VAE output found!")
        sys.exit(1)

    # ═══════════════════════════════════════════════
    # 7. Save image
    # ═══════════════════════════════════════════════
    from PIL import Image
    pil_img = Image.fromarray(img)

    if args.output and (os.sep in args.output or '/' in args.output):
        out_path = Path(args.output)
    else:
        out_name = args.output or f"full_phone_s{args.steps}_seed{args.seed}.png"
        out_path = OUTPUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_path))

    # Save timings
    timings_path = out_path.with_suffix(".timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)

    total_ms = sum(v for k, v in timings.items() if k != "unet_avg")
    print(f"\n{'='*50}")
    print(f"[DONE] Saved: {out_path}")
    print(f"  CLIP-L:    {timings['clip_l']:.0f}ms")
    print(f"  CLIP-G:    {timings['clip_g']:.0f}ms")
    print(f"  Biases:    {timings['bias_compute']:.0f}ms (PC)")
    print(f"  Push:      {timings['push_biases']:.0f}ms")
    print(f"  UNet:      {timings['unet_total']:.0f}ms ({timings['unet_avg']:.0f}ms/step)")
    print(f"  VAE:       {timings['vae']:.0f}ms")
    print(f"  Total:     {total_ms:.0f}ms ({total_ms/1000:.1f}s)")
    print(f"  Image:     {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
