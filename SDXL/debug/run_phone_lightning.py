#!/usr/bin/env python3
"""
Phone-side Lightning UNet runner: 8 steps, CFG=0, single UNet call per step.

Usage:
  python NPU/run_phone_lightning.py --prompt "1girl, upper body, masterpiece"
  python NPU/run_phone_lightning.py --prompt "landscape" --steps 8
"""
import argparse, json, os, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PLATFORM_TOOLS = ROOT.parent
DEFAULT_SDXL_NPU = EXTERNAL_PLATFORM_TOOLS / "sdxl_npu"
SDXL_NPU = DEFAULT_SDXL_NPU if DEFAULT_SDXL_NPU.exists() else (ROOT / "sdxl_npu")
if str(SDXL_NPU) not in sys.path:
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
]
ADB = str(next((p for p in ADB_CANDIDATES if p.exists()), ADB_CANDIDATES[0]))
OUTPUT_DIR = ROOT / "NPU" / "outputs"


def _default_device_root() -> str:
    return os.environ.get("SDXL_PHONE_DEVICE_ROOT", "/data/local/tmp/sdxl_qnn")


def _default_context_bin(device_root: str) -> str:
    return os.environ.get(
        "SDXL_PHONE_UNET_CONTEXT",
        f"{device_root}/context/unet_lightning8step.serialized.bin.bin",
    )


def _default_config_file(device_root: str) -> str:
    return os.environ.get(
        "SDXL_PHONE_QNN_CONFIG",
        f"{device_root}/htp_backend_extensions_lightning.json",
    )

# ── helpers ──

def adb(*args, check=True, capture=False):
    cmd = [ADB] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if check and r.returncode != 0:
            raise RuntimeError(f"adb failed: {r.stderr}")
        return r.stdout.strip()
    else:
        subprocess.run(cmd, check=check, timeout=300)


def to_raw(arr: np.ndarray) -> bytes:
    return arr.tobytes()


def to_qnn_input(arr_nchw: np.ndarray) -> bytes:
    """NCHW float16/32 → NHWC float32 for qnn-net-run."""
    if arr_nchw.ndim == 4:
        arr_nhwc = np.transpose(arr_nchw, (0, 2, 3, 1))
    else:
        arr_nhwc = arr_nchw
    return arr_nhwc.astype(np.float32).tobytes()


def build_phone_env_cmd(device_root: str) -> str:
    return (
        f"cd {device_root} && "
        f"export LD_LIBRARY_PATH={device_root}/lib:{device_root}/bin:{device_root}/model && "
        f"export ADSP_LIBRARY_PATH='{device_root}/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp' && "
    )


def decode_native_output(raw_path: Path, json_path: Path) -> np.ndarray:
    """Decode qnn-net-run native output using sidecar JSON."""
    with open(json_path) as f:
        meta = json.load(f)
    dims = meta["Dimensions"]
    dtype_str = meta["Datatype"]
    qparams = meta.get("QuantaziationParams", meta.get("QuantizationParams", {}))
    scale = qparams.get("Scale", 1.0)
    offset = qparams.get("Offset", 0)
    data = np.fromfile(raw_path, dtype=np.uint8)
    if "UFIXED_POINT_8" in dtype_str:
        arr = (data.astype(np.float32) + offset) * scale
    elif "UFIXED_POINT_16" in dtype_str:
        arr = np.frombuffer(data.tobytes(), dtype=np.uint16).astype(np.float32)
        arr = (arr + offset) * scale
    elif "FLOAT_16" in dtype_str:
        arr = np.frombuffer(data.tobytes(), dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(data.tobytes(), dtype=np.float32)
    arr = arr.reshape(dims)  # NHWC
    if len(dims) == 4:
        arr = np.transpose(arr, (0, 3, 1, 2))  # NCHW
    return arr


def decode_float_output(raw_path: Path, expected_shape=(1, 4, 128, 128)) -> np.ndarray:
    """Fallback: decode float32 output from qnn-net-run."""
    data = np.fromfile(raw_path, dtype=np.float32)
    n = int(np.prod(expected_shape))
    if data.size == n:
        return data.reshape(expected_shape)
    # half-width fallback
    nhwc_shape = (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
    if data.size == int(np.prod(nhwc_shape)):
        return np.transpose(data.reshape(nhwc_shape), (0, 3, 1, 2))
    raise ValueError(f"Cannot decode output: {data.size} elements, expected {n}")


def compute_step_inputs(unet, latents_in, t_value, encoder_hidden_states,
                        text_embeds, time_ids, spatial_shapes, resnet_modules, device):
    """Compute extmaps bias tensors for a single step."""
    sample_t = latents_in.to(device, dtype=torch.float32)
    timestep_t = torch.tensor([t_value], dtype=torch.float32, device=device)
    enc_t = encoder_hidden_states.to(device, dtype=torch.float32)
    text_t = text_embeds.to(device, dtype=torch.float32)
    time_ids_t = time_ids.to(device, dtype=torch.float32)

    with torch.no_grad():
        bias_tensors = compute_external_resnet_biases(
            unet, sample_t, timestep_t, enc_t, text_t, time_ids_t,
        )
    # Expand biases to spatial dimensions
    biases = []
    for (module_name, _), bias_tensor in zip(resnet_modules, bias_tensors):
        h, w = spatial_shapes[module_name]
        biases.append(bias_tensor.expand(-1, -1, h, w).detach().cpu().numpy())
    return biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="1girl, upper body, looking at viewer, masterpiece, best quality")
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device-root", type=str, default=_default_device_root())
    ap.add_argument("--context-bin", type=str, default=None)
    ap.add_argument("--config-file", type=str, default=None)
    ap.add_argument("--perf-profile", type=str, default=os.environ.get("SDXL_PHONE_QNN_PERF", "burst"))
    ap.add_argument("--use-mmap", action="store_true")
    ap.add_argument("--work-subdir", type=str, default="runtime_work_lightning")
    args = ap.parse_args()

    device_root = args.device_root
    context_bin = args.context_bin or _default_context_bin(device_root)
    config_file = args.config_file or _default_config_file(device_root)
    env_cmd = build_phone_env_cmd(device_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[init] device={device}, dtype={dtype}")

    # Load pipeline for CLIP encoding and VAE decode
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    print("[init] Loading pipeline for CLIP/VAE...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(DIFFUSERS_DIR), torch_dtype=dtype, local_files_only=True
    ).to(device)

    # Load Lightning-merged UNet for extmaps bias computation
    from diffusers import UNet2DConditionModel
    print("[init] Loading Lightning-merged UNet for bias computation...")
    lightning_unet = UNet2DConditionModel.from_pretrained(
        str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    lightning_unet.eval()

    # Set scheduler to EulerDiscrete with trailing timestep spacing (Lightning requirement)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    # Collect resnet modules + spatial shapes for extmaps
    resnet_modules = collect_unet_resnet_conditioning_modules(lightning_unet)
    latent_h, latent_w = args.height // 8, args.width // 8
    spatial_shapes = infer_unet_resnet_spatial_shapes(lightning_unet, latent_h, latent_w)
    print(f"[extmaps] {len(resnet_modules)} resnet bias modules")

    # CLIP encode
    print("[clip] Encoding prompt...")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,  # Lightning: no CFG
        )
    print(f"  prompt_embeds: {prompt_embeds.shape}")
    print(f"  pooled_prompt_embeds: {pooled_prompt_embeds.shape}")

    # Prepare time_ids
    add_time_ids = pipe._get_add_time_ids(
        original_size=(args.height, args.width),
        crops_coords_top_left=(0, 0),
        target_size=(args.height, args.width),
        dtype=dtype,
        text_encoder_projection_dim=1280,
    ).to(device)

    # Numpy versions for passing to bias computation
    enc_hs_np = prompt_embeds.detach().cpu().float().numpy()
    text_embeds_np = pooled_prompt_embeds.detach().cpu().float().numpy()
    time_ids_np = add_time_ids.detach().cpu().float().numpy()

    # Initialize latents
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, device=device, dtype=dtype)
    latents = latents * pipe.scheduler.init_noise_sigma

    # Set timesteps
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    print(f"[scheduler] {args.steps} steps, timesteps: {timesteps.tolist()}")

    # Prepare device-side work directory
    work_dir = ROOT / "NPU" / args.work_subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    device_work_root = f"{device_root}/{args.work_subdir}"

    total_phone_ms = 0

    for step_idx, t in enumerate(timesteps):
        step_dir = work_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[step {step_idx}/{len(timesteps)}] t={t.item():.1f}")

        # Scale input
        latents_in = pipe.scheduler.scale_model_input(latents, t)
        sample_np = latents_in.detach().cpu().float().numpy()

        # Compute external resnet biases
        biases = compute_step_inputs(
            lightning_unet, latents_in, t.item(), prompt_embeds,
            pooled_prompt_embeds, add_time_ids, spatial_shapes, resnet_modules, device,
        )

        # Write inputs as NHWC float32 raw files
        # Write sample (spatial: NCHW → NHWC)
        with open(step_dir / "sample.qnn.raw", "wb") as f:
            f.write(to_qnn_input(sample_np))
        # Write encoder_hidden_states (3D tensor: QNN transposed NCF [1,77,2048] → NFC [1,2048,77])
        enc_hs_qnn = np.transpose(enc_hs_np, (0, 2, 1)).astype(np.float32)
        with open(step_dir / "encoder_hidden_states.qnn.raw", "wb") as f:
            f.write(enc_hs_qnn.tobytes())
        # Write bias maps (spatial: NCHW → NHWC)
        bias_files = []
        for i, bias_np in enumerate(biases):
            fname = f"resnet_bias_{i:02d}.qnn.raw"
            with open(step_dir / fname, "wb") as f:
                f.write(to_qnn_input(bias_np))
            bias_files.append(fname)

        # Build input_list.txt
        device_step = f"{device_work_root}/step_{step_idx:03d}"
        input_names = ["sample.qnn.raw", "encoder_hidden_states.qnn.raw"] + bias_files
        input_line = " ".join(f"{device_step}/{n}" for n in input_names)
        with open(step_dir / "input_list.txt", "w") as f:
            f.write(input_line + "\n")

        # Push inputs to phone
        adb("shell", f"mkdir -p {device_step}")
        for fname in input_names + ["input_list.txt"]:
            adb("push", str(step_dir / fname), f"{device_step}/{fname}")

        # Run UNet on phone
        device_out = f"{device_step}/output"
        config_flag = f" --config_file {config_file}" if config_file else ""
        mmap_flag = " --use_mmap" if args.use_mmap else ""
        run_cmd = (
            f"{env_cmd}"
            f"mkdir -p {device_out} && "
            f"{device_root}/bin/qnn-net-run "
            f"--retrieve_context {context_bin} "
            f"--backend {device_root}/lib/libQnnHtp.so "
            f"--input_list {device_step}/input_list.txt "
            f"--output_dir {device_out} "
            f"--perf_profile {args.perf_profile}{config_flag}{mmap_flag} --log_level warn"
        )

        t0 = time.time()
        adb("shell", run_cmd)
        phone_ms = (time.time() - t0) * 1000
        total_phone_ms += phone_ms
        print(f"  phone UNet: {phone_ms:.0f}ms")

        # Pull output
        result_dir = step_dir / "output" / "Result_0"
        result_dir.mkdir(parents=True, exist_ok=True)

        float_raw = f"{device_out}/Result_0/noise_pred.raw"
        adb("pull", float_raw, str(result_dir / "noise_pred.raw"), check=False)

        # Decode output (float32, NHWC → NCHW)
        float_raw_path = result_dir / "noise_pred.raw"
        noise_pred = None
        if float_raw_path.exists():
            data = np.fromfile(str(float_raw_path), dtype=np.float32)
            expected = latent_h * latent_w * 4  # [1, H, W, 4] NHWC
            if data.size == expected:
                noise_pred = np.transpose(data.reshape(1, latent_h, latent_w, 4), (0, 3, 1, 2))
                print(f"  decoded float output: {noise_pred.shape}, range=[{noise_pred.min():.4f}, {noise_pred.max():.4f}]")
            else:
                print(f"  unexpected output size: {data.size} elements, expected {expected}")
        if noise_pred is None:
            print("  [ERROR] No valid output from phone!")
            sys.exit(1)

        # Scheduler step (Lightning: guidance_scale=0, no CFG)
        noise_pred_torch = torch.from_numpy(noise_pred).to(device=device, dtype=dtype)
        step_output = pipe.scheduler.step(noise_pred_torch, t, latents)
        latents = step_output.prev_sample

        print(f"  latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

    # VAE decode
    print("\n[vae] Decoding latents...")
    latents_scaled = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents_scaled.to(pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(image_np)

    # Save
    if args.output and (os.sep in args.output or '/' in args.output):
        out_path = Path(args.output)
    else:
        out_name = args.output or f"lightning_phone_s{args.steps}_seed{args.seed}.png"
        out_path = OUTPUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"\n[done] Saved: {out_path}")
    print(f"  Total phone UNet time: {total_phone_ms:.0f}ms ({total_phone_ms/1000:.1f}s)")
    print(f"  Average per step: {total_phone_ms/args.steps:.0f}ms")


if __name__ == "__main__":
    main()
