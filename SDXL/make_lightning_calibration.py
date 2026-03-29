#!/usr/bin/env python3
"""
Generate proper calibration data for Lightning UNet quantization.

Key fix: calibration latents must be scaled by init_noise_sigma (14.6x)
to match actual inference distribution. Original calibration NPZ used
std=1.0 latents, but EulerDiscrete scheduler scales them by ~14.6.

Usage:
  python NPU/make_lightning_calibration.py --samples 16
"""
import argparse, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16, help="Number of calibration samples")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.out_dir) if args.out_dir else SDXL_NPU / f"calib_lightning_w8a16_{args.samples}s_scaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] samples={args.samples}, out={out_dir}, device={device}")

    # Load pipeline
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
    print("[init] Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(SDXL_NPU / "diffusers_pipeline"),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)

    # Lightning scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.scheduler.set_timesteps(8, device=device)
    timesteps = pipe.scheduler.timesteps
    init_sigma = float(pipe.scheduler.init_noise_sigma)
    print(f"[scheduler] init_noise_sigma={init_sigma:.3f}, timesteps={timesteps.tolist()}")

    # Load Lightning UNet for bias computation
    print("[init] Loading Lightning UNet...")
    lightning_unet = UNet2DConditionModel.from_pretrained(
        str(SDXL_NPU / "unet_lightning8step_merged"),
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    lightning_unet.eval()

    resnet_modules = collect_unet_resnet_conditioning_modules(lightning_unet)
    spatial_shapes = infer_unet_resnet_spatial_shapes(lightning_unet, 128, 128)
    bias_names = [
        f"resnet_bias_{i:02d}_{name.replace('.', '_')}"
        for i, (name, _) in enumerate(resnet_modules)
    ]
    print(f"[extmaps] {len(resnet_modules)} bias sites")

    # Diverse prompts for calibration
    prompts = [
        "1girl, upper body, looking at viewer, masterpiece, best quality",
        "landscape, mountains, sunset, dramatic lighting, 4k",
        "a cute cat sitting on a windowsill, digital art",
        "abstract art, colorful geometric shapes, modern",
        "portrait of an old man, oil painting, rembrandt style",
        "city skyline at night, neon lights, cyberpunk",
        "a red sports car on a highway, motion blur",
        "underwater scene, coral reef, tropical fish",
        "a castle on a hill, medieval fantasy, detailed",
        "anime girl with blue hair, school uniform",
        "still life, flowers in a vase, soft lighting",
        "robot standing in rain, futuristic, cinematic",
        "forest path, autumn leaves, golden hour",
        "spacecraft orbiting a planet, sci-fi illustration",
        "a dog playing in the snow, happy expression",
        "dark gothic cathedral interior, stained glass",
        "beach scene, waves crashing, tropical paradise",
        "steampunk clockwork mechanism, brass and copper",
        "fantasy dragon flying over mountains, epic",
        "cozy cafe interior, warm lighting, detailed",
    ]

    input_list_lines = []
    sample_idx = 0

    for prompt_idx in range(min(args.samples, len(prompts))):
        prompt = prompts[prompt_idx % len(prompts)]

        # Encode prompt (no CFG for Lightning)
        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=prompt, negative_prompt="",
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        add_time_ids = pipe._get_add_time_ids(
            original_size=(1024, 1024),
            crops_coords_top_left=(0, 0),
            target_size=(1024, 1024),
            dtype=pe.dtype,
            text_encoder_projection_dim=1280,
        ).to(device)

        # Generate properly scaled latents for each timestep
        for t_idx, t in enumerate(timesteps):
            if sample_idx >= args.samples:
                break

            sample_dir = out_dir / f"sample_{sample_idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Create latent scaled by init_noise_sigma (matching real inference)
            gen = torch.Generator(device=device).manual_seed(42 + sample_idx * 13 + t_idx * 7)
            latent = torch.randn((1, 4, 128, 128), generator=gen, device=device, dtype=torch.float32)
            latent = latent * init_sigma  # CRITICAL: match actual inference scale

            # Scale model input for this timestep
            latent_in = pipe.scheduler.scale_model_input(latent, t)

            # Compute biases
            with torch.no_grad():
                bias_tensors = compute_external_resnet_biases(
                    lightning_unet,
                    latent_in.to(device, dtype=torch.float32),
                    torch.tensor([float(t)], dtype=torch.float32, device=device),
                    pe.to(device, dtype=torch.float32),
                    ppe.to(device, dtype=torch.float32),
                    add_time_ids.to(device, dtype=torch.float32),
                )

            # Save as NCHW float32 raw files
            # QNN converter IR always reads inputs as float32 (even for fp16 ONNX models)
            sample_np = latent_in.cpu().float().numpy()  # [1,4,128,128] NCHW float32
            sample_np.tofile(str(sample_dir / "sample.raw"))

            ehs_np = pe.cpu().float().numpy()  # [1,77,2048] float32
            ehs_np.tofile(str(sample_dir / "encoder_hidden_states.raw"))

            file_parts = [
                str(sample_dir / "sample.raw"),
                str(sample_dir / "encoder_hidden_states.raw"),
            ]

            for bi, ((name, _), bt) in enumerate(zip(resnet_modules, bias_tensors)):
                h, w = spatial_shapes[name]
                bias_expanded = bt.expand(-1, -1, h, w).cpu().float().numpy()  # NCHW float32
                fname = f"{bias_names[bi]}.raw"
                bias_expanded.tofile(str(sample_dir / fname))
                file_parts.append(str(sample_dir / fname))

            input_list_lines.append(" ".join(file_parts))

            print(f"  [{sample_idx+1}/{args.samples}] prompt={prompt_idx}, t={float(t):.0f}, "
                  f"latent_range=[{sample_np.min():.2f}, {sample_np.max():.2f}]")
            sample_idx += 1

            if sample_idx >= args.samples:
                break

    # Write input list
    input_list_path = out_dir / "unet_extbias_input_list.txt"
    with open(input_list_path, "w") as f:
        f.write("\n".join(input_list_lines) + "\n")

    print(f"\n[done] {sample_idx} samples, input list: {input_list_path}")
    print(f"  Latent scaling: init_noise_sigma={init_sigma:.3f}")


if __name__ == "__main__":
    main()
