#!/usr/bin/env python3
"""
Bake SDXL-Lightning 8-step LoRA into the base waiIllustrious UNet.

Saves the merged UNet weights as a new diffusers-compatible directory
that can be directly used for ONNX export via the existing extmaps pipeline.
"""
import argparse
import gc
import os
import torch


def _load_lora(pipe, lora_path: str | None, lora_repo: str, lora_file: str) -> None:
    if lora_path:
        print(f"[2/4] Loading LoRA from local/remote reference: {lora_path}...")
        pipe.load_lora_weights(lora_path)
        return

    print(f"[2/4] Loading Lightning LoRA: {lora_repo}/{lora_file}...")
    pipe.load_lora_weights(lora_repo, weight_name=lora_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", default=r"D:\platform-tools\sdxl_npu\diffusers_pipeline",
                        help="Path to base diffusers pipeline")
    parser.add_argument("--lora-repo", default="ByteDance/SDXL-Lightning",
                        help="HuggingFace repo for Lightning LoRA")
    parser.add_argument("--lora-file", default="sdxl_lightning_8step_lora.safetensors",
                        help="LoRA weights filename")
    parser.add_argument("--lora-path", default=None,
                        help="Optional local LoRA path/reference. If provided, --lora-repo/--lora-file are ignored")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="LoRA fuse scale (default: 1.0)")
    parser.add_argument("--output-dir", default=r"D:\platform-tools\sdxl_npu\unet_lightning8step_merged",
                        help="Output directory for merged UNet")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False,
                        help="Run quick forward verification after merge (default: off)")
    parser.add_argument("--verify-image", action=argparse.BooleanOptionalAction, default=False,
                        help="Run image generation verification (slow, default: off)")
    args = parser.parse_args()

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    # 1. Load base pipeline
    print(f"[1/4] Loading base pipeline from {args.pipeline_dir}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pipeline_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # 2. Load and apply LoRA
    _load_lora(pipe, args.lora_path, args.lora_repo, args.lora_file)

    # 3. Fuse LoRA into base weights permanently
    print(f"[3/4] Fusing LoRA into base weights (scale={args.lora_scale:.4f})...")
    pipe.fuse_lora(lora_scale=float(args.lora_scale))
    pipe.unload_lora_weights()

    # 4. Save merged UNet
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[4/4] Saving merged UNet to {args.output_dir}...")
    pipe.unet.save_pretrained(args.output_dir)

    # Verify (forward pass)
    if args.verify:
        print("\n=== Verification ===")
        from diffusers import UNet2DConditionModel
        unet_check = UNet2DConditionModel.from_pretrained(args.output_dir, torch_dtype=torch.float16)

        # Quick forward pass check
        unet_check = unet_check.to("cpu").eval()
        g = torch.Generator().manual_seed(42)
        sample = torch.randn(1, 4, 128, 128, dtype=torch.float16, generator=g)
        t = torch.tensor([999], dtype=torch.long)
        enc = torch.randn(1, 77, 2048, dtype=torch.float16, generator=g)
        added = {"text_embeds": torch.randn(1, 1280, dtype=torch.float16, generator=g),
                 "time_ids": torch.randn(1, 6, dtype=torch.float16, generator=g)}

        with torch.no_grad():
            out = unet_check(sample, t, enc, added_cond_kwargs=added).sample
        print(f"  UNet output: shape={out.shape}, mean={out.float().mean():.4f}, std={out.float().std():.4f}")
        print(f"  Finite: {torch.isfinite(out).all().item()}")
        del unet_check, out
        gc.collect()

    # Optional image verification (slow)
    if args.verify_image:
        print("\n=== Quick generation test (8 steps, CFG=0) ===")
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        img = pipe(
            prompt="a cute cat sitting on a windowsill, anime style, high quality",
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=torch.Generator(device).manual_seed(12345),
        ).images[0]
        verify_path = os.path.join(args.output_dir, "verify_merge.png")
        img.save(verify_path)
        print(f"  Saved verification image: {verify_path}")

        # Quick stats
        import numpy as np
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        print(f"  gray_std={gray.std():.1f}, mean={gray.mean():.1f}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n[done] Merged UNet saved successfully.")


if __name__ == "__main__":
    main()
