"""
Test distillation LoRAs (Lightning / LCM) on waiIllustrious SDXL.
Goal: find a LoRA that eliminates CFG dependency for phone-only generation.
"""
import argparse, json, time, sys, os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "sdxl_npu" / "diffusers_pipeline")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "distillation_tests"


def assess(img_path):
    """Quick no-reference check."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from assess_generated_image import assess_image
    return assess_image(img_path)


def pixel_stats(img_path):
    from PIL import Image
    import numpy as np
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    hsv = np.array(Image.open(img_path).convert("HSV")).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    return {
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "sat_mean": float(np.mean(sat)),
        "pixel_min": float(np.min(img)),
        "pixel_max": float(np.max(img)),
    }


def run_config(pipe, cfg_name, prompt, negative, steps, seed, guidance_scale, width, height, out_dir):
    """Run one config and return results dict."""
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    t0 = time.perf_counter()

    # For distilled models, negative prompt is typically not used (guidance_scale <= 1)
    neg = negative if guidance_scale > 1.0 else None

    result = pipe(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=neg,
        negative_prompt_2=neg,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=gen,
    )
    elapsed = time.perf_counter() - t0
    img = result.images[0]
    out_path = out_dir / f"{cfg_name}.png"
    img.save(out_path)

    stats = pixel_stats(out_path)
    assessment = assess(out_path)

    info = {
        "config": cfg_name,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
        **stats,
        "verdict": assessment["verdict"],
        "confidence": assessment["confidence"],
    }
    print(f"  [{cfg_name}] {elapsed:.1f}s | gray_std={stats['gray_std']:.1f} sat={stats['sat_mean']:.3f} "
          f"| {assessment['verdict']} ({assessment['confidence']})")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a cute cat sitting on a windowsill, anime style, high quality")
    parser.add_argument("--negative", default="worst quality, low quality, blurry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-type", choices=["lightning", "lcm", "both"], default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] loading waiIllustrious SDXL pipeline...")
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, LCMScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        PIPELINE_DIR,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(args.device)

    all_results = []

    # --- Baseline: original model, CFG 3.5, 20 steps ---
    print("\n=== Baseline (no LoRA, CFG=3.5, 20 steps) ===")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    r = run_config(pipe, "baseline_cfg35_s20", args.prompt, args.negative,
                   steps=20, seed=args.seed, guidance_scale=3.5,
                   width=1024, height=1024, out_dir=OUT_DIR)
    all_results.append(r)

    # --- Baseline: no CFG, 20 steps ---
    print("\n=== Baseline (no LoRA, CFG=1, 20 steps) ===")
    r = run_config(pipe, "baseline_nocfg_s20", args.prompt, args.negative,
                   steps=20, seed=args.seed, guidance_scale=1.0,
                   width=1024, height=1024, out_dir=OUT_DIR)
    all_results.append(r)

    # --- SDXL-Lightning ---
    if args.lora_type in ("lightning", "both"):
        print("\n=== SDXL-Lightning LoRA ===")
        try:
            from huggingface_hub import hf_hub_download

            for n_steps, lora_file in [(4, "sdxl_lightning_4step_lora.safetensors"),
                                        (8, "sdxl_lightning_8step_lora.safetensors")]:
                print(f"\n  Downloading {lora_file}...")
                lora_path = hf_hub_download("ByteDance/SDXL-Lightning", lora_file)
                print(f"  Loaded from: {lora_path}")

                # Reload fresh pipeline for each LoRA to avoid contamination
                pipe_l = StableDiffusionXLPipeline.from_pretrained(
                    PIPELINE_DIR, torch_dtype=torch.float16, local_files_only=True,
                ).to(args.device)
                pipe_l.scheduler = EulerDiscreteScheduler.from_config(
                    pipe_l.scheduler.config, timestep_spacing="trailing"
                )
                pipe_l.load_lora_weights(lora_path)
                pipe_l.fuse_lora()

                # Test with CFG=0 (official recommendation)
                tag = f"lightning_{n_steps}step_cfg0"
                r = run_config(pipe_l, tag, args.prompt, args.negative,
                               steps=n_steps, seed=args.seed, guidance_scale=0.0,
                               width=1024, height=1024, out_dir=OUT_DIR)
                all_results.append(r)

                # Test with small CFG like 1.5 (some community results suggest this helps with anime)
                tag2 = f"lightning_{n_steps}step_cfg1.5"
                r2 = run_config(pipe_l, tag2, args.prompt, args.negative,
                                steps=n_steps, seed=args.seed, guidance_scale=1.5,
                                width=1024, height=1024, out_dir=OUT_DIR)
                all_results.append(r2)

                del pipe_l
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] Lightning LoRA failed: {e}")
            import traceback; traceback.print_exc()

    # --- LCM-LoRA ---
    if args.lora_type in ("lcm", "both"):
        print("\n=== LCM-LoRA-SDXL ===")
        try:
            from huggingface_hub import hf_hub_download

            lora_file = "pytorch_lora_weights.safetensors"
            print(f"  Downloading LCM-LoRA-SDXL...")
            lora_path = hf_hub_download("latent-consistency/lcm-lora-sdxl", lora_file)
            print(f"  Loaded from: {lora_path}")

            for n_steps, cfg_val in [(4, 1.0), (8, 1.0), (8, 2.0)]:
                pipe_lcm = StableDiffusionXLPipeline.from_pretrained(
                    PIPELINE_DIR, torch_dtype=torch.float16, local_files_only=True,
                ).to(args.device)
                pipe_lcm.scheduler = LCMScheduler.from_config(pipe_lcm.scheduler.config)
                pipe_lcm.load_lora_weights(lora_path)
                pipe_lcm.fuse_lora()

                tag = f"lcm_{n_steps}step_cfg{cfg_val}"
                r = run_config(pipe_lcm, tag, args.prompt, args.negative,
                               steps=n_steps, seed=args.seed, guidance_scale=cfg_val,
                               width=1024, height=1024, out_dir=OUT_DIR)
                all_results.append(r)
                del pipe_lcm
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] LCM-LoRA failed: {e}")
            import traceback; traceback.print_exc()

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'config':<35} {'steps':>5} {'cfg':>5} {'gray_std':>8} {'sat':>6} {'verdict':<20}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['config']:<35} {r['steps']:>5} {r['guidance_scale']:>5.1f} "
              f"{r['gray_std']:>8.1f} {r['sat_mean']:>6.3f} {r['verdict']:<20}")

    summary_path = OUT_DIR / "distillation_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ok] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
