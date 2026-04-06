#!/usr/bin/env python3
"""
Generate host references AND step-by-step latent diagnostics.
Saves latents + noise_pred at each step for comparison with phone output.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT_DIR = Path(r"d:\platform-tools\NPU\outputs")
PIPE_DIR = r"d:\platform-tools\sdxl_npu\diffusers_pipeline"
DIAG_DIR = Path(r"d:\platform-tools\NPU\host_step_diag")

PROMPT = "a cute cat sitting on a windowsill, anime style, high quality"
SEED = 42
STEPS = 20
DEVICE = "cuda"
CONFIGS = [
    # (tag, guidance_scale)
    ("cfg35_s20", 3.5),
    ("cfg5_s20", 5.0),
]


def main():
    print("[info] loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        PIPE_DIR,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    for tag, gs in CONFIGS:
        do_cfg = gs > 1.0
        print(f"\n=== {tag} (guidance_scale={gs}, do_cfg={do_cfg}) ===")

        # Encode prompt  
        with torch.inference_mode():
            prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
                prompt=PROMPT, prompt_2=PROMPT,
                negative_prompt="", negative_prompt_2="",
                do_classifier_free_guidance=do_cfg,
                device=DEVICE, num_images_per_prompt=1,
            )

        add_time_ids = pipe._get_add_time_ids(
            original_size=(1024, 1024),
            crops_coords_top_left=(0, 0),
            target_size=(1024, 1024),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(DEVICE)

        # Scheduler setup
        pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
        g = torch.Generator(device=DEVICE).manual_seed(SEED)
        latents = torch.randn((1, 4, 128, 128), generator=g, device=DEVICE, dtype=torch.float32)
        latents = latents * pipe.scheduler.init_noise_sigma

        diag_dir = DIAG_DIR / tag
        diag_dir.mkdir(parents=True, exist_ok=True)

        step_reports = []
        with torch.inference_mode():
            for i, t_val in enumerate(pipe.scheduler.timesteps.tolist()):
                t_tensor = torch.tensor(t_val, dtype=torch.float32, device=DEVICE)
                latent_in = pipe.scheduler.scale_model_input(latents, t_tensor)

                if do_cfg:
                    latent_cat = torch.cat([latent_in] * 2)
                    prompt_cat = torch.cat([neg_embeds, prompt_embeds])
                    pooled_cat = torch.cat([neg_pooled, pooled])
                    time_ids_cat = torch.cat([add_time_ids] * 2)

                    noise_full = pipe.unet(
                        latent_cat, t_tensor, encoder_hidden_states=prompt_cat,
                        added_cond_kwargs={"text_embeds": pooled_cat, "time_ids": time_ids_cat},
                        return_dict=False,
                    )[0]
                    noise_uncond, noise_cond = noise_full.chunk(2)
                    noise_pred = noise_uncond + gs * (noise_cond - noise_uncond)
                else:
                    noise_pred = pipe.unet(
                        latent_in, t_tensor, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
                        return_dict=False,
                    )[0]

                # Save per-step diagnostics
                lat_np = latent_in.detach().cpu().numpy()[0]  # [4,128,128]
                np_pred = noise_pred.detach().cpu().numpy()[0]  # [4,128,128]
                if do_cfg:
                    np_cond = noise_cond.detach().cpu().numpy()[0]
                    np_uncond = noise_uncond.detach().cpu().numpy()[0]
                    delta = np_cond - np_uncond

                report = {
                    "step": i, "timestep": t_val,
                    "latent_stats": {
                        f"ch{c}": {"mean": float(lat_np[c].mean()), "std": float(lat_np[c].std()),
                                   "min": float(lat_np[c].min()), "max": float(lat_np[c].max())}
                        for c in range(4)
                    },
                    "noise_pred_stats": {
                        f"ch{c}": {"mean": float(np_pred[c].mean()), "std": float(np_pred[c].std()),
                                   "min": float(np_pred[c].min()), "max": float(np_pred[c].max())}
                        for c in range(4)
                    },
                }
                if do_cfg:
                    report["cond_uncond_delta"] = {
                        f"ch{c}": {
                            "delta_mean": float(delta[c].mean()),
                            "delta_std": float(delta[c].std()),
                            "delta_norm": float(np.linalg.norm(delta[c])),
                            "uncond_norm": float(np.linalg.norm(np_uncond[c])),
                            "cond_norm": float(np.linalg.norm(np_cond[c])),
                        }
                        for c in range(4)
                    }
                step_reports.append(report)

                latents = pipe.scheduler.step(noise_pred, t_tensor, latents, return_dict=False)[0]
                if (i % 5 == 0) or i == STEPS - 1:
                    print(f"  step {i}: latent_std=[{', '.join(f'{lat_np[c].std():.3f}' for c in range(4))}]")

        # Final VAE decode
        with torch.inference_mode():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = (decoded / 2 + 0.5).clamp(0, 1)

        img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        img_path = OUT_DIR / f"host_reference_{tag}.png"
        Image.fromarray(img_np).save(img_path)
        print(f"  saved: {img_path}")

        # Per-channel final latent stats
        final_lat = latents.detach().cpu().numpy()[0]
        print(f"  final latent per-ch std: [{', '.join(f'{final_lat[c].std():.4f}' for c in range(4))}]")
        print(f"  final latent per-ch mean: [{', '.join(f'{final_lat[c].mean():.4f}' for c in range(4))}]")

        # Image pixel stats
        r, g_ch, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        maxc = np.max(img_np.astype(float), axis=2)
        minc = np.min(img_np.astype(float), axis=2)
        sat = np.where(maxc > 0, (maxc - minc) / maxc, 0)
        print(f"  image: gray_std={gray.std():.1f}  sat_mean={sat.mean():.3f}")

        # Save report 
        diag_path = diag_dir / "step_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(step_reports, f, indent=2)
        print(f"  diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
