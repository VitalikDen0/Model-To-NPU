#!/usr/bin/env python3
"""
Generate host reference using embedding-space guidance (not noise-space CFG).
This tests the approach we want to use on the phone.
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

PROMPT = "a cute cat sitting on a windowsill, anime style, high quality"
NEGATIVE = ""
SEED = 42
STEPS = 20
DEVICE = "cuda"

CONFIGS = [
    ("embed_cfg20_s20", 2.0),
    ("embed_cfg35_s20", 3.5),
    ("embed_cfg50_s20", 5.0),
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

    # Encode positive and negative prompts
    with torch.inference_mode():
        pos_embeds, neg_embeds, pos_pooled, neg_pooled = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=PROMPT,
            negative_prompt=NEGATIVE, negative_prompt_2=NEGATIVE,
            do_classifier_free_guidance=True,
            device=DEVICE, num_images_per_prompt=1,
        )

    add_time_ids = pipe._get_add_time_ids(
        original_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        target_size=(1024, 1024),
        dtype=pos_embeds.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(DEVICE)

    unet = pipe.unet

    for tag, gs in CONFIGS:
        print(f"\n=== {tag} (embed-space guidance, scale={gs}) ===")

        # Mix embeddings: mixed = neg + gs * (pos - neg)
        mixed_embeds = neg_embeds + gs * (pos_embeds - neg_embeds)
        mixed_pooled = neg_pooled + gs * (pos_pooled - neg_pooled)

        print(f"  pos_embeds: range=[{pos_embeds.min():.3f}, {pos_embeds.max():.3f}], std={pos_embeds.std():.4f}")
        print(f"  neg_embeds: range=[{neg_embeds.min():.3f}, {neg_embeds.max():.3f}], std={neg_embeds.std():.4f}")
        print(f"  mixed: range=[{mixed_embeds.min():.3f}, {mixed_embeds.max():.3f}], std={mixed_embeds.std():.4f}")

        # Scheduler setup
        pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
        g = torch.Generator(device=DEVICE).manual_seed(SEED)
        latents = torch.randn((1, 4, 128, 128), generator=g, device=DEVICE, dtype=torch.float32)
        latents = latents * pipe.scheduler.init_noise_sigma

        with torch.inference_mode():
            for i, t_val in enumerate(pipe.scheduler.timesteps.tolist()):
                t_tensor = torch.tensor(t_val, dtype=torch.float32, device=DEVICE)
                latent_in = pipe.scheduler.scale_model_input(latents, t_tensor)

                # Run UNet ONCE with mixed embeddings (no CFG subtraction)
                noise_pred = unet(
                    latent_in, t_tensor, encoder_hidden_states=mixed_embeds,
                    added_cond_kwargs={"text_embeds": mixed_pooled, "time_ids": add_time_ids},
                    return_dict=False,
                )[0]

                latents = pipe.scheduler.step(noise_pred, t_tensor, latents, return_dict=False)[0]
                if i % 5 == 0 or i == STEPS - 1:
                    lat_np = latents.cpu().numpy()[0]
                    print(f"  step {i}: lat_std=[{', '.join(f'{lat_np[c].std():.3f}' for c in range(4))}]")

        # VAE decode
        with torch.inference_mode():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = (decoded / 2 + 0.5).clamp(0, 1)

        img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        img_path = OUT_DIR / f"host_reference_{tag}.png"
        Image.fromarray(img_np).save(img_path)

        r, g_ch, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        maxc = np.max(img_np.astype(float), axis=2)
        minc = np.min(img_np.astype(float), axis=2)
        sat = np.where(maxc > 0, (maxc - minc) / maxc, 0)
        print(f"  saved: {img_path}")
        print(f"  image: gray_std={gray.std():.1f}  sat_mean={sat.mean():.3f}")

    # Also generate classic noise-CFG reference at same scales for comparison
    print("\n=== Classic noise-CFG comparison (cfg35_s20 already exists) ===")
    print("  See host_reference_cfg35_s20.png and host_reference_cfg5_s20.png")


if __name__ == "__main__":
    main()
