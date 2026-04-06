#!/usr/bin/env python3
"""
PC-only SDXL generation for quality comparison with phone NPU output.
Generates images using:
  1. Lightning merged UNet (8 steps, CFG=0) — same as phone
  2. Lightning merged UNet with CFG (8 steps, CFG=2.0) — quality test  
  3. Full original model (30 steps, CFG=3.5) — ComfyUI-equivalent reference
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SDXL_NPU = ROOT / "sdxl_npu"
sys.path.insert(0, str(SDXL_NPU))

DIFFUSERS_DIR = SDXL_NPU / "diffusers_pipeline"
MERGED_UNET_DIR = SDXL_NPU / "unet_lightning8step_merged"
OUTPUT_DIR = ROOT / "NPU" / "outputs"


def load_pipeline(unet_dir, device="cuda"):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

    tok_l = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer"))
    tok_g = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer_2"))
    te_l = CLIPTextModel.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder"), torch_dtype=torch.float16
    ).to(device).eval()
    te_g = CLIPTextModelWithProjection.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder_2"), torch_dtype=torch.float16
    ).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        str(unet_dir), torch_dtype=torch.float16
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        str(DIFFUSERS_DIR / "vae"), torch_dtype=torch.float16
    ).to(device).eval()

    cfg = json.loads((DIFFUSERS_DIR / "scheduler" / "scheduler_config.json").read_text())
    scheduler = EulerDiscreteScheduler.from_config(cfg, timestep_spacing="trailing")

    return tok_l, tok_g, te_l, te_g, unet, vae, scheduler


def encode_prompt(tok_l, tok_g, te_l, te_g, prompt, device="cuda", negative_prompt=None):
    # Positive
    ids_l = tok_l(prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(device)
    ids_g = tok_g(prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out_l = te_l(ids_l, output_hidden_states=True)
        hs_l = out_l.hidden_states[-2]  # [1,77,768]
        out_g = te_g(ids_g, output_hidden_states=True)
        hs_g = out_g.hidden_states[-2]  # [1,77,1280]
        pooled = out_g.text_embeds       # [1,1280]

    prompt_embeds = torch.cat([hs_l, hs_g], dim=-1)  # [1,77,2048]

    if negative_prompt is not None:
        nids_l = tok_l(negative_prompt, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").input_ids.to(device)
        nids_g = tok_g(negative_prompt, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            nout_l = te_l(nids_l, output_hidden_states=True)
            nhs_l = nout_l.hidden_states[-2]
            nout_g = te_g(nids_g, output_hidden_states=True)
            nhs_g = nout_g.hidden_states[-2]
            npooled = nout_g.text_embeds
        neg_embeds = torch.cat([nhs_l, nhs_g], dim=-1)
        return prompt_embeds, pooled, neg_embeds, npooled

    return prompt_embeds, pooled, None, None


def generate(prompt, seed, steps, cfg_scale, unet_dir, tag,
             negative_prompt=None, h=1024, w=1024):
    device = "cuda"
    print(f"\n{'='*60}")
    print(f"Generating: {tag}")
    print(f"  Prompt: {prompt}")
    print(f"  Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}, Size: {w}x{h}")
    if negative_prompt:
        print(f"  Negative: {negative_prompt[:80]}...")
    print(f"  UNet: {unet_dir.name}")
    print(f"{'='*60}")

    t0 = time.time()
    tok_l, tok_g, te_l, te_g, unet, vae, scheduler = load_pipeline(unet_dir, device)
    print(f"  Pipeline loaded: {time.time()-t0:.1f}s")

    pe, pooled, neg_pe, neg_pooled = encode_prompt(
        tok_l, tok_g, te_l, te_g, prompt, device, negative_prompt
    )

    latent_h, latent_w = h // 8, w // 8
    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 4, latent_h, latent_w), generator=gen,
                          device=device, dtype=torch.float16)

    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    time_ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=torch.float16, device=device)
    added_cond = {"text_embeds": pooled.half(), "time_ids": time_ids}

    if cfg_scale > 1.0 and neg_pe is not None:
        neg_added_cond = {"text_embeds": neg_pooled.half(), "time_ids": time_ids}

    t1 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        lat_in = scheduler.scale_model_input(latents, t)

        if cfg_scale > 1.0 and neg_pe is not None:
            lat_in_2 = torch.cat([lat_in, lat_in])
            pe_2 = torch.cat([neg_pe.half(), pe.half()])
            ac_2 = {
                "text_embeds": torch.cat([neg_added_cond["text_embeds"],
                                          added_cond["text_embeds"]]),
                "time_ids": torch.cat([neg_added_cond["time_ids"],
                                       added_cond["time_ids"]]),
            }
            with torch.no_grad():
                noise_pred = unet(lat_in_2, t, encoder_hidden_states=pe_2,
                                  added_cond_kwargs=ac_2).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            with torch.no_grad():
                noise_pred = unet(lat_in, t, encoder_hidden_states=pe.half(),
                                  added_cond_kwargs=added_cond).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        print(f"    Step {i+1}/{steps}: t={t.item():.0f}, "
              f"pred[{noise_pred.min():.2f}..{noise_pred.max():.2f}]")

    print(f"  Denoising: {time.time()-t1:.1f}s")

    # VAE decode
    vae_sf = vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents / vae_sf).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    img_u8 = (image * 255).round().astype(np.uint8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pc_ref_{tag}.png"
    Image.fromarray(img_u8).save(str(out_path))
    print(f"  Saved: {out_path}")
    print(f"  Total: {time.time()-t0:.1f}s")

    # Free VRAM
    del unet, vae, te_l, te_g
    torch.cuda.empty_cache()
    return str(out_path)


NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality:1.4), "
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, "
    "wrong anatomy, extra limb, missing limb, floating limbs, "
    "(mutated hands and fingers:1.4), disconnected limbs, mutation, "
    "mutated, ugly, disgusting, blurry, amputation, text, watermark, "
    "signature, censor, censored, bar"
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--only", type=str, default=None,
                    help="Run only specific variant: lightning_nocfg, lightning_cfg, full_cfg")
    a = ap.parse_args()

    variants = []

    # 1. Lightning, 8 steps, CFG=0 — same as phone
    if not a.only or a.only == "lightning_nocfg":
        variants.append(dict(
            prompt=a.prompt, seed=a.seed, steps=8, cfg_scale=1.0,
            unet_dir=MERGED_UNET_DIR,
            tag=f"lightning_nocfg_s{a.seed}",
            negative_prompt=None,
        ))

    # 2. Lightning, 8 steps, CFG=2.0 — with guidance
    if not a.only or a.only == "lightning_cfg":
        variants.append(dict(
            prompt=a.prompt, seed=a.seed, steps=8, cfg_scale=2.0,
            unet_dir=MERGED_UNET_DIR,
            tag=f"lightning_cfg2_s{a.seed}",
            negative_prompt=NEGATIVE_PROMPT,
        ))

    # 3. Full original model, 30 steps, CFG=3.5 — ComfyUI reference
    if not a.only or a.only == "full_cfg":
        orig_unet = DIFFUSERS_DIR / "unet"
        if orig_unet.exists():
            variants.append(dict(
                prompt=a.prompt, seed=a.seed, steps=30, cfg_scale=3.5,
                unet_dir=orig_unet,
                tag=f"full_cfg35_s{a.seed}",
                negative_prompt=NEGATIVE_PROMPT,
            ))
        else:
            print(f"[SKIP] Full original UNet not found at {orig_unet}")

    for v in variants:
        generate(**v)
