#!/usr/bin/env python3
"""Generate UNet INT8 calibration tensors from prompts.

Encodes prompts through both SDXL CLIP encoders to get text embeddings,
then creates calibration samples with random latent noise at various timesteps.

Usage (WSL):
  python3 make_calibration_data.py \
    --diffusers-dir /mnt/d/platform-tools/sdxl_npu/diffusers_pipeline \
    --prompts /mnt/d/platform-tools/sdxl_npu/калибровка.json \
    --output-dir /mnt/d/platform-tools/sdxl_npu/calibration_data \
    --resolution 1024x1024,832x1216,1216x832
"""
import argparse, gc, json, os, sys, time
import numpy as np


def encode_prompts_sdxl(diffusers_dir, prompts, device):
    """Encode prompts through both CLIP text encoders → hidden_states + pooled."""
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    print("[clip] Loading tokenizers and text encoders...", flush=True)
    tok1 = CLIPTokenizer.from_pretrained(diffusers_dir, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(diffusers_dir, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(
        diffusers_dir, subfolder="text_encoder", torch_dtype=torch.float32
    ).to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(
        diffusers_dir, subfolder="text_encoder_2", torch_dtype=torch.float32
    ).to(device).eval()

    all_hs, all_pool = [], []

    with torch.no_grad():
        for i, p in enumerate(prompts):
            text = p["positive"]
            ids1 = tok1(text, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
            ids2 = tok2(text, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)

            out1 = te1(ids1, output_hidden_states=True)
            out2 = te2(ids2, output_hidden_states=True)

            # SDXL concatenates penultimate hidden states: [1,77,768]+[1,77,1280]=[1,77,2048]
            hs = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
            pool = out2.text_embeds  # [1, 1280]

            all_hs.append(hs.cpu().numpy())
            all_pool.append(pool.cpu().numpy())

            if (i + 1) % 25 == 0:
                print(f"  Encoded {i+1}/{len(prompts)}", flush=True)

    del te1, te2, tok1, tok2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hs_all = np.concatenate(all_hs, axis=0)   # [N, 77, 2048]
    pool_all = np.concatenate(all_pool, axis=0)  # [N, 1280]
    print(f"[clip] Done: hidden_states {hs_all.shape}, text_embeds {pool_all.shape}", flush=True)
    return hs_all, pool_all


def build_calibration_set(hidden_states, text_embeds, w, h, n_timesteps=5, seed=42):
    """Build calibration arrays for a single resolution."""
    rng = np.random.RandomState(seed)
    lh, lw = h // 8, w // 8
    n_prompts = hidden_states.shape[0]

    # Uniformly sample timesteps across the diffusion schedule
    timesteps = np.linspace(999, 1, n_timesteps).astype(np.float32)

    # SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    time_ids_base = np.array([[h, w, 0, 0, h, w]], dtype=np.float32)

    samples, ts_arr, hs_arr, te_arr, ti_arr = [], [], [], [], []

    for pi in range(n_prompts):
        for t in timesteps:
            z = rng.randn(1, 4, lh, lw).astype(np.float32)
            samples.append(z)
            ts_arr.append(np.array([t], dtype=np.float32))
            hs_arr.append(hidden_states[pi:pi+1].astype(np.float32))
            te_arr.append(text_embeds[pi:pi+1].astype(np.float32))
            ti_arr.append(time_ids_base.copy())

    return {
        "sample": np.concatenate(samples),
        "timestep": np.concatenate(ts_arr),
        "encoder_hidden_states": np.concatenate(hs_arr),
        "text_embeds": np.concatenate(te_arr),
        "time_ids": np.concatenate(ti_arr),
    }


def main():
    ap = argparse.ArgumentParser(description="Build UNet calibration data from SDXL prompts")
    ap.add_argument("--diffusers-dir", required=True)
    ap.add_argument("--prompts", required=True, help="Path to калибровка.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resolution", default="1024x1024,832x1216,1216x832")
    ap.add_argument("--n-timesteps", type=int, default=5,
                    help="Timestep samples per prompt (total = prompts * timesteps)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import torch
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    print(f"Device: {device}", flush=True)

    with open(a.prompts, encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]

    t0 = time.time()
    hidden_states, text_embeds = encode_prompts_sdxl(a.diffusers_dir, prompts, device)

    for r in a.resolution.split(","):
        ww, hh = r.strip().split("x")
        w, h = int(ww), int(hh)
        out = os.path.join(a.output_dir, f"{w}x{h}")
        os.makedirs(out, exist_ok=True)

        out_file = os.path.join(out, "calibration.npz")
        if os.path.exists(out_file):
            print(f"[{w}x{h}] skip (exists): {out_file}", flush=True)
            continue

        print(f"[{w}x{h}] Building calibration set...", flush=True)
        data = build_calibration_set(hidden_states, text_embeds, w, h, a.n_timesteps)
        np.savez(out_file, **data)
        sz = os.path.getsize(out_file)
        n = data["sample"].shape[0]
        print(f"[{w}x{h}] Saved {n} samples → {out_file} ({sz/1e6:.1f} MB)", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
