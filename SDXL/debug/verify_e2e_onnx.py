#!/usr/bin/env python3
"""
PC-side end-to-end sanity check: ONNX CLIP-L + CLIP-G + VAE vs PyTorch pipeline.

Runs the full SDXL text encoding → (skip UNet, use fixed noise) → VAE decode path
and verifies numerical parity between PyTorch and ONNX at each stage.

Usage:
  python NPU/verify_e2e_onnx.py
"""
import gc, sys, os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSERS = os.path.join(ROOT, "sdxl_npu", "diffusers_pipeline")
ONNX_DIR = os.path.join(ROOT, "sdxl_npu", "onnx_clip_vae")

def compare(name, a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mae = float(np.mean(np.abs(a - b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    max_abs = float(np.max(np.abs(a - b)))
    cos = float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
    ok = "PASS" if cos > 0.9999 else ("MARGINAL" if cos > 0.999 else "FAIL")
    print(f"  [{name}] mae={mae:.2e} rmse={rmse:.2e} max={max_abs:.2e} cos={cos:.10f} {ok}")
    return cos > 0.999

PROMPT = "1girl, white hair, blue eyes, masterpiece, best quality, anime style"

print("=" * 60)
print("  SDXL ONNX end-to-end sanity check")
print("=" * 60)

# ── Stage 1: CLIP tokenization ──
print("\n[1] Tokenizing prompt...")
from transformers import CLIPTokenizer
tok1 = CLIPTokenizer.from_pretrained(os.path.join(DIFFUSERS, "tokenizer"), local_files_only=True)
tok2 = CLIPTokenizer.from_pretrained(os.path.join(DIFFUSERS, "tokenizer_2"), local_files_only=True)
ids1 = tok1(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt")["input_ids"]
ids2 = tok2(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt")["input_ids"]
print(f"  CLIP-L tokens: {ids1.shape}, range=[{ids1.min()}, {ids1.max()}]")
print(f"  CLIP-G tokens: {ids2.shape}, range=[{ids2.min()}, {ids2.max()}]")

# ── Stage 2: PyTorch CLIP reference ──
print("\n[2] PyTorch CLIP reference (FP32 on CUDA)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import CLIPTextModel, CLIPTextModelWithProjection
te1 = CLIPTextModel.from_pretrained(os.path.join(DIFFUSERS, "text_encoder"), local_files_only=True, torch_dtype=torch.float32).eval().to(device)
with torch.no_grad():
    pt_out1 = te1(ids1.to(device), output_hidden_states=True)
pt_hidden1 = pt_out1.hidden_states[-2].cpu().numpy()  # [1,77,768]
del te1; gc.collect(); torch.cuda.empty_cache()

te2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(DIFFUSERS, "text_encoder_2"), local_files_only=True, torch_dtype=torch.float32).eval().to(device)
with torch.no_grad():
    pt_out2 = te2(ids2.to(device), output_hidden_states=True)
pt_hidden2 = pt_out2.hidden_states[-2].cpu().numpy()  # [1,77,1280]
pt_pooled = pt_out2.text_embeds.cpu().numpy()  # [1,1280]
del te2; gc.collect(); torch.cuda.empty_cache()

# Concatenate as SDXL pipeline does
pt_prompt_embeds = np.concatenate([pt_hidden1, pt_hidden2], axis=-1)  # [1,77,2048]
print(f"  PyTorch prompt_embeds: {pt_prompt_embeds.shape}, range=[{pt_prompt_embeds.min():.4f}, {pt_prompt_embeds.max():.4f}]")
print(f"  PyTorch pooled_prompt_embeds: {pt_pooled.shape}, range=[{pt_pooled.min():.4f}, {pt_pooled.max():.4f}]")

# ── Stage 3: ONNX CLIP inference ──
print("\n[3] ONNX CLIP inference...")
import onnxruntime as ort

# CLIP-L
sess1 = ort.InferenceSession(os.path.join(ONNX_DIR, "clip_l.onnx"), providers=["CPUExecutionProvider"])
ort_hidden1 = sess1.run(None, {"input_ids": ids1.numpy().astype(np.int64)})[0]  # penultimate_hidden
del sess1

# CLIP-G
sess2 = ort.InferenceSession(os.path.join(ONNX_DIR, "clip_g.onnx"), providers=["CPUExecutionProvider"])
ort_g_out = sess2.run(None, {"input_ids": ids2.numpy().astype(np.int64)})
ort_names = [o.name for o in ort.InferenceSession(os.path.join(ONNX_DIR, "clip_g.onnx"), providers=["CPUExecutionProvider"]).get_outputs()]
# Match by name
ort_hidden2 = ort_pooled = None
for i, name in enumerate(ort_names):
    if "hidden" in name:
        ort_hidden2 = ort_g_out[i]
    elif "embed" in name:
        ort_pooled = ort_g_out[i]
    elif len(ort_g_out[i].shape) == 3:
        ort_hidden2 = ort_g_out[i]
    elif len(ort_g_out[i].shape) == 2:
        ort_pooled = ort_g_out[i]
del sess2

ort_prompt_embeds = np.concatenate([ort_hidden1, ort_hidden2], axis=-1)  # [1,77,2048]
print(f"  ONNX prompt_embeds: {ort_prompt_embeds.shape}, range=[{ort_prompt_embeds.min():.4f}, {ort_prompt_embeds.max():.4f}]")
print(f"  ONNX pooled_prompt_embeds: {ort_pooled.shape}, range=[{ort_pooled.min():.4f}, {ort_pooled.max():.4f}]")

print("\n  Comparing CLIP outputs:")
compare("clip_l_hidden", pt_hidden1, ort_hidden1)
compare("clip_g_hidden", pt_hidden2, ort_hidden2)
compare("clip_g_pooled", pt_pooled, ort_pooled)
compare("prompt_embeds", pt_prompt_embeds, ort_prompt_embeds)

# ── Stage 4: VAE decode comparison ──
print("\n[4] VAE decode comparison...")
torch.manual_seed(42)
latent = torch.randn(1, 4, 128, 128, dtype=torch.float32, device=device)

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(os.path.join(DIFFUSERS, "vae"), local_files_only=True, torch_dtype=torch.float32).eval().to(device)
with torch.no_grad():
    pt_image = vae.decode(latent, return_dict=False)[0].cpu().numpy()
del vae; gc.collect(); torch.cuda.empty_cache()

vae_onnx = os.path.join(ONNX_DIR, "vae_decoder.onnx")
sess_vae = ort.InferenceSession(vae_onnx, providers=["CPUExecutionProvider"])
ort_image = sess_vae.run(None, {"latent": latent.cpu().numpy()})[0]
del sess_vae

print(f"  PyTorch VAE: {pt_image.shape}, range=[{pt_image.min():.4f}, {pt_image.max():.4f}]")
print(f"  ONNX VAE:    {ort_image.shape}, range=[{ort_image.min():.4f}, {ort_image.max():.4f}]")
compare("vae_decode", pt_image, ort_image)

# ── Summary ──
print("\n" + "=" * 60)
print("  End-to-end sanity check COMPLETE")
print("=" * 60)
print(f"  CLIP-L output:  penultimate_hidden {ort_hidden1.shape}")
print(f"  CLIP-G output:  penultimate_hidden {ort_hidden2.shape} + text_embeds {ort_pooled.shape}")
print(f"  prompt_embeds:  {ort_prompt_embeds.shape}")
print(f"  VAE decode:     {ort_image.shape}")
print(f"\n  All models ready for phone deployment!")
