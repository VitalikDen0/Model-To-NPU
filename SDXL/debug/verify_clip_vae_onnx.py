#!/usr/bin/env python3
"""
Verify CLIP-L, CLIP-G, VAE decoder: PyTorch original vs ONNX export.
Ensures numerically identical (or near-identical) outputs.

Usage:
  python NPU/verify_clip_vae_onnx.py --component clip_l
  python NPU/verify_clip_vae_onnx.py --component clip_g
  python NPU/verify_clip_vae_onnx.py --component vae
  python NPU/verify_clip_vae_onnx.py --component all
"""
import argparse, gc, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SDXL_NPU = ROOT / "sdxl_npu"
DIFFUSERS_DIR = SDXL_NPU / "diffusers_pipeline"
ONNX_DIR = SDXL_NPU / "onnx_clip_vae"
PREP_CLIP_L = SDXL_NPU / "onnx_clip_l_prepared" / "clip_l.onnx"
PREP_CLIP_G = SDXL_NPU / "onnx_clip_g_prepared" / "clip_g.onnx"
PREP_VAE = SDXL_NPU / "onnx_vae_prepared" / "vae_decoder.onnx"


def compare(name: str, a: np.ndarray, b: np.ndarray):
    """Print detailed numerical comparison."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mae = np.mean(np.abs(a - b))
    rmse = np.sqrt(np.mean((a - b) ** 2))
    max_abs = np.max(np.abs(a - b))
    cos = np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    print(f"  [{name}] mae={mae:.2e}, rmse={rmse:.2e}, max_abs={max_abs:.2e}, cos={cos:.10f}")
    print(f"  [{name}] a: range=[{a.min():.6f}, {a.max():.6f}], std={a.std():.6f}")
    print(f"  [{name}] b: range=[{b.min():.6f}, {b.max():.6f}], std={b.std():.6f}")
    if cos > 0.9999 and max_abs < 0.01:
        print(f"  [{name}] ✓ PASS")
        return True
    elif cos > 0.999:
        print(f"  [{name}] ~ MARGINAL (cos={cos:.8f})")
        return True
    else:
        print(f"  [{name}] ✗ FAIL")
        return False


def check_weight_dtypes(component: str):
    """Check and report weight dtypes for a component."""
    from safetensors import safe_open
    if component == "clip_l":
        path = DIFFUSERS_DIR / "text_encoder" / "model.safetensors"
    elif component == "clip_g":
        path = DIFFUSERS_DIR / "text_encoder_2" / "model.safetensors"
    elif component == "vae":
        path = list((DIFFUSERS_DIR / "vae").glob("*.safetensors"))[0]
    else:
        return

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        dtypes = set()
        for k in keys:
            dtypes.add(str(f.get_tensor(k).dtype))
        print(f"  [{component}] safetensors: {len(keys)} tensors, dtypes: {dtypes}")


def verify_clip_l():
    """Verify CLIP-L: PyTorch hidden_states[-2] vs ONNX penultimate_hidden."""
    print(f"\n{'='*60}\n  Verifying CLIP-L (penultimate hidden layer)\n{'='*60}")

    check_weight_dtypes("clip_l")

    from transformers import CLIPTextModel, CLIPTokenizer
    import onnxruntime as ort

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FP32 model (same dtype as export wrapper uses)
    print("  Loading PyTorch model (FP32)...")
    model_f32 = CLIPTextModel.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder"), local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().to(device)

    # Test input
    tokenizer = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer"), local_files_only=True)
    tokens = tokenizer(
        "1girl, masterpiece, best quality",
        padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    )
    input_ids = tokens["input_ids"]  # [1, 77] int64
    print(f"  input_ids: shape={list(input_ids.shape)}, dtype={input_ids.dtype}")
    print(f"  token range: [{input_ids.min().item()}, {input_ids.max().item()}]")

    # PyTorch FP32: get hidden_states[-2] (penultimate layer — what SDXL pipeline uses)
    with torch.no_grad():
        pt_out_f32 = model_f32(input_ids.to(device), output_hidden_states=True)
    pt_penultimate = pt_out_f32.hidden_states[-2].cpu().numpy()  # [1, 77, 768]
    pt_last = pt_out_f32.last_hidden_state.cpu().numpy()
    has_nan_last = np.isnan(pt_last).any()
    has_nan_penult = np.isnan(pt_penultimate).any()
    print(f"  PyTorch penultimate hidden (layer -2): {pt_penultimate.shape}, range=[{pt_penultimate.min():.4f}, {pt_penultimate.max():.4f}], nan={has_nan_penult}")
    print(f"  PyTorch last_hidden_state (layer -1): nan={has_nan_last} (expected True for this model)")

    # ONNX inference — now exports penultimate_hidden
    onnx_path = ONNX_DIR / "clip_l.onnx"
    if onnx_path.exists():
        print(f"  Loading ONNX: {onnx_path}...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        print(f"  ONNX output names: {out_names}")
        ort_out = sess.run(None, {"input_ids": input_ids.numpy().astype(np.int64)})
        ort_hidden = ort_out[0]  # penultimate_hidden [1, 77, 768]
        has_nan_onnx = np.isnan(ort_hidden).any()
        print(f"  ONNX penultimate_hidden: {ort_hidden.shape}, range=[{ort_hidden.min():.4f}, {ort_hidden.max():.4f}], nan={has_nan_onnx}")
        print(f"  PyTorch FP32 penultimate vs ONNX:")
        compare("pt_fp32_vs_onnx_penultimate", pt_penultimate, ort_hidden)
        del sess
    else:
        print(f"  [skip] ONNX not found: {onnx_path}")

    del model_f32
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()


def verify_clip_g():
    """Verify CLIP-G: PyTorch hidden_states[-2] + text_embeds vs ONNX."""
    print(f"\n{'='*60}\n  Verifying CLIP-G (penultimate hidden + text_embeds)\n{'='*60}")

    check_weight_dtypes("clip_g")

    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    import onnxruntime as ort

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FP32 model (same as ONNX export path)
    print("  Loading PyTorch model (FP32)...")
    model = CLIPTextModelWithProjection.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder_2"), local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().to(device)

    # Test input
    tokenizer = CLIPTokenizer.from_pretrained(str(DIFFUSERS_DIR / "tokenizer_2"), local_files_only=True)
    tokens = tokenizer(
        "1girl, masterpiece, best quality",
        padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    )
    input_ids = tokens["input_ids"]

    # PyTorch FP32: get hidden_states[-2] + text_embeds
    with torch.no_grad():
        pt_out = model(input_ids.to(device), output_hidden_states=True)
    pt_penultimate = pt_out.hidden_states[-2].cpu().numpy()  # [1, 77, 1280]
    pt_text_embeds = pt_out.text_embeds.cpu().numpy()  # [1, 1280]
    print(f"  PyTorch penultimate hidden: {pt_penultimate.shape}, range=[{pt_penultimate.min():.4f}, {pt_penultimate.max():.4f}]")
    print(f"  PyTorch text_embeds: {pt_text_embeds.shape}, range=[{pt_text_embeds.min():.4f}, {pt_text_embeds.max():.4f}]")

    # ONNX inference — now exports penultimate_hidden + text_embeds
    onnx_path = ONNX_DIR / "clip_g.onnx"
    if onnx_path.exists():
        print(f"  Loading ONNX: {onnx_path}...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        print(f"  ONNX output names: {out_names}")
        ort_out = sess.run(None, {"input_ids": input_ids.numpy().astype(np.int64)})

        # Match outputs by name
        ort_hidden = ort_embeds = None
        for i, name in enumerate(out_names):
            arr = ort_out[i]
            if "hidden" in name:
                ort_hidden = arr
            elif "embed" in name:
                ort_embeds = arr
            elif len(arr.shape) == 3:
                ort_hidden = arr
            elif len(arr.shape) == 2:
                ort_embeds = arr

        if ort_hidden is not None:
            print(f"  ONNX penultimate_hidden: {ort_hidden.shape}, range=[{ort_hidden.min():.4f}, {ort_hidden.max():.4f}]")
            print(f"  PyTorch FP32 vs ONNX (penultimate hidden):")
            compare("pt_fp32_vs_onnx_hidden", pt_penultimate, ort_hidden)
        if ort_embeds is not None:
            print(f"  ONNX text_embeds: {ort_embeds.shape}, range=[{ort_embeds.min():.4f}, {ort_embeds.max():.4f}]")
            print(f"  PyTorch FP32 vs ONNX (text_embeds):")
            compare("pt_fp32_vs_onnx_embeds", pt_text_embeds, ort_embeds)
        del sess

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()


def verify_vae():
    """Verify VAE decoder: PyTorch vs ONNX."""
    print(f"\n{'='*60}\n  Verifying VAE Decoder\n{'='*60}")

    check_weight_dtypes("vae")

    from diffusers import AutoencoderKL
    import onnxruntime as ort

    # Load PyTorch model
    print("  Loading PyTorch model (FP16)...")
    vae = AutoencoderKL.from_pretrained(
        str(DIFFUSERS_DIR / "vae"), local_files_only=True,
        torch_dtype=torch.float16,
    ).eval()

    # Test input — deterministic latent
    torch.manual_seed(42)
    latent_fp16 = torch.randn(1, 4, 128, 128, dtype=torch.float16)
    latent_fp32 = latent_fp16.float()

    # PyTorch inference (FP16)
    with torch.no_grad():
        pt_out = vae.decode(latent_fp16, return_dict=False)[0]
    pt_image = pt_out.cpu().float().numpy()
    print(f"  PyTorch FP16 image: {pt_image.shape}, range=[{pt_image.min():.4f}, {pt_image.max():.4f}]")

    # PyTorch inference (FP32 — same as export path)
    vae_f32 = vae.float()
    with torch.no_grad():
        pt_out_f32 = vae_f32.decode(latent_fp32, return_dict=False)[0]
    pt_image_f32 = pt_out_f32.cpu().numpy()
    print(f"  PyTorch FP32 image: {pt_image_f32.shape}, range=[{pt_image_f32.min():.4f}, {pt_image_f32.max():.4f}]")
    print(f"  PyTorch FP32 vs FP16:")
    compare("fp32_vs_fp16_image", pt_image_f32, pt_image)

    # ONNX inference (original FP32 export)
    onnx_path = ONNX_DIR / "vae_decoder.onnx"
    if onnx_path.exists():
        print(f"  Loading ONNX model: {onnx_path}...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {"latent": latent_fp32.numpy()}
        ort_out = sess.run(None, ort_inputs)
        ort_image = ort_out[0]
        print(f"  ONNX image: {ort_image.shape}, range=[{ort_image.min():.4f}, {ort_image.max():.4f}]")
        print(f"  PyTorch FP32 vs ONNX:")
        compare("pt_fp32_vs_onnx", pt_image_f32, ort_image)
        print(f"  PyTorch FP16 vs ONNX:")
        compare("pt_fp16_vs_onnx", pt_image, ort_image)
        del sess

    # Prepared ONNX (GroupNorm rewrite)
    if PREP_VAE.exists():
        print(f"  Loading prepared ONNX: {PREP_VAE}...")
        sess2 = ort.InferenceSession(str(PREP_VAE), providers=["CPUExecutionProvider"])
        ort_inputs2 = {"latent": latent_fp32.numpy()}
        ort_out2 = sess2.run(None, ort_inputs2)
        ort_image2 = ort_out2[0]
        print(f"  Prepared ONNX image: {ort_image2.shape}, range=[{ort_image2.min():.4f}, {ort_image2.max():.4f}]")
        print(f"  ONNX orig vs Prepared (GroupNorm rewrite):")
        if onnx_path.exists():
            compare("onnx_vs_prepared", ort_out[0] if onnx_path.exists() else pt_image_f32, ort_image2)
        print(f"  PyTorch FP32 vs Prepared:")
        compare("pt_fp32_vs_prepared", pt_image_f32, ort_image2)
        del sess2

    del vae, vae_f32
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["clip_l", "clip_g", "vae", "all"], default="all")
    args = ap.parse_args()

    if args.component in ("clip_l", "all"):
        verify_clip_l()
    if args.component in ("clip_g", "all"):
        verify_clip_g()
    if args.component in ("vae", "all"):
        verify_vae()

    print("All verifications complete!")


if __name__ == "__main__":
    main()
