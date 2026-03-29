#!/usr/bin/env python3
"""
Export CLIP-L, CLIP-G text encoders and VAE decoder to ONNX for QNN conversion.

IMPORTANT: SDXL pipeline uses hidden_states[-2] (penultimate layer), NOT last_hidden_state.
This is because the final layer of some CLIP models can produce NaN in FP16.

CLIP-L: token_ids[1,77] → penultimate_hidden[1,77,768]
CLIP-G: token_ids[1,77] → penultimate_hidden[1,77,1280], text_embeds[1,1280]
VAE:    latent[1,4,128,128] → image[1,3,1024,1024]

Usage:
  python NPU/export_clip_vae_to_onnx.py --component clip_l
  python NPU/export_clip_vae_to_onnx.py --component clip_g
  python NPU/export_clip_vae_to_onnx.py --component vae
  python NPU/export_clip_vae_to_onnx.py --component all
  python NPU/export_clip_vae_to_onnx.py --component all --force  # overwrite existing
"""
import argparse, gc, os, sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SDXL_NPU = ROOT / "sdxl_npu"
DIFFUSERS_DIR = SDXL_NPU / "diffusers_pipeline"
ONNX_OUT = SDXL_NPU / "onnx_clip_vae"


class CLIPLWrapper(torch.nn.Module):
    """Wrapper: returns hidden_states[-2] like SDXL pipeline."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        # SDXL uses penultimate hidden state
        penultimate = out.hidden_states[-2]
        return penultimate


class CLIPGWrapper(torch.nn.Module):
    """Wrapper: returns hidden_states[-2] + text_embeds like SDXL pipeline."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        # SDXL uses penultimate hidden state + pooled text_embeds
        penultimate = out.hidden_states[-2]
        text_embeds = out[0]  # text_embeds (pooled projection)
        return penultimate, text_embeds


def export_clip_l(out_dir: Path, opset: int = 17, force: bool = False):
    """Export CLIP-L text encoder: hidden_states[-2] output."""
    from transformers import CLIPTextModel

    out_path = out_dir / "clip_l.onnx"
    if out_path.exists() and not force:
        print(f"[skip] {out_path} already exists")
        return

    print("[clip_l] Loading from diffusers pipeline...")
    model = CLIPTextModel.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder"),
        local_files_only=True,
    )
    model.eval()
    # Keep native FP16 weights, export wrapper handles dtype
    wrapper = CLIPLWrapper(model).eval()

    dummy_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick sanity: verify no NaN in penultimate layer
    with torch.no_grad():
        test_out = wrapper(dummy_ids)
    assert not torch.isnan(test_out).any(), "NaN in penultimate hidden state!"
    print(f"  sanity check: penultimate hidden range=[{test_out.min():.4f}, {test_out.max():.4f}]")

    print(f"[clip_l] Exporting to {out_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        str(out_path),
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["penultimate_hidden"],
        dynamic_axes=None,  # fixed shape for NPU
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[clip_l] Done: {out_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx
    onnx.checker.check_model(str(out_path), full_check=False)
    m = onnx.load(str(out_path), load_external_data=False)
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f"  inputs: {inputs}")
    print(f"  outputs: {outputs}")
    del model, wrapper, m
    gc.collect()


def export_clip_g(out_dir: Path, opset: int = 17, force: bool = False):
    """Export CLIP-G text encoder: hidden_states[-2] + text_embeds."""
    from transformers import CLIPTextModelWithProjection

    out_path = out_dir / "clip_g.onnx"
    if out_path.exists() and not force:
        print(f"[skip] {out_path} already exists")
        return

    print("[clip_g] Loading from diffusers pipeline...")
    model = CLIPTextModelWithProjection.from_pretrained(
        str(DIFFUSERS_DIR / "text_encoder_2"),
        local_files_only=True,
    )
    model.eval()
    # Keep native FP16 weights
    wrapper = CLIPGWrapper(model).eval()

    dummy_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check
    with torch.no_grad():
        test_hidden, test_embeds = wrapper(dummy_ids)
    assert not torch.isnan(test_hidden).any(), "NaN in penultimate hidden state!"
    assert not torch.isnan(test_embeds).any(), "NaN in text_embeds!"
    print(f"  sanity: hidden range=[{test_hidden.min():.4f}, {test_hidden.max():.4f}]")
    print(f"  sanity: text_embeds range=[{test_embeds.min():.4f}, {test_embeds.max():.4f}]")

    print(f"[clip_g] Exporting to {out_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        str(out_path),
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["penultimate_hidden", "text_embeds"],
        dynamic_axes=None,  # fixed shape for NPU
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[clip_g] Done: {out_path} ({size_mb:.1f} MB)")

    # Validate — use path-based check for potentially large models
    import onnx
    onnx.checker.check_model(str(out_path), full_check=False)
    m = onnx.load(str(out_path), load_external_data=False)
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f"  inputs: {inputs}")
    print(f"  outputs: {outputs}")
    del model, wrapper, m
    gc.collect()


def export_vae_decoder(out_dir: Path, opset: int = 17, force: bool = False):
    """Export VAE decoder from diffusers pipeline."""
    from diffusers import AutoencoderKL

    out_path = out_dir / "vae_decoder.onnx"
    if out_path.exists() and not force:
        print(f"[skip] {out_path} already exists")
        return

    print("[vae] Loading from diffusers pipeline...")
    vae = AutoencoderKL.from_pretrained(
        str(DIFFUSERS_DIR / "vae"),
        local_files_only=True,
    )
    # VAE weights are FP16, ONNX export requires float32 for tracing
    vae.eval()
    vae = vae.to(torch.float32)

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent, return_dict=False)[0]

    wrapper = VAEDecoderWrapper(vae)
    wrapper.eval()

    dummy_latent = torch.randn(1, 4, 128, 128, dtype=torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vae] Exporting decoder to {out_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_latent,),
        str(out_path),
        opset_version=opset,
        dynamo=False,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes=None,  # fixed shape for NPU
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[vae] Done: {out_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx
    onnx.checker.check_model(str(out_path), full_check=False)
    m = onnx.load(str(out_path), load_external_data=False)
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f"  inputs: {inputs}")
    print(f"  outputs: {outputs}")
    del wrapper, vae, m
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["clip_l", "clip_g", "vae", "all"], default="all")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--force", action="store_true", help="Overwrite existing ONNX files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ONNX_OUT
    force = args.force
    print(f"[config] component={args.component}, out={out_dir}, opset={args.opset}, force={force}")

    if args.component in ("clip_l", "all"):
        export_clip_l(out_dir, args.opset, force)
    if args.component in ("clip_g", "all"):
        export_clip_g(out_dir, args.opset, force)
    if args.component in ("vae", "all"):
        export_vae_decoder(out_dir, args.opset, force)

    print("\n[done] Export complete!")


if __name__ == "__main__":
    main()
