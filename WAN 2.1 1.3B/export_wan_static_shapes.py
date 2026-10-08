#!/usr/bin/env python3
"""
Export WAN ONNX with static (non-dynamic) shapes for better QNN compatibility.
This fixes the Reshape shape inference issues in QNN compiler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
import torch

from diffusers import WanTransformer3DModel


def export_transformer_static(
    model_dir: Path,
    out_path: Path,
    *,
    height: int = 480,
    width: int = 832,
    num_frames: int = 17,
    opset: int = 17,
) -> None:
    """
    Export WAN transformer with static shapes for QNN compatibility.
    
    The key difference is that we ensure all shapes are explicitly defined
    without dynamic (-1) dimensions where possible.
    """
    print(f"Loading WAN model from {model_dir}...")
    
    # Load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    transformer.eval()
    transformer.to(dtype=torch.float16)
    
    print(f"Model loaded successfully")
    print(f"Export config: {height}x{width}, {num_frames} frames")
    
    # Create dummy inputs with STATIC shapes
    batch_size = 1
    in_channels = 16  # From config
    
    # Latent shape after VAE encoding
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    
    # Token shape after patch embedding
    patch_size = tuple(int(v) for v in transformer.config.patch_size)  # (1, 2, 2)
    num_tokens = (latent_frames // patch_size[0]) * (latent_h // patch_size[1]) * (latent_w // patch_size[2])
    
    hidden_dim = int(transformer.config.num_attention_heads) * int(transformer.config.attention_head_dim)
    text_seq_len = 128  # max from config
    text_dim = int(transformer.config.text_dim)
    
    print(f"\nComputed shapes:")
    print(f"  Latent: {batch_size}x{in_channels}x{latent_frames}x{latent_h}x{latent_w}")
    print(f"  Num tokens: {num_tokens}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # Create dummy tensors with STATIC shapes (no dynamic -1 dimensions)
    hidden_states = torch.randn(batch_size, num_tokens, hidden_dim, dtype=torch.float16)
    timestep = torch.randn(batch_size, dtype=torch.float16)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim, dtype=torch.float16)
    
    print(f"\nDummy input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    
    # Export to ONNX
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to ONNX (static shapes)...")
    torch.onnx.export(
        transformer,
        (hidden_states, timestep, encoder_hidden_states),
        str(out_path),
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=False,
        verbose=False,
    )
    
    print(f"✅ Export successful: {out_path}")
    print(f"   File size: {out_path.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export WAN 2.1 transformer with static shapes for QNN."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent / "downloads" / "int8-diffusers",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "output_static",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--opset", type=int, default=17)
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")
    
    out_file = args.out_dir / "transformer_static.onnx"
    
    try:
        export_transformer_static(
            args.model_dir,
            out_file,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            opset=args.opset,
        )
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
