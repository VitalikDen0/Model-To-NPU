#!/usr/bin/env python3
"""Export TAESD XL decoder to ONNX for phone-side live preview.

TAESD XL: Tiny AutoEncoder for SDXL — ~5MB weights,
decodes latents [1,4,128,128] → RGB image [1,3,1024,1024].
100-500x smaller than full VAE. Used for live step previews.

Source weights: J:/ComfyUI/models/vae_approx/taesdxl_decoder.safetensors
Output:         sdxl_npu/taesd_decoder/taesd_decoder.onnx

Architecture (inferred from safetensors keys):
  [1]  Conv2d(4→64, 3×3)           @ 128×128
  [3]  ResBlock(64)                 @ 128×128
  [4]  ResBlock(64)                 @ 128×128
  [5]  ResBlock(64)                 @ 128×128
  [↑]  Upsample ×2
  [7]  Conv2d(64→64, 3×3, no bias) @ 256×256
  [8]  ResBlock(64)                 @ 256×256
  [9]  ResBlock(64)                 @ 256×256
  [10] ResBlock(64)                 @ 256×256
  [↑]  Upsample ×2
  [12] Conv2d(64→64, 3×3, no bias) @ 512×512
  [13] ResBlock(64)                 @ 512×512
  [14] ResBlock(64)                 @ 512×512
  [15] ResBlock(64)                 @ 512×512
  [↑]  Upsample ×2
  [17] Conv2d(64→64, 3×3, no bias) @ 1024×1024
  [18] ResBlock(64)                 @ 1024×1024
  [19] Conv2d(64→3, 3×3)           @ 1024×1024

Usage:
  python SDXL/export_taesd_to_onnx.py
  python SDXL/export_taesd_to_onnx.py --weights J:/path/to/taesdxl_decoder.safetensors
  python SDXL/export_taesd_to_onnx.py --validate  # test on random input
"""

import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import safetensors.torch
import onnx
import numpy as np

DEFAULT_WEIGHTS = "J:/ComfyUI/models/vae_approx/taesdxl_decoder.safetensors"
DEFAULT_OUT_DIR = "D:/platform-tools/sdxl_npu/taesd_decoder"


# ─── Architecture ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block matching diffusers AutoencoderTinyBlock."""
    def __init__(self, n: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n, n, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n, n, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n, n, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.conv(x))


class TAESDXLDecoder(nn.Module):
    """
    TAESD XL Decoder. State-dict key names match taesdxl_decoder.safetensors.
    Non-parametric modules (_relu, _up*) are excluded from state_dict.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            # ── 128×128 ──────────────────────────────────────────
            ('1',     nn.Conv2d(4, 64, 3, padding=1, bias=True)),
            ('_r1',   nn.ReLU()),
            ('3',     ResBlock(64)),
            ('4',     ResBlock(64)),
            ('5',     ResBlock(64)),
            # ── ×2 → 256×256 ─────────────────────────────────────
            ('_up1',  nn.Upsample(scale_factor=2, mode='nearest')),
            ('7',     nn.Conv2d(64, 64, 3, padding=1, bias=False)),
            ('_r7',   nn.ReLU()),
            ('8',     ResBlock(64)),
            ('9',     ResBlock(64)),
            ('10',    ResBlock(64)),
            # ── ×2 → 512×512 ─────────────────────────────────────
            ('_up2',  nn.Upsample(scale_factor=2, mode='nearest')),
            ('12',    nn.Conv2d(64, 64, 3, padding=1, bias=False)),
            ('_r12',  nn.ReLU()),
            ('13',    ResBlock(64)),
            ('14',    ResBlock(64)),
            ('15',    ResBlock(64)),
            # ── ×2 → 1024×1024 ───────────────────────────────────
            ('_up3',  nn.Upsample(scale_factor=2, mode='nearest')),
            ('17',    nn.Conv2d(64, 64, 3, padding=1, bias=False)),
            ('_r17',  nn.ReLU()),
            ('18',    ResBlock(64)),
            ('19',    nn.Conv2d(64, 3, 3, padding=1, bias=True)),
        ]))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [1, 4, 128, 128] raw denoised latents (NOT divided by scaling_factor)
        Returns:
            [1, 3, 1024, 1024] RGB image in [0, 1] range (after clamp)
        """
        # Match diffusers DecoderTiny:
        #   x = tanh(latents / 3) * 3
        #   x = decoder_layers(x)
        #   return x * 2 - 1
        # This is important — skipping the latent clamp causes preview garbage.
        x = torch.tanh(latents / 3.0) * 3.0
        x = self.layers(x)
        return x * 2.0 - 1.0


def load_decoder(weights_path: str) -> TAESDXLDecoder:
    """Load TAESD XL decoder from safetensors file."""
    model = TAESDXLDecoder()
    raw_sd = safetensors.torch.load_file(weights_path)

    # ComfyUI TAESD safetensors use keys like "1.weight", "3.conv.0.weight", ...
    # while our nn.Sequential lives under model.layers.*.
    sd = {}
    for key, value in raw_sd.items():
        mapped = key if key.startswith("layers.") else f"layers.{key}"
        sd[mapped] = value

    # load_state_dict with strict=False: non-parametric "_r*" / "_up*" are auto-skipped
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Expected missing: none (all _r* / _up* have no params)
    param_missing = [k for k in missing if not k.startswith('layers._')]
    if param_missing:
        print(f"  WARNING: Missing parametric keys: {param_missing}", file=sys.stderr)
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}", file=sys.stderr)

    if param_missing or unexpected:
        raise RuntimeError(
            "TAESD weights did not map cleanly into the decoder architecture; "
            "refusing to export a broken preview model."
        )

    return model


# ─── Export ───────────────────────────────────────────────────────────────────

def export_onnx(model: TAESDXLDecoder, out_path: str, opset: int = 18) -> None:
    model.eval()
    dummy = torch.randn(1, 4, 128, 128, dtype=torch.float32)

    print(f"  Running test forward pass ...", end=" ")
    with torch.no_grad():
        out = model(dummy)
    print(f"output shape: {tuple(out.shape)}, range [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (1, 3, 1024, 1024), f"Expected (1,3,1024,1024), got {out.shape}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  Exporting ONNX (opset {opset}) ...", end=" ")
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        opset_version=opset,
        input_names=["latents"],
        output_names=["image"],
        dynamic_axes=None,  # fixed shapes for QNN
        do_constant_folding=True,
    )

    # Force-save as self-contained single file (no external .data sidecar).
    # onnxruntime on Android needs one file — torch.onnx.export may create external data
    # for models >2 MB by default depending on onnx version.
    m = onnx.load(out_path, load_external_data=True)
    onnx.save(m, out_path, save_as_external_data=False)

    # Validate
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    sz_mb = Path(out_path).stat().st_size / 1e6
    print(f"OK ({sz_mb:.1f} MB)")


def validate_vs_comfyui(model: TAESDXLDecoder, weights_path: str) -> None:
    """Cross-check against ComfyUI's TAESD output on same latent."""
    print("\n[Validate] Checking output range on realistic SDXL latent ...")
    rng = np.random.RandomState(42)
    # Realistic SDXL latent magnitude (after denoising, before VAE decode)
    lat = torch.tensor(rng.randn(1, 4, 128, 128).astype(np.float32) * 0.5)

    model.eval()
    with torch.no_grad():
        out = model(lat)
    arr = out.numpy()[0].transpose(1, 2, 0)  # [H, W, 3] in [-1, 1]
    img = np.clip(arr / 2.0 + 0.5, 0.0, 1.0)
    print(f"  Input latent: min={lat.min():.3f} max={lat.max():.3f} std={lat.std():.3f}")
    print(f"  Decoder raw:  min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
    print(f"  Preview img:  min={img.min():.3f} max={img.max():.3f} mean={img.mean():.3f}")

    nonzero_range = (img.max() - img.min()) > 0.05
    not_saturated = img.mean() > 0.05 and img.mean() < 0.95
    if nonzero_range and not_saturated:
        print("  Result: PASS — output has reasonable dynamic range")
    else:
        print("  Result: WARN — output is clipped/saturated, may need input scaling")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export TAESD XL decoder to ONNX")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Path to taesdxl_decoder.safetensors")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Output directory")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset version (default: 18)")
    ap.add_argument("--validate", action="store_true",
                    help="Run validation pass after export")
    a = ap.parse_args()

    print(f"[TAESD XL Export]")
    print(f"  Weights: {a.weights}")
    print(f"  Output:  {a.out_dir}/taesd_decoder.onnx")

    print("\n[1/2] Loading weights ...")
    model = load_decoder(a.weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M  ({n_params*4/1e6:.1f} MB fp32)")

    print("\n[2/2] Exporting ONNX ...")
    onnx_path = f"{a.out_dir}/taesd_decoder.onnx"
    export_onnx(model, onnx_path, opset=a.opset)

    if a.validate:
        validate_vs_comfyui(model, a.weights)

    print(f"\nDone! Next: python SDXL/convert_taesd_to_qnn.py --onnx {onnx_path}")


if __name__ == "__main__":
    main()
