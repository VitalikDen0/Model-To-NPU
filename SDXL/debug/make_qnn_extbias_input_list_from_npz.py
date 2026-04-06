#!/usr/bin/env python3
"""Generate QNN converter input_list for external-featuremaps UNet ONNX.

This variant of the exported UNet consumes only:
  - sample
  - encoder_hidden_states
  - 17 external resnet_bias_* featuremaps

The featuremaps are computed from the original UNet using representative
calibration tensors stored in calibration.npz:
  - sample
  - encoder_hidden_states
  - text_embeds
  - time_ids

Output format matches QNN converter input_list expectations: one sample per line
with input files ordered exactly like the ONNX graph inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

from export_sdxl_to_onnx import (
    collect_unet_resnet_conditioning_modules,
    compute_external_resnet_biases,
    infer_unet_resnet_spatial_shapes,
)


CONFIG = {
    "sample_size": 128,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "only_cross_attention": False,
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 2,
    "downsample_padding": 1,
    "mid_block_scale_factor": 1,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "cross_attention_dim": 2048,
    "transformer_layers_per_block": [1, 2, 10],
    "attention_head_dim": [5, 10, 20],
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to calibration.npz")
    ap.add_argument("--diffusers-dir", help="Path to diffusers pipeline directory (preferred)")
    ap.add_argument("--unet-safetensors", help="Path to source UNet safetensors in diffusers-compatible keyspace")
    ap.add_argument("--out-dir", required=True, help="Directory for raw tensors + input_list")
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--list-name", default="unet_extbias_input_list.txt")
    ap.add_argument("--device", default="cpu", help="Torch device for bias computation")
    return ap.parse_args()


def load_unet(unet_safetensors: Path, device: torch.device) -> UNet2DConditionModel:
    model = UNet2DConditionModel(**CONFIG)
    state_dict = load_file(str(unet_safetensors))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict non-strict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device=device, dtype=torch.float32)
    return model


def load_unet_from_diffusers(diffusers_dir: Path, device: torch.device) -> UNet2DConditionModel:
    model = UNet2DConditionModel.from_pretrained(
        str(diffusers_dir),
        subfolder="unet",
        torch_dtype=torch.float32,
    )
    model.eval().to(device=device, dtype=torch.float32)
    return model


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(args.device)
    target_dtype_np = np.float16 if args.dtype == "float16" else np.float32

    if not args.diffusers_dir and not args.unet_safetensors:
        raise ValueError("Provide either --diffusers-dir or --unet-safetensors")

    data = np.load(npz_path, mmap_mode="r")
    total = int(data["sample"].shape[0])
    sample_count = min(total, args.max_samples)

    if args.diffusers_dir:
        model_source = Path(args.diffusers_dir).resolve()
        model = load_unet_from_diffusers(model_source, torch_device)
    else:
        model_source = Path(args.unet_safetensors).resolve()
        model = load_unet(model_source, torch_device)

    sample_shape = tuple(int(v) for v in data["sample"].shape[1:])
    spatial_shapes = infer_unet_resnet_spatial_shapes(model, latent_h=sample_shape[1], latent_w=sample_shape[2])
    resnet_modules = list(collect_unet_resnet_conditioning_modules(model))
    resnet_bias_names = [
        f"resnet_bias_{index:02d}_{name.replace('.', '_')}"
        for index, (name, _) in enumerate(resnet_modules)
    ]

    input_list_lines: list[str] = []

    for i in range(sample_count):
        sample_dir = out_dir / f"sample_{i:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample = np.asarray(data["sample"][i:i + 1], dtype=np.float32)
        timestep = np.asarray(data["timestep"][i:i + 1], dtype=np.float32)
        encoder_hidden_states = np.asarray(data["encoder_hidden_states"][i:i + 1], dtype=np.float32)
        text_embeds = np.asarray(data["text_embeds"][i:i + 1], dtype=np.float32)
        time_ids = np.asarray(data["time_ids"][i:i + 1], dtype=np.float32)

        with torch.no_grad():
            bias_tensors = compute_external_resnet_biases(
                model,
                torch.from_numpy(sample).to(torch_device),
                torch.from_numpy(timestep).to(torch_device),
                torch.from_numpy(encoder_hidden_states).to(torch_device),
                torch.from_numpy(text_embeds).to(torch_device),
                torch.from_numpy(time_ids).to(torch_device),
            )

        ordered_paths: list[str] = []

        sample_path = sample_dir / "sample.raw"
        np.asarray(sample, dtype=target_dtype_np).tofile(sample_path)
        ordered_paths.append(str(sample_path.resolve()))

        ehs_path = sample_dir / "encoder_hidden_states.raw"
        np.asarray(encoder_hidden_states, dtype=target_dtype_np).tofile(ehs_path)
        ordered_paths.append(str(ehs_path.resolve()))

        for bias_name, bias_tensor, (module_name, _) in zip(resnet_bias_names, bias_tensors, resnet_modules):
            bias_path = sample_dir / f"{bias_name}.raw"
            h, w = spatial_shapes[module_name]
            bias_array = bias_tensor.expand(-1, -1, h, w).detach().cpu().numpy().astype(target_dtype_np, copy=False)
            bias_array.tofile(bias_path)
            ordered_paths.append(str(bias_path.resolve()))

        input_list_lines.append(" ".join(ordered_paths))

        if (i + 1) % 4 == 0 or i + 1 == sample_count:
            print(f"[info] prepared {i + 1}/{sample_count} samples")

    list_path = out_dir / args.list_name
    list_path.write_text("\n".join(input_list_lines) + "\n", encoding="utf-8")

    print(f"[ok] npz: {npz_path}")
    print(f"[ok] model_source: {model_source}")
    print(f"[ok] samples: {sample_count}/{total}")
    print(f"[ok] input_list: {list_path}")


if __name__ == "__main__":
    main()