#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
import onnxruntime as ort

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SDXL_NPU_ROOT = WORKSPACE_ROOT / "sdxl_npu"
if str(SDXL_NPU_ROOT) not in sys.path:
    sys.path.insert(0, str(SDXL_NPU_ROOT))

from export_sdxl_to_onnx import (  # noqa: E402
    collect_unet_resnet_conditioning_modules,
    compute_external_resnet_biases,
    infer_unet_resnet_spatial_shapes,
    install_external_resnet_bias_surgery,
)
from quantize_unet import prepare_model_for_ort_quantization  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare PyTorch SDXL UNet output against exported extmaps ONNX output")
    ap.add_argument("--pipeline-dir", default="d:/platform-tools/sdxl_npu/diffusers_pipeline")
    ap.add_argument("--onnx", default="d:/platform-tools/sdxl_npu/onnx_export_extmaps_groupnorm18_pruned_nocast_fp16bias/unet.onnx")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--step-index", type=int, default=0)
    ap.add_argument("--branch", choices=["cond", "uncond"], default="cond")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="d:/platform-tools/NPU/outputs/unet_pytorch_vs_onnx_step0_cond.json")
    return ap.parse_args()


def _tensor_stats(x: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(np.isfinite(x).sum()),
        "nan": int(np.isnan(x).sum()),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
    }


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    mask = np.isfinite(av) & np.isfinite(bv)
    if not np.any(mask):
        return {"finite_overlap": 0}
    av = av[mask]
    bv = bv[mask]
    diff = av - bv
    return {
        "finite_overlap": int(mask.sum()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": float(np.dot(av, bv) / ((np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12)),
    }


def _cast_for_input(value: np.ndarray, ort_type: str) -> np.ndarray:
    if ort_type == "tensor(float16)":
        return value.astype(np.float16, copy=False)
    if ort_type == "tensor(float)":
        return value.astype(np.float32, copy=False)
    if ort_type == "tensor(double)":
        return value.astype(np.float64, copy=False)
    raise RuntimeError(f"Unsupported ONNX input type: {ort_type}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    device = torch.device(args.device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pipeline_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet
    unet.eval()
    install_external_resnet_bias_surgery(unet)

    with torch.inference_mode():
        pe, ne, pp, npool = pipe.encode_prompt(
            prompt=args.prompt,
            prompt_2=args.prompt,
            negative_prompt=args.negative,
            negative_prompt_2=args.negative,
            do_classifier_free_guidance=True,
            device=device,
            num_images_per_prompt=1,
        )

    add_time_ids = pipe._get_add_time_ids(
        original_size=(args.height, args.width),
        crops_coords_top_left=(0, 0),
        target_size=(args.height, args.width),
        dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(device)

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timestep_scalar = pipe.scheduler.timesteps[args.step_index]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 4, args.height // 8, args.width // 8), generator=generator, device=device, dtype=torch.float32)
    latents = latents * pipe.scheduler.init_noise_sigma
    for i, t in enumerate(pipe.scheduler.timesteps):
        if i == args.step_index:
            timestep = t.reshape(1)
            sample = pipe.scheduler.scale_model_input(latents, t)
            break
    else:
        raise RuntimeError(f"Invalid step_index={args.step_index} for steps={args.steps}")

    if args.branch == "cond":
        encoder_hidden_states = pe
        text_embeds = pp
    else:
        encoder_hidden_states = ne
        text_embeds = npool

    with torch.no_grad():
        pytorch_out = unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": add_time_ids},
            return_dict=False,
        )[0].detach().cpu().numpy().astype(np.float32)

    spatial_shapes = infer_unet_resnet_spatial_shapes(unet, latent_h=args.height // 8, latent_w=args.width // 8)
    resnet_sites = collect_unet_resnet_conditioning_modules(unet)
    biases = compute_external_resnet_biases(unet, sample, timestep, encoder_hidden_states, text_embeds, add_time_ids)
    expanded_biases = {
        module._copilot_external_bias_name: bias.expand(-1, -1, *spatial_shapes[name]).detach().cpu().numpy().astype(np.float32)
        for (name, module), bias in zip(resnet_sites, biases)
    }

    normalized_onnx = str(Path(args.onnx))
    normalized_candidate = str(out_path.with_name(Path(args.onnx).stem + "_ort_normalized.onnx"))
    try:
        session = ort.InferenceSession(normalized_onnx, providers=["CPUExecutionProvider"])
    except Exception:
        changed = prepare_model_for_ort_quantization(str(Path(args.onnx)), normalized_candidate)
        if not changed:
            normalized_candidate = str(Path(args.onnx))
        normalized_onnx = normalized_candidate
        session = ort.InferenceSession(normalized_onnx, providers=["CPUExecutionProvider"])
    ort_inputs: dict[str, np.ndarray] = {}
    input_meta = {inp.name: inp.type for inp in session.get_inputs()}

    source_map: dict[str, np.ndarray] = {
        "sample": sample.detach().cpu().numpy().astype(np.float32),
        "timestep": timestep.detach().cpu().numpy().astype(np.float32),
        "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
        "text_embeds": text_embeds.detach().cpu().numpy().astype(np.float32),
        "time_ids": add_time_ids.detach().cpu().numpy().astype(np.float32),
    }
    time_ids_np = add_time_ids.detach().cpu().numpy().astype(np.float32)
    for i in range(6):
        source_map[f"time_id_{i}"] = time_ids_np[:, i : i + 1]
    source_map.update(expanded_biases)

    missing_inputs: list[str] = []
    for name, ort_type in input_meta.items():
        if name not in source_map:
            missing_inputs.append(name)
            continue
        ort_inputs[name] = _cast_for_input(source_map[name], ort_type)
    if missing_inputs:
        raise RuntimeError(f"Could not map ONNX inputs: {missing_inputs}")

    ort_out = session.run(None, ort_inputs)[0]
    ort_out = ort_out.astype(np.float32, copy=False)
    if ort_out.shape == (1, 128, 128, 4):
        ort_out = np.transpose(ort_out, (0, 3, 1, 2)).astype(np.float32, copy=False)

    report = {
        "prompt": args.prompt,
        "negative": args.negative,
        "steps": args.steps,
        "seed": args.seed,
        "step_index": args.step_index,
        "branch": args.branch,
        "onnx": str(Path(args.onnx)),
        "onnx_used": normalized_onnx,
        "timestep": float(timestep_scalar.detach().cpu().item()),
        "onnx_input_types": input_meta,
        "pytorch_output": _tensor_stats(pytorch_out),
        "onnx_output": _tensor_stats(ort_out),
        "diff": _diff_metrics(ort_out, pytorch_out),
        "trigger_names": list(expanded_biases.keys()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_path.with_suffix('.onnx_output.npy'), ort_out)
    print(f"[ok] saved report: {out_path}")
    print(json.dumps(report["diff"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
