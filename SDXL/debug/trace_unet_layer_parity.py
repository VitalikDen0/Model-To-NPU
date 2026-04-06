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


TARGET_CLASSES = {
    "Conv2d",
    "GroupNorm",
    "ResnetBlock2D",
    "BasicTransformerBlock",
    "Transformer2DModel",
    "Attention",
    "Downsample2D",
    "Upsample2D",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trace SDXL UNet layer parity between internal path and extmaps bias-injected path")
    ap.add_argument("--pipeline-dir", default="d:/platform-tools/sdxl_npu/diffusers_pipeline")
    ap.add_argument("--runtime-work", default="d:/platform-tools/NPU/runtime_work")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--step-index", type=int, default=0)
    ap.add_argument("--branch", choices=["cond", "uncond"], default="cond")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="d:/platform-tools/NPU/outputs/unet_layer_parity_step0_cond.json")
    return ap.parse_args()


def _extract_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_stats(x: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(x)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(finite.sum()),
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


def _capture_run(unet: torch.nn.Module, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, text_embeds: torch.Tensor, time_ids: torch.Tensor, target_names: set[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    captured: dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is None:
                return
            captured[name] = tensor.detach().cpu().to(torch.float32).numpy().copy()
        return _hook

    for name, module in unet.named_modules():
        if name in target_names:
            hooks.append(module.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        out = unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0].detach().cpu().to(torch.float32).numpy().copy()

    for hook in hooks:
        hook.remove()
    return out, captured


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

    install_external_resnet_bias_surgery(unet)
    resnet_sites = collect_unet_resnet_conditioning_modules(unet)
    spatial_shapes = infer_unet_resnet_spatial_shapes(unet, latent_h=args.height // 8, latent_w=args.width // 8)
    biases = compute_external_resnet_biases(unet, sample, timestep, encoder_hidden_states, text_embeds, add_time_ids)

    target_names: set[str] = {
        name for name, module in unet.named_modules() if module.__class__.__name__ in TARGET_CLASSES
    }
    target_names.update(name for name, _ in resnet_sites)
    target_names.update({"conv_in", "conv_norm_out", "conv_act", "conv_out"})

    for _, module in resnet_sites:
        module._copilot_external_bias = None
    internal_out, internal_layers = _capture_run(unet, sample, timestep, encoder_hidden_states, text_embeds, add_time_ids, target_names)

    runtime_trigger_report: list[dict[str, Any]] = []
    runtime_step_dir = Path(args.runtime_work) / f"step_{args.step_index:03d}_{args.branch}"
    for (name, module), bias in zip(resnet_sites, biases):
        h, w = spatial_shapes[name]
        expanded = bias.expand(-1, -1, h, w).detach().cpu().to(torch.float32).numpy().copy()
        module._copilot_external_bias = bias
        entry: dict[str, Any] = {
            "name": name,
            "external_bias_name": getattr(module, "_copilot_external_bias_name", name),
            "computed_bias_stats": _tensor_stats(expanded),
        }
        raw_path = runtime_step_dir / f"{module._copilot_external_bias_name}.raw"
        if raw_path.exists():
            runtime_bias = np.fromfile(raw_path, dtype=np.float16).astype(np.float32).reshape(expanded.shape)
            entry["runtime_bias_stats"] = _tensor_stats(runtime_bias)
            entry["runtime_bias_diff"] = _diff_metrics(runtime_bias, expanded)
        runtime_trigger_report.append(entry)

    external_out, external_layers = _capture_run(unet, sample, timestep, encoder_hidden_states, text_embeds, add_time_ids, target_names)

    for _, module in resnet_sites:
        module._copilot_external_bias = None

    layer_reports: list[dict[str, Any]] = []
    common_names = sorted(set(internal_layers) & set(external_layers))
    for name in common_names:
        layer_reports.append(
            {
                "name": name,
                "internal": _tensor_stats(internal_layers[name]),
                "external": _tensor_stats(external_layers[name]),
                "diff": _diff_metrics(external_layers[name], internal_layers[name]),
            }
        )

    layer_reports.sort(key=lambda item: item["diff"].get("max_abs", -1.0), reverse=True)

    report = {
        "prompt": args.prompt,
        "negative": args.negative,
        "steps": args.steps,
        "seed": args.seed,
        "step_index": args.step_index,
        "branch": args.branch,
        "timestep": float(timestep_scalar.detach().cpu().item()),
        "sample_stats": _tensor_stats(sample.detach().cpu().numpy().astype(np.float32)),
        "encoder_hidden_states_stats": _tensor_stats(encoder_hidden_states.detach().cpu().numpy().astype(np.float32)),
        "text_embeds_stats": _tensor_stats(text_embeds.detach().cpu().numpy().astype(np.float32)),
        "time_ids_stats": _tensor_stats(add_time_ids.detach().cpu().numpy().astype(np.float32)),
        "bias_count": len(biases),
        "final_output_internal": _tensor_stats(internal_out),
        "final_output_external": _tensor_stats(external_out),
        "final_output_diff": _diff_metrics(external_out, internal_out),
        "runtime_trigger_report": runtime_trigger_report,
        "layer_count": len(layer_reports),
        "top_layer_diffs": layer_reports[:40],
        "all_layer_diffs": layer_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved report: {out_path}")
    print(json.dumps({
        "final_output_diff": report["final_output_diff"],
        "bias_count": report["bias_count"],
        "top_5_layer_diffs": report["top_layer_diffs"][:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
