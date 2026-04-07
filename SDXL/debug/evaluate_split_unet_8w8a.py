#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PLATFORM_TOOLS = WORKSPACE_ROOT.parent
DEFAULT_SDXL_NPU_ROOT = EXTERNAL_PLATFORM_TOOLS / "sdxl_npu"
SDXL_NPU_ROOT = DEFAULT_SDXL_NPU_ROOT if DEFAULT_SDXL_NPU_ROOT.exists() else (WORKSPACE_ROOT / "sdxl_npu")
if str(SDXL_NPU_ROOT) not in sys.path:
    sys.path.insert(0, str(SDXL_NPU_ROOT))

from assess_generated_image import assess_image  # noqa: E402
from export_split_unet import SDXLUNetEncoder  # noqa: E402
from quantize_unet import UNetCalibrationReader, prepare_model_for_ort_quantization  # noqa: E402

DEFAULT_NEG = (
    "lowres, bad anatomy, bad hands, text, error, worst quality, "
    "low quality, blurry"
)

GENTLE_EXCLUDE_SUBSTRINGS = [
    "conv_in",
    "conv_out",
    "conv_norm_out",
    "time_embed",
    "add_embedding",
    "softmax",
    "layernorm",
    "groupnorm",
    "attention",
    "attn",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
]
GENTLE_EXCLUDE_OP_TYPES = {
    "Softmax",
    "LayerNormalization",
    "GroupNormalization",
}
GENTLE_OP_TYPES_TO_QUANTIZE = [
    "Conv",
    "MatMul",
    "Gemm",
]


@dataclass
class StepSnapshot:
    step_index: int
    timestep: float
    sample: np.ndarray
    latents_before: np.ndarray
    latents_after: np.ndarray
    outputs: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    default_split_root = WORKSPACE_ROOT / "NPU" / "onnx_lightning_split"
    default_quant_root = SDXL_NPU_ROOT / "onnx_lightning_split_8w8a_gentle"
    default_out_dir = WORKSPACE_ROOT / "NPU" / "outputs" / "unet_split_8w8a_eval"

    ap = argparse.ArgumentParser(
        description=(
            "Safely evaluate a gentle split-UNet 8W8A candidate on PC before any phone-side "
            "deployment. Stage 1 compares tensors; Stage 2 optionally runs full denoising + "
            "VAE decode and compares images."
        )
    )
    ap.add_argument("--pipeline-dir", default=str(SDXL_NPU_ROOT / "diffusers_pipeline"))
    ap.add_argument("--split-root", default=str(default_split_root))
    ap.add_argument(
        "--calibration-npz",
        default=str(SDXL_NPU_ROOT / "calibration_data" / "1024x1024" / "calibration.npz"),
    )
    ap.add_argument("--quantized-root", default=str(default_quant_root))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--tensor-step-indices", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--calib-samples", type=int, default=16)
    ap.add_argument(
        "--force-requantize",
        action="store_true",
        help="Rebuild the gentle split 8W8A candidates even if output files already exist.",
    )
    ap.add_argument(
        "--force-full-cycle",
        action="store_true",
        help="Run the full image cycle even if the tensor gate calls the candidate bad.",
    )
    ap.add_argument(
        "--tensor-only",
        action="store_true",
        help="Stop after Stage 1 tensor parity and skip full image generation.",
    )
    ap.add_argument("--out-dir", default=str(default_out_dir))
    return ap.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.float32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tensor_stats(x: np.ndarray) -> dict[str, Any]:
    xf = x.astype(np.float64, copy=False)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(np.isfinite(xf).sum()),
        "nan": int(np.isnan(xf).sum()),
        "min": float(np.nanmin(xf)),
        "max": float(np.nanmax(xf)),
        "mean": float(np.nanmean(xf)),
        "std": float(np.nanstd(xf)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12
    return float(np.dot(av, bv) / denom)


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


def _parse_step_indices(spec: str, total_steps: int) -> list[int]:
    result: list[int] = []
    for part in spec.split(","):
        value = int(part.strip())
        if value < 0 or value >= total_steps:
            raise ValueError(f"step index {value} outside valid range [0, {total_steps - 1}]")
        result.append(value)
    return sorted(dict.fromkeys(result))


def _choose_ort_providers() -> list[str]:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_onnx_session(onnx_path: Path) -> tuple[ort.InferenceSession, str, list[str]]:
    providers = _choose_ort_providers()
    normalized_onnx = str(onnx_path)
    normalized_candidate = str(onnx_path.with_name(onnx_path.stem + "_ort_normalized.onnx"))

    def _make_session(path: str) -> ort.InferenceSession:
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        return ort.InferenceSession(path, sess_options=sess_options, providers=providers)

    try:
        session = _make_session(normalized_onnx)
    except Exception:
        changed = prepare_model_for_ort_quantization(str(onnx_path), normalized_candidate)
        if not changed:
            normalized_candidate = str(onnx_path)
        normalized_onnx = normalized_candidate
        session = _make_session(normalized_onnx)
    return session, normalized_onnx, providers


def _build_gentle_exclude_nodes(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    try:
        excluded: list[str] = []
        for node in model.graph.node:
            name = node.name or ""
            lowered = name.lower()
            if any(substr in lowered for substr in GENTLE_EXCLUDE_SUBSTRINGS):
                excluded.append(name)
                continue
            if node.op_type in GENTLE_EXCLUDE_OP_TYPES:
                excluded.append(name)
        return sorted(dict.fromkeys(excluded))
    finally:
        del model
        gc.collect()


class _EncoderCalibrationReader(CalibrationDataReader):
    def __init__(self, npz_path: Path, max_samples: int) -> None:
        self.inner = UNetCalibrationReader(str(npz_path), dtype="float32", max_samples=max_samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return self.inner.get_next()

    def rewind(self) -> None:
        self.inner.rewind()


class _DecoderCalibrationReader(CalibrationDataReader):
    def __init__(self, npz_path: Path, unet_dir: Path, device: str, max_samples: int) -> None:
        self.inner = UNetCalibrationReader(str(npz_path), dtype="float32", max_samples=max_samples)
        self.device = torch.device(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            str(unet_dir),
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(self.device)
        self.unet.eval()
        self.encoder = SDXLUNetEncoder(self.unet).to(self.device)
        self.encoder.eval()

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = self.inner.get_next()
        if batch is None:
            return None

        sample = torch.from_numpy(batch["sample"]).to(self.device, dtype=torch.float32)
        timestep = torch.from_numpy(batch["timestep"]).reshape(-1).to(self.device, dtype=torch.float32)
        encoder_hidden_states = torch.from_numpy(batch["encoder_hidden_states"]).to(self.device, dtype=torch.float32)
        text_embeds = torch.from_numpy(batch["text_embeds"]).to(self.device, dtype=torch.float32)
        time_ids = torch.from_numpy(batch["time_ids"]).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            enc_outputs = self.encoder(sample, timestep, encoder_hidden_states, text_embeds, time_ids)

        names = ["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb"]
        enriched = {
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in zip(names, enc_outputs)
        }
        enriched["encoder_hidden_states"] = batch["encoder_hidden_states"].astype(np.float32, copy=False)
        return enriched

    def rewind(self) -> None:
        self.inner.rewind()


def _quantize_one_half(
    input_onnx: Path,
    output_onnx: Path,
    calibration_reader: CalibrationDataReader,
) -> dict[str, Any]:
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    ort_ready_onnx = output_onnx.with_name(output_onnx.stem + "_ort_ready.onnx")
    changed = prepare_model_for_ort_quantization(str(input_onnx), str(ort_ready_onnx))
    quant_input = ort_ready_onnx if changed else input_onnx
    exclude_nodes = _build_gentle_exclude_nodes(quant_input)

    quantize_static(
        model_input=str(quant_input),
        model_output=str(output_onnx),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=GENTLE_OP_TYPES_TO_QUANTIZE,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        nodes_to_exclude=exclude_nodes,
        use_external_data_format=True,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "CalibPercentile": 99.99,
            "CalibMovingAverage": True,
        },
    )

    return {
        "input_onnx": str(input_onnx),
        "quant_input": str(quant_input),
        "candidate_onnx": str(output_onnx),
        "ort_ready_copy_created": bool(changed),
        "excluded_nodes_count": len(exclude_nodes),
        "excluded_node_examples": exclude_nodes[:30],
        "exclude_substrings": GENTLE_EXCLUDE_SUBSTRINGS,
        "exclude_op_types": sorted(GENTLE_EXCLUDE_OP_TYPES),
        "op_types_to_quantize": GENTLE_OP_TYPES_TO_QUANTIZE,
    }


def _quantize_split_halves(
    split_root: Path,
    calibration_npz: Path,
    quantized_root: Path,
    max_samples: int,
    unet_dir: Path,
    device: str,
) -> dict[str, Any]:
    baseline_encoder = split_root / "unet_encoder.onnx" / "model.onnx"
    baseline_decoder = split_root / "unet_decoder.onnx" / "model.onnx"
    candidate_encoder = quantized_root / "unet_encoder_gentle_8w8a" / "model.onnx"
    candidate_decoder = quantized_root / "unet_decoder_gentle_8w8a" / "model.onnx"

    if not baseline_encoder.exists():
        raise FileNotFoundError(f"Split encoder ONNX not found: {baseline_encoder}")
    if not baseline_decoder.exists():
        raise FileNotFoundError(f"Split decoder ONNX not found: {baseline_decoder}")

    meta: dict[str, Any] = {
        "baseline_encoder": str(baseline_encoder),
        "baseline_decoder": str(baseline_decoder),
        "candidate_encoder": str(candidate_encoder),
        "candidate_decoder": str(candidate_decoder),
    }

    encoder_reader = _EncoderCalibrationReader(calibration_npz, max_samples=max_samples)
    try:
        start = time.perf_counter()
        try:
            meta["encoder"] = _quantize_one_half(baseline_encoder, candidate_encoder, encoder_reader)
        except Exception as exc:
            raise RuntimeError(f"encoder_half_failed: {exc}") from exc
        meta["encoder"]["elapsed_sec"] = time.perf_counter() - start
    finally:
        del encoder_reader
        gc.collect()

    decoder_reader = _DecoderCalibrationReader(
        calibration_npz,
        unet_dir=unet_dir,
        device=device,
        max_samples=max_samples,
    )
    try:
        start = time.perf_counter()
        try:
            meta["decoder"] = _quantize_one_half(baseline_decoder, candidate_decoder, decoder_reader)
        except Exception as exc:
            raise RuntimeError(f"decoder_half_failed: {exc}") from exc
        meta["decoder"]["elapsed_sec"] = time.perf_counter() - start
    finally:
        del decoder_reader
        _cleanup_cuda()

    return meta


def _load_pipeline(pipeline_dir: str, device: torch.device, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pipeline_dir,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    return pipe


def _encode_prompt_bundle(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        pe, ne, pp, npool = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            negative_prompt=negative,
            negative_prompt_2=negative,
            do_classifier_free_guidance=True,
            device=device,
            num_images_per_prompt=1,
        )
    add_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(device)
    return {
        "prompt_embeds": pe,
        "negative_prompt_embeds": ne,
        "pooled_prompt_embeds": pp,
        "negative_pooled_prompt_embeds": npool,
        "time_ids": add_time_ids,
    }


def _run_diffusers_pair(
    pipe: StableDiffusionXLPipeline,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    enc: dict[str, torch.Tensor],
    guidance_scale: float,
) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        uncond = pipe.unet(
            sample,
            timestep,
            encoder_hidden_states=enc["negative_prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": enc["negative_pooled_prompt_embeds"],
                "time_ids": enc["time_ids"],
            },
            return_dict=False,
        )[0]
        cond = pipe.unet(
            sample,
            timestep,
            encoder_hidden_states=enc["prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": enc["pooled_prompt_embeds"],
                "time_ids": enc["time_ids"],
            },
            return_dict=False,
        )[0]
        mixed = uncond + guidance_scale * (cond - uncond)
    return {
        "uncond": uncond.detach().cpu().numpy().astype(np.float32),
        "cond": cond.detach().cpu().numpy().astype(np.float32),
        "mixed": mixed.detach().cpu().numpy().astype(np.float32),
    }


def _build_canonical_trajectory(
    pipe: StableDiffusionXLPipeline,
    enc: dict[str, torch.Tensor],
    steps: int,
    step_indices: list[int],
    seed: int,
    width: int,
    height: int,
    guidance_scale: float,
    device: torch.device,
) -> list[StepSnapshot]:
    pipe.scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=torch.float32)
    latents = latents * pipe.scheduler.init_noise_sigma

    requested = set(step_indices)
    snapshots: list[StepSnapshot] = []

    with torch.inference_mode():
        for i, t in enumerate(pipe.scheduler.timesteps):
            sample = pipe.scheduler.scale_model_input(latents, t)
            timestep = t.reshape(1)
            outputs = _run_diffusers_pair(pipe, sample, timestep, enc, guidance_scale)
            next_latents = pipe.scheduler.step(
                torch.from_numpy(outputs["mixed"]).to(device=device, dtype=torch.float32),
                t,
                latents,
                return_dict=False,
            )[0]
            if i in requested:
                snapshots.append(
                    StepSnapshot(
                        step_index=i,
                        timestep=float(t.detach().cpu().item()),
                        sample=sample.detach().cpu().numpy().astype(np.float32),
                        latents_before=latents.detach().cpu().numpy().astype(np.float32),
                        latents_after=next_latents.detach().cpu().numpy().astype(np.float32),
                        outputs=outputs,
                    )
                )
            latents = next_latents
    return snapshots


def _run_split_branch(
    encoder_session: ort.InferenceSession,
    encoder_input_meta: dict[str, str],
    encoder_output_names: list[str],
    decoder_session: ort.InferenceSession,
    decoder_input_meta: dict[str, str],
    sample: np.ndarray,
    timestep_value: float,
    encoder_hidden_states: np.ndarray,
    text_embeds: np.ndarray,
    time_ids: np.ndarray,
) -> np.ndarray:
    enc_source_map = {
        "sample": sample.astype(np.float32, copy=False),
        "timestep": np.array([timestep_value], dtype=np.float32),
        "encoder_hidden_states": encoder_hidden_states.astype(np.float32, copy=False),
        "text_embeds": text_embeds.astype(np.float32, copy=False),
        "time_ids": time_ids.astype(np.float32, copy=False),
    }
    enc_inputs = {
        input_name: _cast_for_input(enc_source_map[input_name], ort_type)
        for input_name, ort_type in encoder_input_meta.items()
    }
    encoder_outputs = encoder_session.run(None, enc_inputs)
    encoder_output_map = {
        name: value.astype(np.float32, copy=False)
        for name, value in zip(encoder_output_names, encoder_outputs)
    }

    dec_source_map = dict(encoder_output_map)
    dec_source_map["encoder_hidden_states"] = encoder_hidden_states.astype(np.float32, copy=False)
    dec_inputs = {
        input_name: _cast_for_input(dec_source_map[input_name], ort_type)
        for input_name, ort_type in decoder_input_meta.items()
    }
    output = decoder_session.run(None, dec_inputs)[0].astype(np.float32, copy=False)
    if output.shape[-1] == 4 and output.ndim == 4 and output.shape[1] != 4:
        output = np.transpose(output, (0, 3, 1, 2)).astype(np.float32, copy=False)
    return output


def _run_split_onnx_pair(
    encoder_session: ort.InferenceSession,
    encoder_input_meta: dict[str, str],
    encoder_output_names: list[str],
    decoder_session: ort.InferenceSession,
    decoder_input_meta: dict[str, str],
    sample: np.ndarray,
    timestep_value: float,
    enc: dict[str, torch.Tensor],
    guidance_scale: float,
) -> dict[str, np.ndarray]:
    branch_results: dict[str, np.ndarray] = {}
    for branch_name, ehs_key, pooled_key in (
        ("uncond", "negative_prompt_embeds", "negative_pooled_prompt_embeds"),
        ("cond", "prompt_embeds", "pooled_prompt_embeds"),
    ):
        branch_results[branch_name] = _run_split_branch(
            encoder_session,
            encoder_input_meta,
            encoder_output_names,
            decoder_session,
            decoder_input_meta,
            sample,
            timestep_value,
            enc[ehs_key].detach().cpu().numpy(),
            enc[pooled_key].detach().cpu().numpy(),
            enc["time_ids"].detach().cpu().numpy(),
        )

    branch_results["mixed"] = branch_results["uncond"] + guidance_scale * (
        branch_results["cond"] - branch_results["uncond"]
    )
    return branch_results


def _verdict_from_diff(diff: dict[str, Any], ref_stats: dict[str, Any]) -> dict[str, Any]:
    ref_std = max(float(ref_stats.get("std", 0.0)), 1e-6)
    rmse = float(diff.get("rmse", math.inf))
    mae = float(diff.get("mae", math.inf))
    cosine = float(diff.get("cosine", -1.0))
    rel_rmse = rmse / ref_std
    rel_mae = mae / ref_std

    if cosine >= 0.999 and rel_rmse <= 0.03:
        verdict = "excellent"
    elif cosine >= 0.995 and rel_rmse <= 0.08:
        verdict = "acceptable"
    elif cosine >= 0.985 and rel_rmse <= 0.15:
        verdict = "borderline"
    else:
        verdict = "bad"

    return {
        "verdict": verdict,
        "cosine": cosine,
        "rel_rmse": rel_rmse,
        "rel_mae": rel_mae,
        "rmse": rmse,
        "mae": mae,
        "max_abs": float(diff.get("max_abs", math.inf)),
    }


def _aggregate_tensor_gate(step_reports: list[dict[str, Any]]) -> dict[str, Any]:
    order = {"excellent": 0, "acceptable": 1, "borderline": 2, "bad": 3}
    worst = "excellent"
    worst_item: dict[str, Any] | None = None

    for step in step_reports:
        for branch_name, verdict in step["candidate_vs_diffusers_verdict"].items():
            label = verdict["verdict"]
            if order[label] > order[worst]:
                worst = label
                worst_item = {
                    "step_index": step["step_index"],
                    "branch": branch_name,
                    **verdict,
                }

    proceed = worst in {"excellent", "acceptable", "borderline"}
    return {
        "overall_verdict": worst,
        "worst_case": worst_item,
        "proceed_to_full_cycle": proceed,
    }


def _run_tensor_stage(
    pipe: StableDiffusionXLPipeline,
    enc: dict[str, torch.Tensor],
    args: argparse.Namespace,
    baseline_encoder_onnx: Path,
    baseline_decoder_onnx: Path,
    candidate_encoder_onnx: Path,
    candidate_decoder_onnx: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    step_indices = _parse_step_indices(args.tensor_step_indices, args.steps)
    snapshots = _build_canonical_trajectory(
        pipe,
        enc,
        args.steps,
        step_indices,
        args.seed,
        args.width,
        args.height,
        args.guidance_scale,
        device,
    )

    baseline_encoder_session, baseline_encoder_used, baseline_encoder_providers = _load_onnx_session(baseline_encoder_onnx)
    baseline_decoder_session, baseline_decoder_used, baseline_decoder_providers = _load_onnx_session(baseline_decoder_onnx)
    candidate_encoder_session, candidate_encoder_used, candidate_encoder_providers = _load_onnx_session(candidate_encoder_onnx)
    candidate_decoder_session, candidate_decoder_used, candidate_decoder_providers = _load_onnx_session(candidate_decoder_onnx)

    baseline_encoder_input_meta = {inp.name: inp.type for inp in baseline_encoder_session.get_inputs()}
    baseline_decoder_input_meta = {inp.name: inp.type for inp in baseline_decoder_session.get_inputs()}
    candidate_encoder_input_meta = {inp.name: inp.type for inp in candidate_encoder_session.get_inputs()}
    candidate_decoder_input_meta = {inp.name: inp.type for inp in candidate_decoder_session.get_inputs()}

    baseline_encoder_output_names = [out.name for out in baseline_encoder_session.get_outputs()]
    candidate_encoder_output_names = [out.name for out in candidate_encoder_session.get_outputs()]

    step_reports: list[dict[str, Any]] = []
    for snap in snapshots:
        baseline_out = _run_split_onnx_pair(
            baseline_encoder_session,
            baseline_encoder_input_meta,
            baseline_encoder_output_names,
            baseline_decoder_session,
            baseline_decoder_input_meta,
            snap.sample,
            snap.timestep,
            enc,
            args.guidance_scale,
        )
        candidate_out = _run_split_onnx_pair(
            candidate_encoder_session,
            candidate_encoder_input_meta,
            candidate_encoder_output_names,
            candidate_decoder_session,
            candidate_decoder_input_meta,
            snap.sample,
            snap.timestep,
            enc,
            args.guidance_scale,
        )

        baseline_vs_diffusers = {
            branch: _diff_metrics(baseline_out[branch], snap.outputs[branch])
            for branch in ("uncond", "cond", "mixed")
        }
        candidate_vs_diffusers = {
            branch: _diff_metrics(candidate_out[branch], snap.outputs[branch])
            for branch in ("uncond", "cond", "mixed")
        }
        candidate_vs_baseline = {
            branch: _diff_metrics(candidate_out[branch], baseline_out[branch])
            for branch in ("uncond", "cond", "mixed")
        }

        baseline_vs_diffusers_verdict = {
            branch: _verdict_from_diff(baseline_vs_diffusers[branch], _tensor_stats(snap.outputs[branch]))
            for branch in ("uncond", "cond", "mixed")
        }
        candidate_vs_diffusers_verdict = {
            branch: _verdict_from_diff(candidate_vs_diffusers[branch], _tensor_stats(snap.outputs[branch]))
            for branch in ("uncond", "cond", "mixed")
        }

        step_reports.append(
            {
                "step_index": snap.step_index,
                "timestep": snap.timestep,
                "sample_stats": _tensor_stats(snap.sample),
                "diffusers_outputs": {branch: _tensor_stats(arr) for branch, arr in snap.outputs.items()},
                "baseline_split_outputs": {branch: _tensor_stats(arr) for branch, arr in baseline_out.items()},
                "candidate_outputs": {branch: _tensor_stats(arr) for branch, arr in candidate_out.items()},
                "baseline_split_vs_diffusers": baseline_vs_diffusers,
                "baseline_split_vs_diffusers_verdict": baseline_vs_diffusers_verdict,
                "candidate_vs_diffusers": candidate_vs_diffusers,
                "candidate_vs_baseline_split": candidate_vs_baseline,
                "candidate_vs_diffusers_verdict": candidate_vs_diffusers_verdict,
            }
        )

    gate = _aggregate_tensor_gate(step_reports)
    return {
        "baseline_encoder_onnx_used": baseline_encoder_used,
        "baseline_decoder_onnx_used": baseline_decoder_used,
        "baseline_encoder_providers": baseline_encoder_providers,
        "baseline_decoder_providers": baseline_decoder_providers,
        "candidate_encoder_onnx_used": candidate_encoder_used,
        "candidate_decoder_onnx_used": candidate_decoder_used,
        "candidate_encoder_providers": candidate_encoder_providers,
        "candidate_decoder_providers": candidate_decoder_providers,
        "step_indices": step_indices,
        "steps": step_reports,
        "gate": gate,
    }


def _decode_latents_to_image(pipe: StableDiffusionXLPipeline, latents: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = (decoded / 2 + 0.5).clamp(0, 1)
    return image[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _run_diffusers_full_cycle(
    pipe: StableDiffusionXLPipeline,
    enc: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    pipe.scheduler.set_timesteps(args.steps, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 4, args.height // 8, args.width // 8), generator=generator, device=device, dtype=torch.float32)
    latents = latents * pipe.scheduler.init_noise_sigma

    step_times: list[float] = []
    with torch.inference_mode():
        for t in pipe.scheduler.timesteps:
            step_start = time.perf_counter()
            sample = pipe.scheduler.scale_model_input(latents, t)
            outputs = _run_diffusers_pair(pipe, sample, t.reshape(1), enc, args.guidance_scale)
            latents = pipe.scheduler.step(
                torch.from_numpy(outputs["mixed"]).to(device=device, dtype=torch.float32),
                t,
                latents,
                return_dict=False,
            )[0]
            _sync(device)
            step_times.append((time.perf_counter() - step_start) * 1000.0)

    image = _decode_latents_to_image(pipe, latents)
    return {
        "latents": latents.detach().cpu().numpy().astype(np.float32),
        "image": image,
        "step_times_ms": [round(x, 3) for x in step_times],
        "total_unet_ms": float(sum(step_times)),
    }


def _run_split_full_cycle(
    pipe: StableDiffusionXLPipeline,
    enc: dict[str, torch.Tensor],
    args: argparse.Namespace,
    encoder_onnx: Path,
    decoder_onnx: Path,
    device: torch.device,
) -> dict[str, Any]:
    pipe.scheduler.set_timesteps(args.steps, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 4, args.height // 8, args.width // 8), generator=generator, device=device, dtype=torch.float32)
    latents = latents * pipe.scheduler.init_noise_sigma

    encoder_session, encoder_used, encoder_providers = _load_onnx_session(encoder_onnx)
    decoder_session, decoder_used, decoder_providers = _load_onnx_session(decoder_onnx)
    encoder_input_meta = {inp.name: inp.type for inp in encoder_session.get_inputs()}
    decoder_input_meta = {inp.name: inp.type for inp in decoder_session.get_inputs()}
    encoder_output_names = [out.name for out in encoder_session.get_outputs()]

    step_times: list[float] = []
    for t in pipe.scheduler.timesteps:
        step_start = time.perf_counter()
        sample = pipe.scheduler.scale_model_input(latents, t)
        outputs = _run_split_onnx_pair(
            encoder_session,
            encoder_input_meta,
            encoder_output_names,
            decoder_session,
            decoder_input_meta,
            sample.detach().cpu().numpy().astype(np.float32),
            float(t.detach().cpu().item()),
            enc,
            args.guidance_scale,
        )
        latents = pipe.scheduler.step(
            torch.from_numpy(outputs["mixed"]).to(device=device, dtype=torch.float32),
            t,
            latents,
            return_dict=False,
        )[0]
        _sync(device)
        step_times.append((time.perf_counter() - step_start) * 1000.0)

    image = _decode_latents_to_image(pipe, latents)
    return {
        "encoder_onnx_used": encoder_used,
        "decoder_onnx_used": decoder_used,
        "encoder_providers": encoder_providers,
        "decoder_providers": decoder_providers,
        "latents": latents.detach().cpu().numpy().astype(np.float32),
        "image": image,
        "step_times_ms": [round(x, 3) for x in step_times],
        "total_unet_ms": float(sum(step_times)),
    }


def _image_diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = _diff_metrics(a.astype(np.float32), b.astype(np.float32))
    gray_a = a.mean(axis=2)
    gray_b = b.mean(axis=2)
    diff.update(
        {
            "gray_corr": _corr(gray_a, gray_b),
            "rgb_corr": _corr(a, b),
            "mean_rgb_a": [float(x) for x in a.mean(axis=(0, 1))],
            "mean_rgb_b": [float(x) for x in b.mean(axis=(0, 1))],
            "std_rgb_a": [float(x) for x in a.std(axis=(0, 1))],
            "std_rgb_b": [float(x) for x in b.std(axis=(0, 1))],
            "mean_rgb_abs_delta": [float(x) for x in np.abs(a.mean(axis=(0, 1)) - b.mean(axis=(0, 1)))],
            "std_rgb_abs_delta": [float(x) for x in np.abs(a.std(axis=(0, 1)) - b.std(axis=(0, 1)))],
        }
    )
    return diff


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).round().astype(np.uint8)).save(path)


def _summarize_full_cycle(
    out_dir: Path,
    baseline: dict[str, Any],
    baseline_split: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_path = out_dir / "baseline_diffusers.png"
    baseline_split_path = out_dir / "baseline_split_float.png"
    candidate_path = out_dir / "candidate_split_gentle_8w8a.png"

    _save_image(baseline_path, baseline["image"])
    _save_image(baseline_split_path, baseline_split["image"])
    _save_image(candidate_path, candidate["image"])

    baseline_assess = assess_image(baseline_path)
    baseline_split_assess = assess_image(baseline_split_path)
    candidate_assess = assess_image(candidate_path)

    return {
        "baseline_image": str(baseline_path),
        "baseline_split_image": str(baseline_split_path),
        "candidate_image": str(candidate_path),
        "baseline_vs_split_float": _image_diff_metrics(baseline["image"], baseline_split["image"]),
        "baseline_vs_candidate": _image_diff_metrics(baseline["image"], candidate["image"]),
        "baseline_assessment": baseline_assess,
        "baseline_split_assessment": baseline_split_assess,
        "candidate_assessment": candidate_assess,
        "baseline_unet_total_ms": baseline["total_unet_ms"],
        "baseline_split_total_ms": baseline_split["total_unet_ms"],
        "candidate_total_ms": candidate["total_unet_ms"],
        "baseline_step_times_ms": baseline["step_times_ms"],
        "baseline_split_step_times_ms": baseline_split["step_times_ms"],
        "candidate_step_times_ms": candidate["step_times_ms"],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_root = Path(args.split_root)
    quantized_root = Path(args.quantized_root)
    calibration_npz = Path(args.calibration_npz)
    baseline_encoder_onnx = split_root / "unet_encoder.onnx" / "model.onnx"
    baseline_decoder_onnx = split_root / "unet_decoder.onnx" / "model.onnx"
    candidate_encoder_onnx = quantized_root / "unet_encoder_gentle_8w8a" / "model.onnx"
    candidate_decoder_onnx = quantized_root / "unet_decoder_gentle_8w8a" / "model.onnx"

    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)

    if not baseline_encoder_onnx.exists():
        raise FileNotFoundError(f"Baseline split encoder ONNX not found: {baseline_encoder_onnx}")
    if not baseline_decoder_onnx.exists():
        raise FileNotFoundError(f"Baseline split decoder ONNX not found: {baseline_decoder_onnx}")
    if not calibration_npz.exists():
        raise FileNotFoundError(f"Calibration NPZ not found: {calibration_npz}")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "paths": {
            "baseline_encoder_onnx": str(baseline_encoder_onnx),
            "baseline_decoder_onnx": str(baseline_decoder_onnx),
            "candidate_encoder_onnx": str(candidate_encoder_onnx),
            "candidate_decoder_onnx": str(candidate_decoder_onnx),
        },
        "environment": {
            "device": str(device),
            "torch_cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ort_available_providers": ort.get_available_providers(),
            "ort_selected_providers": _choose_ort_providers(),
        },
        "notes": [
            "This script does not touch phone-side production runtime or deployment artifacts.",
            "Split evaluator avoids monolithic extmaps external-bias inputs and measures parity half-by-half.",
            "Stage 1 uses tensor parity on canonical latent inputs before Stage 2 attempts a full image cycle.",
        ],
    }

    quant_start = time.perf_counter()
    try:
        if args.force_requantize or not (candidate_encoder_onnx.exists() and candidate_decoder_onnx.exists()):
            quant_meta = _quantize_split_halves(
                split_root,
                calibration_npz,
                quantized_root,
                max_samples=args.calib_samples,
                unet_dir=SDXL_NPU_ROOT / "unet_lightning8step_merged",
                device=args.device,
            )
            quant_meta["rebuilt"] = True
        else:
            quant_meta = {
                "rebuilt": False,
                "message": "Existing gentle split 8W8A candidates reused",
                "candidate_encoder": str(candidate_encoder_onnx),
                "candidate_decoder": str(candidate_decoder_onnx),
            }
        quant_meta["elapsed_sec"] = time.perf_counter() - quant_start
        report["quantization"] = quant_meta
    except Exception as exc:
        report["quantization"] = {
            "candidate_encoder": str(candidate_encoder_onnx),
            "candidate_decoder": str(candidate_decoder_onnx),
            "elapsed_sec": time.perf_counter() - quant_start,
            "error": f"{type(exc).__name__}: {exc}",
            "status": "blocked",
            "next_step": (
                "Split gentle 8W8A still hit a tooling/resource blocker on this host. "
                "Inspect which half failed and consider reducing calibration samples or narrowing exclusions further."
            ),
        }
        report["tensor_stage"] = {
            "skipped": True,
            "reason": "quantization_blocked",
        }
        report["full_cycle"] = {
            "skipped": True,
            "reason": "quantization_blocked",
        }
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "report": str(report_path),
            "quantization_status": "blocked",
            "error": report["quantization"]["error"],
        }, ensure_ascii=False, indent=2))
        return

    pipe_load_start = time.perf_counter()
    pipe = _load_pipeline(args.pipeline_dir, device, dtype)
    _sync(device)
    report["pipeline_load_sec"] = time.perf_counter() - pipe_load_start

    encode_start = time.perf_counter()
    enc = _encode_prompt_bundle(pipe, args.prompt, args.negative, args.width, args.height, device)
    _sync(device)
    report["prompt_encode_sec"] = time.perf_counter() - encode_start

    tensor_start = time.perf_counter()
    tensor_stage = _run_tensor_stage(
        pipe,
        enc,
        args,
        baseline_encoder_onnx,
        baseline_decoder_onnx,
        candidate_encoder_onnx,
        candidate_decoder_onnx,
    )
    tensor_stage["elapsed_sec"] = time.perf_counter() - tensor_start
    report["tensor_stage"] = tensor_stage

    gate = tensor_stage["gate"]
    run_full_cycle = (not args.tensor_only) and (gate["proceed_to_full_cycle"] or args.force_full_cycle)

    if run_full_cycle:
        full_cycle_start = time.perf_counter()
        baseline = _run_diffusers_full_cycle(pipe, enc, args, device)
        baseline_split_cycle = _run_split_full_cycle(
            pipe,
            enc,
            args,
            baseline_encoder_onnx,
            baseline_decoder_onnx,
            device,
        )
        candidate_cycle = _run_split_full_cycle(
            pipe,
            enc,
            args,
            candidate_encoder_onnx,
            candidate_decoder_onnx,
            device,
        )
        report["full_cycle"] = _summarize_full_cycle(out_dir, baseline, baseline_split_cycle, candidate_cycle)
        report["full_cycle"]["elapsed_sec"] = time.perf_counter() - full_cycle_start
    else:
        reason = "tensor_only requested" if args.tensor_only else f"tensor gate verdict={gate['overall_verdict']}"
        report["full_cycle"] = {
            "skipped": True,
            "reason": reason,
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "report": str(report_path),
        "tensor_gate": gate,
        "full_cycle_skipped": bool(report["full_cycle"].get("skipped", False)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()