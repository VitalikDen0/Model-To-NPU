#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
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
from quantize_unet import prepare_model_for_ort_quantization  # noqa: E402


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Host-side GPU baseline compare: checkpoint original vs diffusers original vs extmaps ONNX")
	ap.add_argument("--checkpoint", required=True)
	ap.add_argument("--pipeline-dir", default="d:/platform-tools/sdxl_npu/diffusers_pipeline")
	ap.add_argument("--onnx", default="d:/platform-tools/sdxl_npu/onnx_export_extmaps_groupnorm18_pruned_nocast_fp16bias/unet.onnx")
	ap.add_argument("--prompt", required=True)
	ap.add_argument("--negative", default="")
	ap.add_argument("--steps", type=int, default=4)
	ap.add_argument("--step-indices", default="0")
	ap.add_argument("--seed", type=int, default=123)
	ap.add_argument("--guidance-scale", type=float, default=5.0)
	ap.add_argument("--width", type=int, default=1024)
	ap.add_argument("--height", type=int, default=1024)
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
	ap.add_argument("--out", default="d:/platform-tools/NPU/outputs/host_compare_baselines.json")
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


def _parse_step_indices(spec: str, total_steps: int) -> list[int]:
	result: list[int] = []
	for part in spec.split(","):
		p = part.strip()
		if not p:
			continue
		value = int(p)
		if value < 0 or value >= total_steps:
			raise ValueError(f"step index {value} outside valid range [0, {total_steps - 1}]")
		result.append(value)
	if not result:
		raise ValueError("No step indices requested")
	return sorted(dict.fromkeys(result))


def _cast_for_input(value: np.ndarray, ort_type: str) -> np.ndarray:
	if ort_type == "tensor(float16)":
		return value.astype(np.float16, copy=False)
	if ort_type == "tensor(float)":
		return value.astype(np.float32, copy=False)
	if ort_type == "tensor(double)":
		return value.astype(np.float64, copy=False)
	raise RuntimeError(f"Unsupported ONNX input type: {ort_type}")


def _choose_ort_providers() -> list[str]:
	available = ort.get_available_providers()
	if "CUDAExecutionProvider" in available:
		return ["CUDAExecutionProvider", "CPUExecutionProvider"]
	return ["CPUExecutionProvider"]


def _load_onnx_session(onnx_path: Path) -> tuple[ort.InferenceSession, str, list[str]]:
	providers = _choose_ort_providers()
	normalized_onnx = str(onnx_path)
	normalized_candidate = str(onnx_path.with_name(onnx_path.stem + "_ort_normalized.onnx"))
	try:
		session = ort.InferenceSession(normalized_onnx, providers=providers)
	except Exception:
		changed = prepare_model_for_ort_quantization(str(onnx_path), normalized_candidate)
		if not changed:
			normalized_candidate = str(onnx_path)
		normalized_onnx = normalized_candidate
		session = ort.InferenceSession(normalized_onnx, providers=providers)
	return session, normalized_onnx, providers


def _save_array(base_path: Path, suffix: str, arr: np.ndarray) -> str:
	out = base_path.with_name(base_path.stem + suffix)
	np.save(out, arr)
	return str(out)


def _load_pipeline_from_dir(pipeline_dir: str, device: torch.device, dtype: torch.dtype) -> StableDiffusionXLPipeline:
	pipe = StableDiffusionXLPipeline.from_pretrained(
		pipeline_dir,
		torch_dtype=dtype,
		local_files_only=True,
	)
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	return pipe


def _load_pipeline_from_checkpoint(checkpoint: str, device: torch.device, dtype: torch.dtype) -> StableDiffusionXLPipeline:
	pipe = StableDiffusionXLPipeline.from_single_file(
		checkpoint,
		torch_dtype=dtype,
		use_safetensors=True,
	)
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
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


def _run_unet_pair(
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
			added_cond_kwargs={"text_embeds": enc["negative_pooled_prompt_embeds"], "time_ids": enc["time_ids"]},
			return_dict=False,
		)[0]
		cond = pipe.unet(
			sample,
			timestep,
			encoder_hidden_states=enc["prompt_embeds"],
			added_cond_kwargs={"text_embeds": enc["pooled_prompt_embeds"], "time_ids": enc["time_ids"]},
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
) -> list[dict[str, Any]]:
	pipe.scheduler.set_timesteps(steps, device=device)
	generator = torch.Generator(device=device).manual_seed(seed)
	latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=torch.float32)
	latents = latents * pipe.scheduler.init_noise_sigma

	requested = set(step_indices)
	collected: list[dict[str, Any]] = []

	with torch.inference_mode():
		for i, t in enumerate(pipe.scheduler.timesteps):
			sample = pipe.scheduler.scale_model_input(latents, t)
			timestep = t.reshape(1)
			outputs = _run_unet_pair(pipe, sample, timestep, enc, guidance_scale)
			next_latents = pipe.scheduler.step(
				torch.from_numpy(outputs["mixed"]).to(device=device, dtype=torch.float32),
				t,
				latents,
				return_dict=False,
			)[0]

			if i in requested:
				collected.append(
					{
						"step_index": i,
						"timestep": float(t.detach().cpu().item()),
						"sample": sample.detach().cpu().numpy().astype(np.float32),
						"latents_before": latents.detach().cpu().numpy().astype(np.float32),
						"latents_after": next_latents.detach().cpu().numpy().astype(np.float32),
						"outputs": outputs,
					}
				)
			latents = next_latents

	return collected


def _run_extmaps_onnx(
	onnx_path: Path,
	pipe: StableDiffusionXLPipeline,
	enc: dict[str, torch.Tensor],
	canonical_steps: list[dict[str, Any]],
	width: int,
	height: int,
	guidance_scale: float,
) -> dict[str, Any]:
	session, normalized_onnx, providers = _load_onnx_session(onnx_path)
	input_meta = {inp.name: inp.type for inp in session.get_inputs()}

	unet = pipe.unet
	install_external_resnet_bias_surgery(unet)
	spatial_shapes = infer_unet_resnet_spatial_shapes(unet, latent_h=height // 8, latent_w=width // 8)
	resnet_sites = collect_unet_resnet_conditioning_modules(unet)

	step_outputs: list[dict[str, Any]] = []
	for step in canonical_steps:
		sample_t = torch.from_numpy(step["sample"]).to(pipe.device, dtype=torch.float32)
		timestep_t = torch.tensor([step["timestep"]], device=pipe.device, dtype=torch.float32)
		branch_results: dict[str, np.ndarray] = {}
		bias_names: list[str] = []

		for branch_name, ehs_key, pooled_key in (
			("uncond", "negative_prompt_embeds", "negative_pooled_prompt_embeds"),
			("cond", "prompt_embeds", "pooled_prompt_embeds"),
		):
			encoder_hidden_states = enc[ehs_key]
			text_embeds = enc[pooled_key]
			biases = compute_external_resnet_biases(
				unet,
				sample_t,
				timestep_t,
				encoder_hidden_states,
				text_embeds,
				enc["time_ids"],
			)
			expanded_biases = {
				module._copilot_external_bias_name: bias.expand(-1, -1, *spatial_shapes[name]).detach().cpu().numpy().astype(np.float32)
				for (name, module), bias in zip(resnet_sites, biases)
			}
			bias_names = list(expanded_biases.keys())
			source_map: dict[str, np.ndarray] = {
				"sample": step["sample"].astype(np.float32, copy=False),
				"timestep": np.array([step["timestep"]], dtype=np.float32),
				"encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
				"text_embeds": text_embeds.detach().cpu().numpy().astype(np.float32),
				"time_ids": enc["time_ids"].detach().cpu().numpy().astype(np.float32),
			}
			time_ids_np = source_map["time_ids"]
			for idx in range(6):
				source_map[f"time_id_{idx}"] = time_ids_np[:, idx : idx + 1]
			source_map.update(expanded_biases)

			ort_inputs: dict[str, np.ndarray] = {}
			missing_inputs: list[str] = []
			for input_name, ort_type in input_meta.items():
				if input_name not in source_map:
					missing_inputs.append(input_name)
					continue
				ort_inputs[input_name] = _cast_for_input(source_map[input_name], ort_type)
			if missing_inputs:
				raise RuntimeError(f"Could not map ONNX inputs for step {step['step_index']}: {missing_inputs}")

			ort_out = session.run(None, ort_inputs)[0].astype(np.float32, copy=False)
			if ort_out.shape == (1, 128, 128, 4):
				ort_out = np.transpose(ort_out, (0, 3, 1, 2)).astype(np.float32, copy=False)
			branch_results[branch_name] = ort_out

		branch_results["mixed"] = branch_results["uncond"] + guidance_scale * (branch_results["cond"] - branch_results["uncond"])
		step_outputs.append(
			{
				"step_index": step["step_index"],
				"timestep": step["timestep"],
				"outputs": branch_results,
				"external_bias_names": bias_names,
			}
		)

	return {
		"onnx_used": normalized_onnx,
		"providers": providers,
		"input_types": input_meta,
		"steps": step_outputs,
	}


def main() -> None:
	args = parse_args()
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	device = torch.device(args.device)
	torch_dtype = _torch_dtype(args.dtype)
	step_indices = _parse_step_indices(args.step_indices, args.steps)
	timings: dict[str, float] = {}
	report: dict[str, Any] = {
		"checkpoint": args.checkpoint,
		"pipeline_dir": args.pipeline_dir,
		"onnx": args.onnx,
		"prompt": args.prompt,
		"negative": args.negative,
		"steps": args.steps,
		"step_indices": step_indices,
		"seed": args.seed,
		"guidance_scale": args.guidance_scale,
		"device": str(device),
		"dtype": args.dtype,
		"timings_sec": timings,
	}

	t0 = time.perf_counter()
	diffusers_pipe = _load_pipeline_from_dir(args.pipeline_dir, device, torch_dtype)
	_sync(device)
	timings["load_diffusers_pipeline"] = time.perf_counter() - t0

	t0 = time.perf_counter()
	diffusers_enc = _encode_prompt_bundle(diffusers_pipe, args.prompt, args.negative, args.width, args.height, device)
	_sync(device)
	timings["encode_diffusers_prompt"] = time.perf_counter() - t0

	t0 = time.perf_counter()
	canonical_steps = _build_canonical_trajectory(
		diffusers_pipe,
		diffusers_enc,
		args.steps,
		step_indices,
		args.seed,
		args.width,
		args.height,
		args.guidance_scale,
		device,
	)
	_sync(device)
	timings["build_canonical_trajectory"] = time.perf_counter() - t0

	t0 = time.perf_counter()
	onnx_bundle = _run_extmaps_onnx(
		Path(args.onnx),
		diffusers_pipe,
		diffusers_enc,
		canonical_steps,
		args.width,
		args.height,
		args.guidance_scale,
	)
	timings["run_extmaps_onnx"] = time.perf_counter() - t0

	diffusers_steps_report: list[dict[str, Any]] = []
	for step in canonical_steps:
		step_id = step["step_index"]
		arrays = {
			branch: _save_array(out_path, f".step{step_id:03d}.diffusers_{branch}", arr)
			for branch, arr in step["outputs"].items()
		}
		diffusers_steps_report.append(
			{
				"step_index": step_id,
				"timestep": step["timestep"],
				"sample_stats": _tensor_stats(step["sample"]),
				"latents_before_stats": _tensor_stats(step["latents_before"]),
				"latents_after_stats": _tensor_stats(step["latents_after"]),
				"outputs": {branch: _tensor_stats(arr) for branch, arr in step["outputs"].items()},
				"array_paths": arrays,
			}
		)

	report["diffusers_baseline"] = {
		"prompt_embeds": _tensor_stats(diffusers_enc["prompt_embeds"].detach().cpu().numpy().astype(np.float32)),
		"negative_prompt_embeds": _tensor_stats(diffusers_enc["negative_prompt_embeds"].detach().cpu().numpy().astype(np.float32)),
		"pooled_prompt_embeds": _tensor_stats(diffusers_enc["pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32)),
		"negative_pooled_prompt_embeds": _tensor_stats(diffusers_enc["negative_pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32)),
		"time_ids": _tensor_stats(diffusers_enc["time_ids"].detach().cpu().numpy().astype(np.float32)),
		"steps": diffusers_steps_report,
	}
	report["extmaps_onnx"] = {
		"onnx_used": onnx_bundle["onnx_used"],
		"providers": onnx_bundle["providers"],
		"input_types": onnx_bundle["input_types"],
		"steps": [],
	}

	onnx_by_step = {step["step_index"]: step for step in onnx_bundle["steps"]}
	for step in canonical_steps:
		step_id = step["step_index"]
		onnx_step = onnx_by_step[step_id]
		arrays = {
			branch: _save_array(out_path, f".step{step_id:03d}.onnx_{branch}", arr)
			for branch, arr in onnx_step["outputs"].items()
		}
		report["extmaps_onnx"]["steps"].append(
			{
				"step_index": step_id,
				"timestep": step["timestep"],
				"outputs": {branch: _tensor_stats(arr) for branch, arr in onnx_step["outputs"].items()},
				"array_paths": arrays,
				"diff_vs_diffusers": {
					branch: _diff_metrics(onnx_step["outputs"][branch], step["outputs"][branch])
					for branch in ("uncond", "cond", "mixed")
				},
				"external_bias_names": onnx_step["external_bias_names"],
			}
		)

	del diffusers_pipe
	_cleanup_cuda()

	checkpoint_section: dict[str, Any] = {}
	try:
		t0 = time.perf_counter()
		checkpoint_pipe = _load_pipeline_from_checkpoint(args.checkpoint, device, torch_dtype)
		_sync(device)
		timings["load_checkpoint_pipeline"] = time.perf_counter() - t0

		t0 = time.perf_counter()
		checkpoint_enc = _encode_prompt_bundle(checkpoint_pipe, args.prompt, args.negative, args.width, args.height, device)
		_sync(device)
		timings["encode_checkpoint_prompt"] = time.perf_counter() - t0

		checkpoint_section["prompt_embeds"] = _tensor_stats(checkpoint_enc["prompt_embeds"].detach().cpu().numpy().astype(np.float32))
		checkpoint_section["negative_prompt_embeds"] = _tensor_stats(checkpoint_enc["negative_prompt_embeds"].detach().cpu().numpy().astype(np.float32))
		checkpoint_section["pooled_prompt_embeds"] = _tensor_stats(checkpoint_enc["pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32))
		checkpoint_section["negative_pooled_prompt_embeds"] = _tensor_stats(checkpoint_enc["negative_pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32))
		checkpoint_section["time_ids"] = _tensor_stats(checkpoint_enc["time_ids"].detach().cpu().numpy().astype(np.float32))
		checkpoint_section["embedding_diff_vs_diffusers"] = {
			"prompt_embeds": _diff_metrics(
				checkpoint_enc["prompt_embeds"].detach().cpu().numpy().astype(np.float32),
				diffusers_enc["prompt_embeds"].detach().cpu().numpy().astype(np.float32),
			),
			"negative_prompt_embeds": _diff_metrics(
				checkpoint_enc["negative_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
				diffusers_enc["negative_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
			),
			"pooled_prompt_embeds": _diff_metrics(
				checkpoint_enc["pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
				diffusers_enc["pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
			),
			"negative_pooled_prompt_embeds": _diff_metrics(
				checkpoint_enc["negative_pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
				diffusers_enc["negative_pooled_prompt_embeds"].detach().cpu().numpy().astype(np.float32),
			),
			"time_ids": _diff_metrics(
				checkpoint_enc["time_ids"].detach().cpu().numpy().astype(np.float32),
				diffusers_enc["time_ids"].detach().cpu().numpy().astype(np.float32),
			),
		}
		checkpoint_section["steps"] = []

		for step in canonical_steps:
			sample_t = torch.from_numpy(step["sample"]).to(device=device, dtype=torch.float32)
			timestep_t = torch.tensor([step["timestep"]], device=device, dtype=torch.float32)
			outputs = _run_unet_pair(checkpoint_pipe, sample_t, timestep_t, checkpoint_enc, args.guidance_scale)
			arrays = {
				branch: _save_array(out_path, f".step{step['step_index']:03d}.checkpoint_{branch}", arr)
				for branch, arr in outputs.items()
			}
			onnx_step = onnx_by_step[step["step_index"]]
			checkpoint_section["steps"].append(
				{
					"step_index": step["step_index"],
					"timestep": step["timestep"],
					"outputs": {branch: _tensor_stats(arr) for branch, arr in outputs.items()},
					"array_paths": arrays,
					"diff_vs_diffusers": {
						branch: _diff_metrics(outputs[branch], step["outputs"][branch])
						for branch in ("uncond", "cond", "mixed")
					},
					"diff_vs_extmaps_onnx": {
						branch: _diff_metrics(outputs[branch], onnx_step["outputs"][branch])
						for branch in ("uncond", "cond", "mixed")
					},
				}
			)

		del checkpoint_pipe
		_cleanup_cuda()
	except Exception as exc:
		checkpoint_section["error"] = f"{type(exc).__name__}: {exc}"

	report["checkpoint_baseline"] = checkpoint_section
	report["summary"] = {
		"checkpoint_loaded": "error" not in checkpoint_section,
		"onnx_provider_used": report["extmaps_onnx"]["providers"],
		"notes": [
			"Canonical latent trajectory is generated from current diffusers pipeline original UNet.",
			"Checkpoint and extmaps ONNX are evaluated on the same latent/timestep inputs for each requested step.",
			"This makes later-step comparisons meaningful; unlike naive step-index probes, latents are actually advanced between steps.",
		],
	}

	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[ok] saved report: {out_path}")
	print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
