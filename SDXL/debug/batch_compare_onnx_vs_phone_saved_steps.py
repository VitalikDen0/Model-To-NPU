#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]


def _load_native_phone_output(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    meta_path = path.with_name(path.name + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = tuple(int(x) for x in meta.get("Dimensions", []))
    dtype_name = str(meta.get("Datatype", ""))
    quant = meta.get("QuantaziationParams", {}) or {}

    if dims != (1, 128, 128, 4):
        raise RuntimeError(f"Unexpected native output dims={dims} for {path}")

    if dtype_name == "QNN_DATATYPE_UFIXED_POINT_8":
        raw = np.fromfile(path, dtype=np.uint8)
        scale = float(quant.get("Scale", 0.0))
        offset = float(quant.get("Offset", 0.0))
        arr = (raw.astype(np.float32).reshape(dims) + offset) * scale
        arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
        return arr, {
            "decode_mode": "native_u8_nhwc_dequantized",
            "byte_size": path.stat().st_size,
            "element_count": int(raw.size),
            "datatype": dtype_name,
            "dims": list(dims),
            "quant": quant,
        }

    if dtype_name == "QNN_DATATYPE_UFIXED_POINT_16":
        raw = np.fromfile(path, dtype=np.uint16)
        scale = float(quant.get("Scale", 0.0))
        offset = float(quant.get("Offset", 0.0))
        expected = int(np.prod(dims))
        if raw.size == expected:
            arr = (raw.astype(np.float32).reshape(dims) + offset) * scale
        elif raw.size == expected // 2:
            half_dims = (dims[0], dims[1], dims[2] // 2, dims[3])
            arr = (raw.astype(np.float32).reshape(half_dims) + offset) * scale
        else:
            raise RuntimeError(f"Unexpected uint16 element count={raw.size} for {path}")
        arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
        return arr, {
            "decode_mode": "native_u16_nhwc_dequantized",
            "byte_size": path.stat().st_size,
            "element_count": int(raw.size),
            "datatype": dtype_name,
            "dims": list(dims),
            "quant": quant,
        }

    if dtype_name == "QNN_DATATYPE_FLOAT_16":
        raw = np.fromfile(path, dtype=np.float16)
        arr = raw.reshape(dims)
        arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
        return arr, {
            "decode_mode": "native_float16_nhwc",
            "byte_size": path.stat().st_size,
            "element_count": int(raw.size),
            "datatype": dtype_name,
            "dims": list(dims),
        }

    raise RuntimeError(f"Unsupported native datatype={dtype_name} for {path}")


def _tensor_stats(x: np.ndarray) -> dict[str, object]:
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


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
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


def reconstruct_half_width(arr32: np.ndarray, mode: str) -> np.ndarray:
    half_width = arr32.reshape(1, 4, 128, 64)
    if mode == "repeat":
        return np.repeat(half_width, 2, axis=3).astype(np.float32, copy=False)
    repaired = np.empty((1, 4, 128, 128), dtype=np.float32)
    if mode == "odd_zero_even":
        repaired.fill(0.0)
        repaired[:, :, :, 1::2] = half_width
        return repaired
    if mode == "odd_prev_even":
        repaired[:, :, :, 1::2] = half_width
        repaired[:, :, :, 0] = half_width[:, :, :, 0]
        repaired[:, :, :, 2::2] = half_width[:, :, :, :-1]
        return repaired
    if mode == "odd_linear_even":
        repaired[:, :, :, 1::2] = half_width
        repaired[:, :, :, 0] = half_width[:, :, :, 0]
        repaired[:, :, :, 2:126:2] = 0.5 * (half_width[:, :, :, :-2] + half_width[:, :, :, 1:-1])
        repaired[:, :, :, -2] = 0.5 * (half_width[:, :, :, -2] + half_width[:, :, :, -1])
        repaired[:, :, :, -1] = half_width[:, :, :, -1]
        return repaired
    raise ValueError(f"Unsupported mode={mode}")


def load_phone_output(path: Path, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    if path.name.endswith("_native.raw") and path.with_name(path.name + ".json").exists():
        return _load_native_phone_output(path)
    native_path = path.with_name(path.name.replace(".raw", "_native.raw"))
    if native_path.exists() and native_path.with_name(native_path.name + ".json").exists():
        return _load_native_phone_output(native_path)

    expected = 1 * 4 * 128 * 128
    byte_size = path.stat().st_size
    if byte_size == expected * 2:
        arr32 = np.fromfile(path, dtype=np.float32)
        if arr32.size == expected // 2 and np.isfinite(arr32).all() and float(np.max(np.abs(arr32))) < 256.0:
            repaired = reconstruct_half_width(arr32, mode)
            return repaired, {"decode_mode": f"float32_halfwidth_{mode}", "byte_size": byte_size, "element_count": int(arr32.size)}
        arr16 = np.fromfile(path, dtype=np.float16).astype(np.float32)
        arr = np.transpose(arr16.reshape(1, 128, 128, 4), (0, 3, 1, 2)).astype(np.float32, copy=False)
        return arr, {"decode_mode": "float16_full_nhwc", "byte_size": byte_size, "element_count": int(arr16.size)}
    if byte_size == expected * 4:
        arr32 = np.fromfile(path, dtype=np.float32)
        if arr32.size == expected:
            arr = np.transpose(arr32.reshape(1, 128, 128, 4), (0, 3, 1, 2)).astype(np.float32, copy=False)
            return arr, {"decode_mode": "float32_full_nhwc", "byte_size": byte_size, "element_count": int(arr32.size)}
        if arr32.size == expected // 2:
            repaired = reconstruct_half_width(arr32, mode)
            return repaired, {"decode_mode": f"float32_halfwidth_{mode}", "byte_size": byte_size, "element_count": int(arr32.size)}
    raise RuntimeError(f"Unsupported phone output size for {path}: {byte_size}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare normalized ONNX outputs against saved phone outputs for all runtime_work steps")
    ap.add_argument("--onnx", default="d:/platform-tools/NPU/outputs/unet_ort_normalized.onnx")
    ap.add_argument("--runtime-work", default="d:/platform-tools/NPU/runtime_work")
    ap.add_argument("--half-width-mode", choices=["repeat", "odd_zero_even", "odd_prev_even", "odd_linear_even"], default="odd_zero_even")
    ap.add_argument("--out", default="d:/platform-tools/NPU/outputs/onnx_vs_phone_saved_steps.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = Path(args.runtime_work)
    session = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])
    inputs_meta = {inp.name: inp for inp in session.get_inputs()}
    input_types = {name: inp.type for name, inp in inputs_meta.items()}

    reports: list[dict[str, Any]] = []
    for step_dir in sorted(runtime_root.glob("step_*_*")):
        if not step_dir.is_dir():
            continue
        phone_raw = step_dir / "noise_pred.raw"
        phone_native = step_dir / "noise_pred_native.raw"
        if not phone_raw.exists() and not phone_native.exists():
            continue

        ort_inputs: dict[str, np.ndarray] = {}
        missing = []
        for name, ort_type in input_types.items():
            raw_path = step_dir / f"{name}.raw"
            if not raw_path.exists():
                missing.append(name)
                continue
            data = np.fromfile(raw_path, dtype=np.float16).astype(np.float32)
            shape = [dim if isinstance(dim, int) else 1 for dim in inputs_meta[name].shape]
            data = data.reshape(shape)
            if ort_type == "tensor(float16)":
                ort_inputs[name] = data.astype(np.float16, copy=False)
            else:
                ort_inputs[name] = data.astype(np.float32, copy=False)
        if missing:
            continue

        onnx_out = session.run(None, ort_inputs)[0].astype(np.float32, copy=False)
        if onnx_out.shape == (1, 128, 128, 4):
            onnx_out = np.transpose(onnx_out, (0, 3, 1, 2)).astype(np.float32, copy=False)

        phone_path = phone_raw if phone_raw.exists() else phone_native
        phone_out, decode_info = load_phone_output(phone_path, args.half_width_mode)
        reports.append({
            "step_dir": str(step_dir),
            "phone_path": str(phone_path),
            "phone_decode": decode_info,
            "onnx_output": _tensor_stats(onnx_out),
            "phone_output": _tensor_stats(phone_out),
            "diff": _diff_metrics(phone_out, onnx_out),
        })

    summary = {
        "onnx": str(Path(args.onnx)),
        "runtime_work": str(runtime_root),
        "half_width_mode": args.half_width_mode,
        "reports": reports,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved report: {out_path}")
    for item in reports:
        d = item["diff"]
        step_name = Path(str(item["step_dir"])).name
        print(step_name, f"cos={d.get('cosine', float('nan')):.6f}", f"rmse={d.get('rmse', float('nan')):.6f}", f"mae={d.get('mae', float('nan')):.6f}")


if __name__ == "__main__":
    main()
