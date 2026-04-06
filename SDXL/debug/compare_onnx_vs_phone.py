#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def load_phone_output(path: str | Path, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    p = Path(path)
    if p.name.endswith("_native.raw") and p.with_name(p.name + ".json").exists():
        return _load_native_phone_output(p)
    native_path = p.with_name(p.name.replace(".raw", "_native.raw"))
    if native_path.exists() and native_path.with_name(native_path.name + ".json").exists():
        return _load_native_phone_output(native_path)

    expected = 1 * 4 * 128 * 128
    byte_size = p.stat().st_size
    if byte_size == expected * 2:
        arr32 = np.fromfile(p, dtype=np.float32)
        if arr32.size == expected // 2 and np.isfinite(arr32).all() and float(np.max(np.abs(arr32))) < 256.0:
            repaired = reconstruct_half_width(arr32, mode)
            return repaired, {
                "decode_mode": f"float32_halfwidth_{mode}",
                "byte_size": byte_size,
                "element_count": int(arr32.size),
            }
        arr16 = np.fromfile(p, dtype=np.float16).astype(np.float32)
        arr = np.transpose(arr16.reshape(1, 128, 128, 4), (0, 3, 1, 2)).astype(np.float32, copy=False)
        return arr, {
            "decode_mode": "float16_full_nhwc",
            "byte_size": byte_size,
            "element_count": int(arr16.size),
        }
    if byte_size == expected * 4:
        arr32 = np.fromfile(p, dtype=np.float32)
        if arr32.size == expected:
            arr = np.transpose(arr32.reshape(1, 128, 128, 4), (0, 3, 1, 2)).astype(np.float32, copy=False)
            return arr, {
                "decode_mode": "float32_full_nhwc",
                "byte_size": byte_size,
                "element_count": int(arr32.size),
            }
        if arr32.size == expected // 2:
            repaired = reconstruct_half_width(arr32, mode)
            return repaired, {
                "decode_mode": f"float32_halfwidth_{mode}",
                "byte_size": byte_size,
                "element_count": int(arr32.size),
            }
    raise RuntimeError(f"Unsupported phone output size: {byte_size}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare normalized ONNX output against saved phone output")
    ap.add_argument("--onnx-output-json", required=True, help="JSON from compare_unet_pytorch_vs_onnx.py")
    ap.add_argument("--phone-raw", required=True)
    ap.add_argument("--half-width-mode", choices=["repeat", "odd_zero_even", "odd_prev_even", "odd_linear_even"], default="odd_zero_even")
    ap.add_argument("--out", default="d:/platform-tools/NPU/outputs/onnx_vs_phone_step0_cond.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    onnx_report = json.loads(Path(args.onnx_output_json).read_text(encoding="utf-8"))
    phone, phone_decode = load_phone_output(args.phone_raw, args.half_width_mode)

    # Reconstruct ONNX output by reusing saved stats is impossible; load from sidecar npy if present would be nicer.
    # For now this script expects the ONNX tensor to be stored beside the JSON.
    npy_path = Path(args.onnx_output_json).with_suffix('.onnx_output.npy')
    if not npy_path.exists():
        raise RuntimeError(f"Expected ONNX tensor dump next to JSON: {npy_path}")
    onnx = np.load(npy_path).astype(np.float32, copy=False)

    report = {
        "onnx_json": str(Path(args.onnx_output_json)),
        "phone_raw": str(Path(args.phone_raw)),
        "phone_decode": phone_decode,
        "onnx_output": _tensor_stats(onnx),
        "phone_output": _tensor_stats(phone),
        "diff": _diff_metrics(phone, onnx),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved report: {out_path}")
    print(json.dumps(report["diff"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
