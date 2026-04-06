#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12
    return float(np.dot(av, bv) / denom)


def assess_image(image_path: str | Path) -> dict[str, object]:
    path = Path(image_path)
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    gray = img.mean(axis=2)

    gray_std = float(gray.std())
    if gray_std < 1e-6:
        return {
            "image": str(path),
            "verdict": "blank",
            "is_noise_like": True,
            "confidence": "high",
            "summary": "Изображение практически константное/чёрное.",
            "metrics": {
                "gray_std": gray_std,
                "neighbor_corr_x": 0.0,
                "neighbor_corr_y": 0.0,
                "corr_x_lag16": 0.0,
                "corr_y_lag16": 0.0,
                "corr_x_lag32": 0.0,
                "corr_y_lag32": 0.0,
                "long_corr_mean": 0.0,
                "edge_coherence": 1.0,
                "compress_bpp": 0.0,
            },
            "reasons": ["gray_std≈0 -> blank image"],
        }

    neighbor_corr_x = _corr(gray[:, :-1], gray[:, 1:])
    neighbor_corr_y = _corr(gray[:-1, :], gray[1:, :])
    corr_x_lag16 = _corr(gray[:, :-16], gray[:, 16:])
    corr_y_lag16 = _corr(gray[:-16, :], gray[16:, :])
    corr_x_lag32 = _corr(gray[:, :-32], gray[:, 32:])
    corr_y_lag32 = _corr(gray[:-32, :], gray[32:, :])
    long_corr_mean = float(np.mean([corr_x_lag16, corr_y_lag16, corr_x_lag32, corr_y_lag32]))

    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx_c = gx[1:-1, :]
    gy_c = gy[:, 1:-1]
    gmag = np.sqrt(gx_c * gx_c + gy_c * gy_c)
    theta = np.arctan2(gy_c, gx_c)
    weights = gmag + 1e-6
    edge_coherence = float(np.abs(np.sum(weights * np.exp(1j * 2 * theta)) / np.sum(weights)))

    compress_bpp = float(len(zlib.compress(gray.astype(np.uint8).tobytes(), level=9)) / gray.size)

    reasons: list[str] = []
    noise_votes = 0

    if long_corr_mean < 0.12:
        noise_votes += 2
        reasons.append(f"long_corr_mean={long_corr_mean:.3f} очень низкая -> дальняя пространственная структура почти отсутствует")
    elif long_corr_mean < 0.22:
        noise_votes += 1
        reasons.append(f"long_corr_mean={long_corr_mean:.3f} низкая -> структура слабая")
    else:
        reasons.append(f"long_corr_mean={long_corr_mean:.3f} -> дальняя структура заметна")

    if edge_coherence < 0.12:
        noise_votes += 1
        reasons.append(f"edge_coherence={edge_coherence:.3f} низкая -> направленные контуры выражены слабо")
    else:
        reasons.append(f"edge_coherence={edge_coherence:.3f} -> есть согласованные контуры")

    if compress_bpp > 0.79:
        noise_votes += 1
        reasons.append(f"compress_bpp={compress_bpp:.3f} высокая -> картинка плохо сжимается, похоже на текстурный шум")
    else:
        reasons.append(f"compress_bpp={compress_bpp:.3f} -> сжимаемость ближе к структурированному изображению")

    if gray_std > 80.0 and long_corr_mean < 0.18:
        noise_votes += 1
        reasons.append(f"gray_std={gray_std:.3f} высокая при слабой дальней корреляции -> вероятен сильный визуальный шум")

    verdict: str
    confidence: str
    if noise_votes >= 3:
        verdict = "likely_noise"
        confidence = "high"
    elif noise_votes >= 2:
        verdict = "possibly_noise"
        confidence = "medium"
    else:
        verdict = "has_structure"
        confidence = "medium"

    summary_map = {
        "likely_noise": "Автооценка считает изображение в основном шумовым/бессодержательным.",
        "possibly_noise": "Автооценка видит слабую структуру, но шум пока доминирует.",
        "has_structure": "Автооценка видит достаточно пространственной структуры; это уже не похоже на чистый шум.",
    }

    return {
        "image": str(path),
        "verdict": verdict,
        "is_noise_like": verdict != "has_structure",
        "confidence": confidence,
        "summary": summary_map[verdict],
        "metrics": {
            "gray_std": gray_std,
            "neighbor_corr_x": neighbor_corr_x,
            "neighbor_corr_y": neighbor_corr_y,
            "corr_x_lag16": corr_x_lag16,
            "corr_y_lag16": corr_y_lag16,
            "corr_x_lag32": corr_x_lag32,
            "corr_y_lag32": corr_y_lag32,
            "long_corr_mean": long_corr_mean,
            "edge_coherence": edge_coherence,
            "compress_bpp": compress_bpp,
        },
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assess whether a generated image is likely pure noise or contains scene structure")
    ap.add_argument("image")
    ap.add_argument("--json-out", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = assess_image(args.image)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()