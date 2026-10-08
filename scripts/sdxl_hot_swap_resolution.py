#!/usr/bin/env python3
"""
Quickly export additional SDXL ONNX resolution buckets from an already prepared work dir.

This avoids repeating checkpoint->diffusers conversion and Lightning merge. It is intended
as a fast "hot-swap" preparation step after a primary 1024x1024 build.

Example:
  python scripts/sdxl_hot_swap_resolution.py \
    --work-dir build/sdxl_work \
    --resolution 1216x832,832x1216
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = ROOT / "SDXL"


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))


def parse_resolution_list(raw: str, arg_name: str) -> list[tuple[int, int]]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{arg_name} must contain at least one WxH value")

    parsed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in values:
        if "x" not in token:
            raise SystemExit(f"{arg_name}: invalid resolution '{token}' (expected WxH)")
        ws, hs = token.split("x", 1)
        try:
            w = int(ws)
            h = int(hs)
        except ValueError as exc:
            raise SystemExit(f"{arg_name}: invalid resolution '{token}' (expected integers)") from exc
        if w <= 0 or h <= 0:
            raise SystemExit(f"{arg_name}: resolution must be positive, got '{token}'")
        if (w % 8) != 0 or (h % 8) != 0:
            raise SystemExit(f"{arg_name}: resolution '{token}' must be divisible by 8")
        pair = (w, h)
        if pair not in seen:
            seen.add(pair)
            parsed.append(pair)
    return parsed


def format_resolution_list(values: list[tuple[int, int]]) -> str:
    return ",".join(f"{w}x{h}" for w, h in values)


def ensure_tmp_lightning_pipeline(diffusers_dir: Path, merged_dir: Path, tmp_pipeline: Path) -> None:
    unet_dir = tmp_pipeline / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)

    config_src = merged_dir / "config.json"
    config_dst = unet_dir / "config.json"
    if config_src.exists() and not config_dst.exists():
        shutil.copy2(config_src, config_dst)

    weights_src = merged_dir / "diffusion_pytorch_model.safetensors"
    weights_dst = unet_dir / "diffusion_pytorch_model.safetensors"
    if weights_src.exists() and not weights_dst.exists():
        try:
            os.link(weights_src, weights_dst)
        except OSError:
            shutil.copy2(weights_src, weights_dst)

    for name in ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"):
        src = diffusers_dir / name
        dst = tmp_pipeline / name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)


def merge_manifest(manifest_path: Path, new_resolutions: list[tuple[int, int]]) -> None:
    data: dict[str, object]
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    old_hot = data.get("hot_swap_exported_resolutions", [])
    if not isinstance(old_hot, list):
        old_hot = []

    merged = {str(item) for item in old_hot}
    merged.update(f"{w}x{h}" for w, h in new_resolutions)

    if "default_resolution" not in data:
        data["default_resolution"] = "1024x1024"
    if "primary_exported_resolutions" not in data:
        data["primary_exported_resolutions"] = ["1024x1024"]

    data["hot_swap_exported_resolutions"] = sorted(merged)
    all_res = set(data.get("primary_exported_resolutions", []))
    all_res.update(data["hot_swap_exported_resolutions"])
    if data.get("default_resolution"):
        all_res.add(str(data["default_resolution"]))
    data["resolutions"] = sorted(all_res)
    data["updated_utc"] = datetime.now(timezone.utc).isoformat()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export extra SDXL ONNX resolution buckets from an existing build work dir")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "build" / "sdxl_work")
    ap.add_argument("--resolution", required=True, help="One or more WxH resolutions (comma-separated)")
    ap.add_argument("--include-vae", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest", type=Path, default=None, help="Optional path to resolution manifest (default: <work-dir>/resolution_manifest.json)")
    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    resolutions = parse_resolution_list(args.resolution, "--resolution")
    resolution_arg = format_resolution_list(resolutions)

    work_dir = args.work_dir.resolve()
    diffusers_dir = work_dir / "diffusers_pipeline"
    merged_dir = work_dir / "unet_lightning_merged"
    tmp_pipeline = work_dir / "_tmp_lightning_pipeline"
    out_dir = work_dir / "onnx_hot_swap"
    manifest_path = args.manifest.resolve() if args.manifest else (work_dir / "resolution_manifest.json")

    if not diffusers_dir.exists():
        raise SystemExit(f"diffusers pipeline not found: {diffusers_dir}")
    if not merged_dir.exists():
        raise SystemExit(f"merged lightning UNet not found: {merged_dir}")

    ensure_tmp_lightning_pipeline(diffusers_dir, merged_dir, tmp_pipeline)

    run([
        args.python,
        str(SDXL_DIR / "export_sdxl_to_onnx.py"),
        "--diffusers-dir", str(tmp_pipeline),
        "--out-dir", str(out_dir),
        "--component", "unet",
        "--resolution", resolution_arg,
        "--opset", "17",
        "--onnx-exporter", "legacy",
        "--timestep-input-mode", "rank2",
        "--resnet-temb-mode", "external_featuremaps",
        "--skip-validate",
    ])

    if args.include_vae:
        run([
            args.python,
            str(SDXL_DIR / "export_sdxl_to_onnx.py"),
            "--diffusers-dir", str(tmp_pipeline),
            "--out-dir", str(out_dir),
            "--component", "vae",
            "--resolution", resolution_arg,
            "--opset", "17",
            "--onnx-exporter", "legacy",
            "--skip-validate",
        ])

    merge_manifest(manifest_path, resolutions)

    print("\n[done]")
    print(f"Work dir:  {work_dir}")
    print(f"Exported:  {resolution_arg}")
    print(f"Out dir:   {out_dir}")
    print(f"Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
