#!/usr/bin/env python3
"""
Experimental SDXL build helper.

This helper currently automates the early, reproducible stages of the SDXL
pipeline inside this repository:

  checkpoint -> diffusers conversion -> Lightning LoRA merge -> ONNX export

The later QNN conversion / Android model-lib stages are still being re-tested
end-to-end and are therefore documented in README instead of being launched
blindly from here.

Usage:
    python scripts/build_all.py
    python scripts/build_all.py --checkpoint path/to/model.safetensors
  python scripts/build_all.py --help
"""
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
DEFAULT_CHECKPOINT = Path(r"J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v170.safetensors")


def run(cmd, cwd=None):
    print(f"\n{'='*60}")
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd or ROOT))


def resolve_checkpoint_arg(raw_checkpoint: str | None) -> Path:
    """Resolve checkpoint path from argument or interactive prompt."""
    if raw_checkpoint:
        checkpoint = Path(raw_checkpoint).expanduser()
    else:
        if not (sys.stdin and sys.stdin.isatty()):
            print("ERROR: --checkpoint is required in non-interactive mode.")
            sys.exit(2)
        entered = input(f"Path to SDXL checkpoint (.safetensors) [{DEFAULT_CHECKPOINT}]: ").strip()
        checkpoint = Path(entered) if entered else DEFAULT_CHECKPOINT
        checkpoint = checkpoint.expanduser()

    try:
        return checkpoint.resolve()
    except OSError:
        return checkpoint.absolute()


def ensure_tmp_lightning_pipeline(diffusers_dir: Path, merged_dir: Path, tmp_pipeline: Path):
    """Create a minimal temp pipeline that reuses the merged Lightning UNet."""
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


def write_resolution_manifest(
    manifest_path: Path,
    primary_resolutions: list[tuple[int, int]],
    hot_swap_resolutions: list[tuple[int, int]],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "default_resolution": f"{primary_resolutions[0][0]}x{primary_resolutions[0][1]}",
        "primary_exported_resolutions": [f"{w}x{h}" for w, h in primary_resolutions],
        "hot_swap_exported_resolutions": [f"{w}x{h}" for w, h in hot_swap_resolutions],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_csv(raw: str, arg_name: str) -> list[float]:
    values = parse_csv_values(raw)
    parsed: list[float] = []
    for idx, token in enumerate(values, start=1):
        try:
            parsed.append(float(token))
        except ValueError as exc:
            raise SystemExit(f"{arg_name}: invalid float at position {idx}: {token!r}") from exc
    return parsed


def export_unet_with_extmaps(tmp_pipeline: Path, out_dir: Path, resolution_arg: str) -> None:
    run([
        sys.executable,
        SDXL_DIR / "export_sdxl_to_onnx.py",
        "--diffusers-dir",
        str(tmp_pipeline),
        "--out-dir",
        str(out_dir),
        "--component",
        "unet",
        "--resolution",
        resolution_arg,
        "--opset",
        "17",
        "--onnx-exporter",
        "legacy",
        "--timestep-input-mode",
        "rank2",
        "--resnet-temb-mode",
        "external_featuremaps",
        "--skip-validate",
    ])


def check_prereqs():
    """Check that required tools are available."""
    print("[*] Checking prerequisites...")

    v = sys.version_info
    if v.minor != 10:
        print(f"  WARNING: Python {v.major}.{v.minor} (recommended: 3.10.x)")

    try:
        import torch
        print(
            f"  PyTorch: {torch.__version__}"
            + (f" (CUDA {torch.version.cuda})" if torch.cuda.is_available() else " (CPU)")
        )
    except ImportError:
        print("  ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import diffusers
        print(f"  diffusers: {diffusers.__version__}")
    except ImportError:
        print("  ERROR: diffusers not installed. Run: pip install diffusers transformers")
        sys.exit(1)

    try:
        import onnx
        print(f"  ONNX: {getattr(onnx, '__version__', 'unknown')}")
    except ImportError:
        print("  ERROR: onnx not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    adb = os.environ.get("ADB_PATH", str(ROOT / "adb.exe"))
    if not Path(adb).exists():
        adb = "adb"
    try:
        r = subprocess.run([adb, "version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"  ADB: {r.stdout.strip().splitlines()[0]}")
        else:
            print("  WARNING: ADB not available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  WARNING: ADB not found. Required for phone deployment.")

    qnn_root = os.environ.get("QNN_SDK_ROOT", "")
    if qnn_root and Path(qnn_root).exists():
        print(f"  QNN SDK: {qnn_root}")
    else:
        print("  WARNING: QNN_SDK_ROOT not set. Required for later QNN conversion stages.")

    print()


def main():
    ap = argparse.ArgumentParser(description="Build current SDXL-on-NPU stages")
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to SDXL .safetensors checkpoint (if omitted, script asks interactively)",
    )
    ap.add_argument(
        "--lightning-lora",
        type=str,
        default=None,
        help="Optional primary LoRA path/reference for base UNet merge",
    )
    ap.add_argument(
        "--lightning-lora-scale",
        type=float,
        default=1.0,
        help="Primary LoRA fuse scale for --lightning-lora (default: 1.0)",
    )
    ap.add_argument(
        "--hot-swap-loras",
        type=str,
        default="",
        help="Optional comma-separated LoRA paths/references for hot-swap slots (max 4)",
    )
    ap.add_argument(
        "--hot-swap-lora-scales",
        type=str,
        default="",
        help="Optional comma-separated fuse scales for --hot-swap-loras (same count)",
    )
    ap.add_argument(
        "--verify-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable slow merge verification in bake_lora_into_unet.py (default: off)",
    )
    ap.add_argument("--output-dir", type=str, default=str(ROOT / "build"), help="Output directory for generated artifacts")
    ap.add_argument("--skip-deploy", action="store_true", help="Reserved for future use; deploy is manual for now")
    ap.add_argument("--steps", type=int, default=8, help="Number of Lightning steps (default: 8)")
    ap.add_argument("--resolution", type=str, default="1024x1024",
                    help="Primary image resolution WxH for UNet export (default: 1024x1024). "
                         "Supports comma-separated list: 1024x1024,896x1152,832x1216")
    ap.add_argument(
        "--hot-swap-resolutions",
        type=str,
        default="",
        help=(
            "Optional extra WxH resolutions to export as quick hot-swap ONNX buckets "
            "after the main build (comma-separated, e.g. 1216x832,832x1216)."
        ),
    )
    args = ap.parse_args()

    primary_resolutions = parse_resolution_list(args.resolution, "--resolution")
    hot_swap_resolutions = []
    if args.hot_swap_resolutions.strip():
        hot_swap_candidates = parse_resolution_list(args.hot_swap_resolutions, "--hot-swap-resolutions")
        primary_set = set(primary_resolutions)
        hot_swap_resolutions = [r for r in hot_swap_candidates if r not in primary_set]

    primary_resolution_arg = format_resolution_list(primary_resolutions)
    hot_swap_resolution_arg = format_resolution_list(hot_swap_resolutions)

    hot_swap_loras = parse_csv_values(args.hot_swap_loras)
    if len(hot_swap_loras) > 4:
        raise SystemExit("--hot-swap-loras supports up to 4 entries")

    if args.hot_swap_lora_scales.strip():
        hot_swap_lora_scales = parse_float_csv(args.hot_swap_lora_scales, "--hot-swap-lora-scales")
    else:
        hot_swap_lora_scales = [1.0] * len(hot_swap_loras)

    if hot_swap_lora_scales and len(hot_swap_lora_scales) != len(hot_swap_loras):
        raise SystemExit("--hot-swap-lora-scales must match --hot-swap-loras count")

    check_prereqs()

    checkpoint = resolve_checkpoint_arg(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model-to-NPU Pipeline for Snapdragon")
    print("Current automated target: SDXL")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output:     {out}")
    print(f"  Steps:      {args.steps}")
    print(f"  Resolution: {primary_resolution_arg}")
    if hot_swap_resolutions:
        print(f"  Hot-swap:   {hot_swap_resolution_arg}")
    if args.lightning_lora:
        print(f"  Primary LoRA: {args.lightning_lora} (scale={args.lightning_lora_scale:.4f})")
    if hot_swap_loras:
        print(f"  LoRA slots: {len(hot_swap_loras)}")
    print("  Status:     QNN/deploy stages are under repeated re-validation")
    print()

    diffusers_dir = out / "diffusers_pipeline"
    if not diffusers_dir.exists():
        print("\n[Step 1/6] Converting checkpoint to diffusers format...")
        run([
            sys.executable,
            SDXL_DIR / "convert_sdxl_checkpoint_to_diffusers.py",
            "--input",
            str(checkpoint),
            "--output",
            str(diffusers_dir),
        ])
    else:
        print("[Step 1/6] Diffusers pipeline already exists, skipping.")

    merged_dir = out / "unet_lightning_merged"
    if not merged_dir.exists():
        print("\n[Step 2/6] Merging Lightning LoRA into UNet...")
        cmd = [
            sys.executable,
            SDXL_DIR / "bake_lora_into_unet.py",
            "--pipeline-dir",
            str(diffusers_dir),
            "--output-dir",
            str(merged_dir),
            "--lora-scale",
            str(args.lightning_lora_scale),
        ]
        if args.lightning_lora:
            cmd.extend(["--lora-path", args.lightning_lora])
        cmd.append("--verify" if args.verify_merge else "--no-verify")
        run(cmd)
    else:
        print("[Step 2/6] Merged UNet already exists, skipping.")

    onnx_clip_vae = out / "onnx_clip_vae"
    if not onnx_clip_vae.exists():
        print("\n[Step 3/6] Exporting CLIP-L, CLIP-G, VAE to ONNX...")
        run([
            sys.executable,
            SDXL_DIR / "export_clip_vae_to_onnx.py",
            "--diffusers-dir",
            str(diffusers_dir),
            "--out-dir",
            str(onnx_clip_vae),
        ])
    else:
        print("[Step 3/6] CLIP/VAE ONNX already exists, skipping.")

    tmp_pipeline = out / "_tmp_lightning_pipeline"
    onnx_unet = out / "onnx_unet"
    if not onnx_unet.exists():
        print("\n[Step 4/6] Exporting UNet to ONNX (extmaps surgery)...")
        ensure_tmp_lightning_pipeline(diffusers_dir, merged_dir, tmp_pipeline)
        export_unet_with_extmaps(tmp_pipeline, onnx_unet, primary_resolution_arg)
    else:
        print("[Step 4/6] UNet ONNX already exists, skipping.")

    if hot_swap_resolutions:
        print("\n[Step 4b/6] Exporting optional hot-swap ONNX buckets...")
        ensure_tmp_lightning_pipeline(diffusers_dir, merged_dir, tmp_pipeline)

        hot_dir = out / "onnx_hot_swap"
        export_unet_with_extmaps(tmp_pipeline, hot_dir, hot_swap_resolution_arg)

        run([
            sys.executable,
            SDXL_DIR / "export_sdxl_to_onnx.py",
            "--diffusers-dir",
            str(tmp_pipeline),
            "--out-dir",
            str(hot_dir),
            "--component",
            "vae",
            "--resolution",
            hot_swap_resolution_arg,
            "--opset",
            "17",
            "--onnx-exporter",
            "legacy",
            "--skip-validate",
        ])

    lora_slot_manifest: list[dict[str, object]] = []
    if hot_swap_loras:
        print("\n[Step 4c/6] Building hot-swap LoRA UNet slots...")
        export_slot_resolution_arg = primary_resolution_arg
        if hot_swap_resolutions:
            export_slot_resolution_arg = format_resolution_list(primary_resolutions + hot_swap_resolutions)

        for slot_index, lora_path in enumerate(hot_swap_loras, start=1):
            lora_scale = hot_swap_lora_scales[slot_index - 1]
            slot_tag = f"slot{slot_index}"
            merged_slot_dir = out / f"unet_lora_{slot_tag}_merged"
            slot_tmp_pipeline = out / f"_tmp_lora_{slot_tag}_pipeline"
            slot_onnx_dir = out / f"onnx_unet_lora_{slot_tag}"

            if not merged_slot_dir.exists():
                print(f"  [LoRA {slot_tag}] Merging: {lora_path} (scale={lora_scale:.4f})")
                cmd = [
                    sys.executable,
                    SDXL_DIR / "bake_lora_into_unet.py",
                    "--pipeline-dir", str(diffusers_dir),
                    "--output-dir", str(merged_slot_dir),
                    "--lora-path", lora_path,
                    "--lora-scale", str(lora_scale),
                    "--verify" if args.verify_merge else "--no-verify",
                ]
                run(cmd)
            else:
                print(f"  [LoRA {slot_tag}] merged UNet exists, skipping merge")

            if not slot_onnx_dir.exists():
                ensure_tmp_lightning_pipeline(diffusers_dir, merged_slot_dir, slot_tmp_pipeline)
                export_unet_with_extmaps(slot_tmp_pipeline, slot_onnx_dir, export_slot_resolution_arg)
            else:
                print(f"  [LoRA {slot_tag}] ONNX exists, skipping export")

            lora_slot_manifest.append(
                {
                    "slot": slot_tag,
                    "lora": lora_path,
                    "scale": lora_scale,
                    "merged_unet_dir": str(merged_slot_dir),
                    "onnx_dir": str(slot_onnx_dir),
                    "resolutions": [f"{w}x{h}" for w, h in (primary_resolutions + hot_swap_resolutions)],
                }
            )

    manifest_path = out / "resolution_manifest.json"
    write_resolution_manifest(manifest_path, primary_resolutions, hot_swap_resolutions)
    print(f"\n[info] Resolution manifest: {manifest_path}")

    if lora_slot_manifest:
        lora_manifest_path = out / "lora_hot_swap_manifest.json"
        lora_manifest_payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "max_supported_slots": 4,
            "slots": lora_slot_manifest,
        }
        lora_manifest_path.write_text(
            json.dumps(lora_manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[info] LoRA slot manifest: {lora_manifest_path}")

    print("\n[Step 5/6] QNN conversion is currently under re-validation.")
    print(f"  Use the manual SDXL scripts in: {SDXL_DIR}")

    if not args.skip_deploy:
        print("\n[Step 6/6] Phone deployment is also handled manually for now.")
        print(f"  When artifacts are ready, use: {ROOT / 'scripts' / 'deploy_to_phone.py'}")
    else:
        print("[Step 6/6] Skipping phone deployment (--skip-deploy).")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nArtifacts in: {out}")
    print("\nGenerate images:")
    print('  python phone_generate.py "your prompt here"')
    print("\nNote: QNN conversion and deploy steps remain documented in README while they are being re-tested.")


if __name__ == "__main__":
    main()
