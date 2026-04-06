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
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = ROOT / "SDXL"
DEFAULT_CHECKPOINT = Path(r"J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v160.safetensors")


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
        print(f"  ONNX: {onnx.__version__}")
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
        help="Path to Lightning LoRA .safetensors (auto-downloads if not provided by the downstream script)",
    )
    ap.add_argument("--output-dir", type=str, default=str(ROOT / "build"), help="Output directory for generated artifacts")
    ap.add_argument("--skip-deploy", action="store_true", help="Reserved for future use; deploy is manual for now")
    ap.add_argument("--steps", type=int, default=8, help="Number of Lightning steps (default: 8)")
    args = ap.parse_args()

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
        if args.lightning_lora:
            print("  WARNING: custom --lightning-lora is currently ignored by bake_lora_into_unet.py; it uses its own configured Lightning source.")
        cmd = [
            sys.executable,
            SDXL_DIR / "bake_lora_into_unet.py",
            "--pipeline-dir",
            str(diffusers_dir),
            "--output-dir",
            str(merged_dir),
        ]
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

    onnx_unet = out / "onnx_unet"
    if not onnx_unet.exists():
        print("\n[Step 4/6] Exporting UNet to ONNX (extmaps surgery)...")
        tmp_pipeline = out / "_tmp_lightning_pipeline"
        ensure_tmp_lightning_pipeline(diffusers_dir, merged_dir, tmp_pipeline)
        run([
            sys.executable,
            SDXL_DIR / "export_sdxl_to_onnx.py",
            "--diffusers-dir",
            str(tmp_pipeline),
            "--out-dir",
            str(onnx_unet),
            "--component",
            "unet",
        ])
    else:
        print("[Step 4/6] UNet ONNX already exists, skipping.")

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
