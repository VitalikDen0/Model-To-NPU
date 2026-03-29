#!/usr/bin/env python3
"""
Full pipeline: Lightning-merged UNet → ONNX → QNN model lib for phone deployment.

Steps:
  1. Export UNet to ONNX (with extmaps surgery)
  2. Rewrite InstanceNorm → GroupNorm (opset 18)
  3. Rewrite bias inputs to fp16, remove trivial Casts
  4. Generate calibration data (multi-sample)
  5. Run QNN converter (W8A16: 8-bit weights, 16-bit activations)
  6. Build Android model lib

Modes:
  --mode w8a16   W8A16 quantization (default, best quality/performance tradeoff)
  --mode fp16    Pure FP16 (no quantization, larger model)
  --mode int8    Full INT8 (smallest, but quality risk)

Usage:
  python NPU/convert_lightning_to_qnn.py
  python NPU/convert_lightning_to_qnn.py --mode w8a16 --calib-samples 16
  python NPU/convert_lightning_to_qnn.py --start-from 5
"""
import argparse, os, subprocess, sys, shutil
from pathlib import Path

# ── paths ──
ROOT         = Path(r"D:\platform-tools")
SDXL_NPU     = ROOT / "sdxl_npu"
MERGED_UNET  = SDXL_NPU / "unet_lightning8step_merged" / "diffusion_pytorch_model.safetensors"
CALIB_NPZ    = SDXL_NPU / "calibration_data" / "1024x1024" / "calibration.npz"
DIFFUSERS    = SDXL_NPU / "diffusers_pipeline"

# output dirs — ONNX stages are shared across modes
ONNX_RAW     = SDXL_NPU / "onnx_lightning_extmaps"
ONNX_GN18    = SDXL_NPU / "onnx_lightning_extmaps_gn18"
ONNX_FINAL   = SDXL_NPU / "onnx_lightning_extmaps_gn18_fp16bias"

PYTHON       = sys.executable

QAIRT_PYTHON = r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130\lib\python"
QNN_SDK      = r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130"
NDK_ROOT     = r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358"


def run(cmd, label, cwd=None, extra_env=None):
    print(f"\n{'='*60}\n  [{label}]\n{'='*60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=7200,
    )
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited with code {result.returncode}")
        sys.exit(1)
    print(f"  [OK] {label}")


def _ensure_tmp_pipeline():
    """Ensure temp pipeline dir has Lightning UNet symlinked."""
    tmp_pipeline = SDXL_NPU / "_tmp_lightning_pipeline"
    unet_subdir = tmp_pipeline / "unet"
    unet_subdir.mkdir(parents=True, exist_ok=True)
    merged_dir = SDXL_NPU / "unet_lightning8step_merged"
    shutil.copy2(str(merged_dir / "config.json"), str(unet_subdir / "config.json"))
    dst_weights = unet_subdir / "diffusion_pytorch_model.safetensors"
    if not dst_weights.exists():
        try:
            os.link(str(merged_dir / "diffusion_pytorch_model.safetensors"), str(dst_weights))
        except OSError:
            shutil.copy2(str(merged_dir / "diffusion_pytorch_model.safetensors"), str(dst_weights))


def step1_export_onnx():
    """Export UNet to ONNX with extmaps surgery."""
    if (ONNX_RAW / "unet.onnx").exists():
        print(f"[skip] ONNX raw already exists: {ONNX_RAW / 'unet.onnx'}")
        return
    _ensure_tmp_pipeline()
    tmp_pipeline = SDXL_NPU / "_tmp_lightning_pipeline"
    run([
        PYTHON, str(SDXL_NPU / "export_sdxl_to_onnx.py"),
        "--diffusers-dir", str(tmp_pipeline),
        "--out-dir", str(ONNX_RAW),
        "--component", "unet",
        "--resolution", "1024x1024",
        "--opset", "17",
        "--timestep-input-mode", "rank2",
        "--resnet-temb-mode", "external_featuremaps",
        "--onnx-exporter", "legacy",
        "--skip-validate",
    ], "ONNX export (extmaps)")


def step2_rewrite_groupnorm():
    """Rewrite InstanceNorm patterns to GroupNormalization (opset 18)."""
    if (ONNX_GN18 / "unet.onnx").exists():
        print(f"[skip] GroupNorm rewrite already exists")
        return
    ONNX_GN18.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON, str(SDXL_NPU / "rewrite_onnx_instancenorm_to_groupnorm.py"),
        "--input", str(ONNX_RAW / "unet.onnx"),
        "--output", str(ONNX_GN18 / "unet.onnx"),
        "--target-opset", "18",
    ], "GroupNorm rewrite")


def step3_rewrite_fp16bias():
    """Remove trivial Casts, convert bias inputs to fp16."""
    if (ONNX_FINAL / "unet.onnx").exists():
        print(f"[skip] FP16 bias rewrite already exists")
        return
    ONNX_FINAL.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON, str(SDXL_NPU / "rewrite_onnx_extmaps_bias_inputs_to_fp16.py"),
        "--input", str(ONNX_GN18 / "unet.onnx"),
        "--output", str(ONNX_FINAL / "unet.onnx"),
    ], "FP16 bias rewrite")


def step4_generate_calibration(calib_out: Path, calib_samples: int):
    """Generate multi-sample calibration data with Lightning-merged UNet.
    
    Uses make_lightning_calibration.py which:
    - Scales latents by init_noise_sigma (critical for correct calibration)
    - Saves in NCHW float32 (as QNN converter expects)
    - Computes proper resnet biases from Lightning UNet
    """
    if (calib_out / "unet_extbias_input_list.txt").exists():
        print(f"[skip] Calibration data already exists: {calib_out}")
        return
    run([
        PYTHON, str(ROOT / "NPU" / "make_lightning_calibration.py"),
        "--samples", str(calib_samples),
        "--out-dir", str(calib_out),
    ], f"Calibration data generation ({calib_samples} samples)")


def step5_qnn_convert(mode: str, qnn_out: Path, calib_out: Path):
    """Run QNN converter with monkey-patches."""
    qnn_out.mkdir(parents=True, exist_ok=True)
    if (qnn_out / "model.cpp").exists() or (qnn_out / "model.bin").exists():
        print(f"[skip] QNN model already exists: {qnn_out}")
        return

    qairt_env = {
        "PYTHONPATH": QAIRT_PYTHON + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "QNN_SDK_ROOT": QNN_SDK,
        "TMPDIR": str(SDXL_NPU / "_qairt_tmp"),
    }
    (SDXL_NPU / "_qairt_tmp").mkdir(exist_ok=True)

    cmd = [
        PYTHON, str(SDXL_NPU / "qnn_onnx_converter_expanddims_patch.py"),
        "--input_network", str(ONNX_FINAL / "unet.onnx"),
        "--output_path", str(qnn_out / "model"),
        "--float_bitwidth", "16",
    ]

    input_list_file = calib_out / "unet_extbias_input_list.txt"

    if mode == "fp16":
        label = "QNN ONNX → model (FP16, no quantization)"
    elif mode == "w8a16":
        cmd += [
            "--input_list", str(input_list_file),
            "--act_bitwidth", "16",
            "--weights_bitwidth", "8",
        ]
        label = "QNN ONNX → model (W8A16)"
    elif mode == "int8":
        cmd += [
            "--input_list", str(input_list_file),
        ]
        label = "QNN ONNX → model (INT8)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    run(cmd, label, extra_env=qairt_env)


def step6_build_android(qnn_out: Path, model_lib: Path):
    """Build Android model lib (.so) for ARM64."""
    if list(model_lib.glob("**/libunet_lightning8step.so")):
        print(f"[skip] Android model lib already exists")
        return
    model_lib.mkdir(parents=True, exist_ok=True)
    model_cpp = qnn_out / "model"
    model_bin = qnn_out / "model.bin"
    run([
        PYTHON, str(SDXL_NPU / "build_android_model_lib_windows.py"),
        "--sdk-root", QNN_SDK,
        "--model-cpp", str(model_cpp),
        "--model-bin", str(model_bin),
        "--ndk-root", NDK_ROOT,
        "--build-dir", str(model_lib),
        "--lib-name", "libunet_lightning8step.so",
    ], "Android model lib build")


def main():
    ap = argparse.ArgumentParser(description="Lightning UNet → QNN pipeline")
    ap.add_argument("--mode", choices=["w8a16", "fp16", "int8"], default="w8a16",
                    help="Quantization mode (default: w8a16)")
    ap.add_argument("--calib-samples", type=int, default=16,
                    help="Number of calibration samples (default: 16)")
    ap.add_argument("--start-from", type=int, default=1, choices=range(1, 7),
                    help="Start from step N (1-6)")
    ap.add_argument("--stop-after", type=int, default=6, choices=range(1, 7),
                    help="Stop after step N")
    args = ap.parse_args()

    mode = args.mode

    # Mode-dependent output dirs
    calib_out = SDXL_NPU / f"calib_lightning_{mode}_{args.calib_samples}s"
    qnn_out   = SDXL_NPU / f"qnn_lightning_{mode}"
    model_lib = SDXL_NPU / f"qnn_lightning_{mode}_android"

    print(f"[config] Mode: {mode}, Calibration samples: {args.calib_samples}")
    print(f"[config] Calib dir: {calib_out}")
    print(f"[config] QNN out: {qnn_out}")
    print(f"[config] Model lib: {model_lib}")

    # Verify prerequisites
    if not MERGED_UNET.exists():
        print(f"[error] Merged UNet not found: {MERGED_UNET}")
        print("  Run: python NPU/bake_lora_into_unet.py")
        sys.exit(1)
    if mode != "fp16" and not CALIB_NPZ.exists():
        print(f"[error] Calibration NPZ not found: {CALIB_NPZ}")
        sys.exit(1)

    steps = [
        (1, lambda: step1_export_onnx()),
        (2, lambda: step2_rewrite_groupnorm()),
        (3, lambda: step3_rewrite_fp16bias()),
        (4, lambda: step4_generate_calibration(calib_out, args.calib_samples)),
        (5, lambda: step5_qnn_convert(mode, qnn_out, calib_out)),
        (6, lambda: step6_build_android(qnn_out, model_lib)),
    ]

    for n, fn in steps:
        if n < args.start_from:
            continue
        if n > args.stop_after:
            break
        fn()

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Mode: {mode}")
    print(f"{'='*60}")
    outputs = [
        ("ONNX raw", ONNX_RAW / "unet.onnx"),
        ("ONNX GN18", ONNX_GN18 / "unet.onnx"),
        ("ONNX final", ONNX_FINAL / "unet.onnx"),
        ("Calib data", calib_out / "unet_extbias_input_list.txt"),
        ("QNN model", qnn_out / "model.cpp"),
        ("Android .so", model_lib),
    ]
    for label, path in outputs:
        exists = path.exists() if path.is_file() else (path.is_dir() and any(path.iterdir()))
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")


if __name__ == "__main__":
    main()
