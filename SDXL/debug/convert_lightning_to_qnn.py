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


def _default_qairt_python(sdk_root: str) -> str:
    return str(Path(sdk_root) / "lib" / "python")


def _apply_thread_caps(env: dict, max_threads: int | None) -> None:
    if max_threads is None:
        return
    value = str(max_threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "ORT_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        env[key] = value


def _build_qairt_env(qnn_sdk_root: str, qairt_python: str | None, qairt_tmp_dir: Path, max_threads: int | None) -> dict:
    env = {
        "PYTHONPATH": (qairt_python or _default_qairt_python(qnn_sdk_root)) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "QNN_SDK_ROOT": qnn_sdk_root,
        "QAIRT_SDK_ROOT": qnn_sdk_root,
        "TMPDIR": str(qairt_tmp_dir),
        "TEMP": str(qairt_tmp_dir),
        "TMP": str(qairt_tmp_dir),
        # Prefer the newer name when QAIRT warns about the old one.
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "max_split_size_mb:256"),
    }
    _apply_thread_caps(env, max_threads)
    return env


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


def step4_generate_calibration(calib_out: Path, calib_samples: int, args):
    """Generate multi-sample calibration data with Lightning-merged UNet.
    
    Uses make_lightning_calibration.py which:
    - Scales latents by init_noise_sigma (critical for correct calibration)
    - Saves in NCHW float32 (as QNN converter expects)
    - Computes proper resnet biases from Lightning UNet
    """
    if (calib_out / "unet_extbias_input_list.txt").exists():
        print(f"[skip] Calibration data already exists: {calib_out}")
        return
    cmd = [
        PYTHON, str(ROOT / "NPU" / "make_lightning_calibration.py"),
        "--samples", str(calib_samples),
        "--out-dir", str(calib_out),
    ]

    if args.calib_device:
        cmd += ["--device", args.calib_device]
    if args.calib_memory_mode:
        cmd += ["--memory-mode", args.calib_memory_mode]
    if args.calib_text_encoder_device:
        cmd += ["--text-encoder-device", args.calib_text_encoder_device]
    if args.calib_unet_device:
        cmd += ["--unet-device", args.calib_unet_device]
    if args.diffusers_dir:
        cmd += ["--diffusers-dir", args.diffusers_dir]
    if args.merged_unet_dir:
        cmd += ["--merged-unet-dir", args.merged_unet_dir]
    if args.calib_prompts_json:
        cmd += ["--prompts-json", args.calib_prompts_json]
    if args.calib_prompt_limit is not None:
        cmd += ["--prompt-limit", str(args.calib_prompt_limit)]
    if args.calib_sampling_order:
        cmd += ["--sampling-order", args.calib_sampling_order]

    run(cmd, f"Calibration data generation ({calib_samples} samples)")


def step5_qnn_convert(mode: str, qnn_out: Path, calib_out: Path, args):
    """Run QNN converter with monkey-patches."""
    qnn_out.mkdir(parents=True, exist_ok=True)
    if (qnn_out / "model.cpp").exists() or (qnn_out / "model.bin").exists():
        print(f"[skip] QNN model already exists: {qnn_out}")
        return

    qairt_tmp_dir = Path(args.qairt_tmp_dir) if args.qairt_tmp_dir else (SDXL_NPU / "_qairt_tmp")
    qairt_tmp_dir.mkdir(exist_ok=True)
    qairt_env = _build_qairt_env(
        qnn_sdk_root=args.qnn_sdk_root,
        qairt_python=args.qairt_python,
        qairt_tmp_dir=qairt_tmp_dir,
        max_threads=args.converter_max_threads,
    )

    input_network = args.input_onnx or str(ONNX_FINAL / "unet.onnx")

    cmd = [
        PYTHON, str(SDXL_NPU / "qnn_onnx_converter_expanddims_patch.py"),
        "--input_network", input_network,
        "--output_path", str(qnn_out / "model"),
        "--float_bitwidth", "16",
    ]

    applied_int8_defaults: dict[str, object] = {}
    use_per_channel_quantization = args.use_per_channel_quantization
    bias_bitwidth = args.bias_bitwidth
    act_quantizer_calibration = args.act_quantizer_calibration
    percentile_calibration_value = args.percentile_calibration_value

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
        if args.int8_gentle_defaults:
            if not use_per_channel_quantization:
                use_per_channel_quantization = True
                applied_int8_defaults["use_per_channel_quantization"] = True
            if bias_bitwidth is None:
                bias_bitwidth = 32
                applied_int8_defaults["bias_bitwidth"] = 32
            if act_quantizer_calibration is None:
                act_quantizer_calibration = "percentile"
                applied_int8_defaults["act_quantizer_calibration"] = "percentile"
            if percentile_calibration_value is None:
                percentile_calibration_value = 99.99
                applied_int8_defaults["percentile_calibration_value"] = 99.99
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if applied_int8_defaults:
        print(f"[int8 defaults] applied gentle preset: {applied_int8_defaults}")

    if use_per_channel_quantization:
        cmd.append("--use_per_channel_quantization")
    if args.use_per_row_quantization:
        cmd.append("--use_per_row_quantization")
    if args.enable_per_row_quantized_bias:
        cmd.append("--enable_per_row_quantized_bias")
    if bias_bitwidth is not None:
        cmd += ["--bias_bitwidth", str(bias_bitwidth)]
    if act_quantizer_calibration:
        cmd += ["--act_quantizer_calibration", act_quantizer_calibration]
    if args.param_quantizer_calibration:
        cmd += ["--param_quantizer_calibration", args.param_quantizer_calibration]
    if args.act_quantizer_schema:
        cmd += ["--act_quantizer_schema", args.act_quantizer_schema]
    if args.param_quantizer_schema:
        cmd += ["--param_quantizer_schema", args.param_quantizer_schema]
    if percentile_calibration_value is not None:
        cmd += ["--percentile_calibration_value", str(percentile_calibration_value)]
    if args.quantizer_log:
        cmd += ["--quantizer_log", args.quantizer_log]
    if args.quantizer_log_level:
        cmd += ["--quantizer_log_level", args.quantizer_log_level]

    run(cmd, label, extra_env=qairt_env)


def step6_build_android(qnn_out: Path, model_lib: Path, args):
    """Build Android model lib (.so) for ARM64."""
    lib_name = args.android_lib_name
    if list(model_lib.glob(f"**/{lib_name}")):
        print(f"[skip] Android model lib already exists")
        return
    model_lib.mkdir(parents=True, exist_ok=True)
    model_cpp = qnn_out / "model"
    model_bin = qnn_out / "model.bin"
    run([
        PYTHON, str(SDXL_NPU / "build_android_model_lib_windows.py"),
        "--sdk-root", args.qnn_sdk_root,
        "--model-cpp", str(model_cpp),
        "--model-bin", str(model_bin),
        "--ndk-root", args.ndk_root,
        "--build-dir", str(model_lib),
        "--lib-name", lib_name,
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
    ap.add_argument("--qnn-sdk-root", type=str, default=QNN_SDK,
                    help="QAIRT/QNN SDK root (default preserves legacy 2.31 path)")
    ap.add_argument("--qairt-python", type=str, default=None,
                    help="Optional QAIRT python path override (default: <sdk>/lib/python)")
    ap.add_argument("--ndk-root", type=str, default=NDK_ROOT,
                    help="Android NDK root for model-lib build")
    ap.add_argument("--input-onnx", type=str, default=None,
                    help="Optional ONNX override for step 5 (e.g. static reshape 2.44 lab ONNX)")
    ap.add_argument("--calib-dir", type=str, default=None,
                    help="Use an existing calibration directory instead of the default mode-based path")
    ap.add_argument("--qnn-out-dir", type=str, default=None,
                    help="Optional QNN output directory override")
    ap.add_argument("--model-lib-dir", type=str, default=None,
                    help="Optional Android model-lib output directory override")
    ap.add_argument("--android-lib-name", type=str, default="libunet_lightning8step.so",
                    help="Output Android shared library name")
    ap.add_argument("--qairt-tmp-dir", type=str, default=None,
                    help="Temporary directory for QAIRT conversion")
    ap.add_argument("--converter-max-threads", type=int, default=None,
                    help="Cap CPU thread pools for conversion/quantization (OpenMP/MKL/ORT/etc)")

    ap.add_argument("--calib-device", type=str, default=None,
                    help="Pass-through device for make_lightning_calibration.py")
    ap.add_argument("--calib-memory-mode", choices=["legacy", "staged"], default=None,
                    help="Pass-through calibration memory mode")
    ap.add_argument("--calib-text-encoder-device", type=str, default=None,
                    help="Pass-through text encoder device for staged calibration")
    ap.add_argument("--calib-unet-device", type=str, default=None,
                    help="Pass-through UNet device for staged calibration")
    ap.add_argument("--calib-prompts-json", type=str, default=None,
                    help="Pass-through prompts JSON for calibration generation")
    ap.add_argument("--calib-prompt-limit", type=int, default=None,
                    help="Pass-through prompt limit for calibration generation")
    ap.add_argument("--calib-sampling-order", choices=["legacy", "cyclic_diverse"], default=None,
                    help="Pass-through sample ordering for calibration generation")
    ap.add_argument("--diffusers-dir", type=str, default=str(DIFFUSERS),
                    help="Diffusers pipeline directory")
    ap.add_argument("--merged-unet-dir", type=str, default=str(SDXL_NPU / "unet_lightning8step_merged"),
                    help="Lightning merged UNet directory")

    ap.add_argument("--use-per-channel-quantization", action="store_true",
                    help="Enable per-channel weight quantization for the converter")
    ap.add_argument("--use-per-row-quantization", action="store_true",
                    help="Enable per-row quantization for MatMul/FullyConnected ops")
    ap.add_argument("--enable-per-row-quantized-bias", action="store_true",
                    help="Enable per-row quantized bias when per-row quantization is active")
    ap.add_argument("--bias-bitwidth", type=int, default=None,
                    help="Optional bias quantization bitwidth override (8 or 32)")
    ap.add_argument("--act-quantizer-calibration", type=str, default=None,
                    help="Optional activation calibration method (min-max, mse, percentile, ...)")
    ap.add_argument("--param-quantizer-calibration", type=str, default=None,
                    help="Optional parameter calibration method")
    ap.add_argument("--act-quantizer-schema", type=str, default=None,
                    help="Optional activation quantization schema")
    ap.add_argument("--param-quantizer-schema", type=str, default=None,
                    help="Optional parameter quantization schema")
    ap.add_argument("--percentile-calibration-value", type=float, default=None,
                    help="Optional percentile calibration value")
    ap.add_argument("--quantizer-log", type=str, default=None,
                    help="Optional quantizer v2 log file path")
    ap.add_argument("--quantizer-log-level", type=str, default=None,
                    help="Optional quantizer v2 log level")
    ap.add_argument("--int8-gentle-defaults", dest="int8_gentle_defaults", action="store_true", default=True,
                    help="For --mode int8, auto-apply a safer preset when explicit quantizer flags are missing")
    ap.add_argument("--no-int8-gentle-defaults", dest="int8_gentle_defaults", action="store_false",
                    help="Disable the safer preset for --mode int8")
    args = ap.parse_args()

    mode = args.mode

    # Mode-dependent output dirs
    calib_out = Path(args.calib_dir) if args.calib_dir else (SDXL_NPU / f"calib_lightning_{mode}_{args.calib_samples}s")
    qnn_out   = Path(args.qnn_out_dir) if args.qnn_out_dir else (SDXL_NPU / f"qnn_lightning_{mode}")
    model_lib = Path(args.model_lib_dir) if args.model_lib_dir else (SDXL_NPU / f"qnn_lightning_{mode}_android")

    print(f"[config] Mode: {mode}, Calibration samples: {args.calib_samples}")
    print(f"[config] Calib dir: {calib_out}")
    print(f"[config] QNN out: {qnn_out}")
    print(f"[config] Model lib: {model_lib}")
    print(f"[config] QNN SDK root: {args.qnn_sdk_root}")
    if args.converter_max_threads is not None:
        print(f"[config] converter_max_threads: {args.converter_max_threads}")

    # Verify prerequisites
    if not MERGED_UNET.exists():
        print(f"[error] Merged UNet not found: {MERGED_UNET}")
        print("  Run: python NPU/bake_lora_into_unet.py")
        sys.exit(1)
    if mode != "fp16" and args.calib_dir is None and not CALIB_NPZ.exists():
        print(f"[error] Calibration NPZ not found: {CALIB_NPZ}")
        sys.exit(1)

    steps = [
        (1, lambda: step1_export_onnx()),
        (2, lambda: step2_rewrite_groupnorm()),
        (3, lambda: step3_rewrite_fp16bias()),
        (4, lambda: step4_generate_calibration(calib_out, args.calib_samples, args)),
        (5, lambda: step5_qnn_convert(mode, qnn_out, calib_out, args)),
        (6, lambda: step6_build_android(qnn_out, model_lib, args)),
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
