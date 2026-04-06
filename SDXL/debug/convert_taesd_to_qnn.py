#!/usr/bin/env python3
"""Convert TAESD XL decoder ONNX → QNN FP16 context binary for phone preview.

TAESD XL is pure Conv2d + ReLU + Upsample — no GroupNorm, no attention.
Should convert cleanly without any patches.

Steps:
  1. qnn-onnx-converter  → model.cpp / model.bin
  2. qnn-model-lib-generator → libTAESDDecoder.so  (Android arm64)
    3. (on phone) qnn-context-binary-generator → taesd_decoder.serialized.bin.bin

Usage:
  # First export ONNX:
    python SDXL/debug/export_taesd_to_onnx.py

  # Then convert to QNN:
    python SDXL/debug/convert_taesd_to_qnn.py
    python SDXL/debug/convert_taesd_to_qnn.py --step 1   # converter only
    python SDXL/debug/convert_taesd_to_qnn.py --step 2   # model-lib only
    python SDXL/debug/convert_taesd_to_qnn.py --backend gpu
    python SDXL/debug/convert_taesd_to_qnn.py --step 3   # show phone ctxgen command
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ROOT       = Path(r"D:\platform-tools")
SDXL_NPU   = ROOT / "sdxl_npu"
TAESD_DIR  = SDXL_NPU / "taesd_decoder"
ONNX_PATH  = TAESD_DIR / "taesd_decoder.onnx"

QNN_SDK    = Path(r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130")
NDK_ROOT   = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")
QAIRT_PYTHON = QNN_SDK / "lib" / "python"

# Output directories
QNN_OUT    = TAESD_DIR / "qnn_model"
LIB_OUT    = TAESD_DIR / "android_lib"


def run(cmd, label: str, cwd=None, extra_env=None):
    print(f"\n{'='*60}\n  [{label}]\n{'='*60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=3600,
    )
    if result.returncode != 0:
        print(f"  [FAIL] exit {result.returncode}")
        sys.exit(result.returncode)
    print(f"  [OK] {label}")


# ── Step 1: ONNX → QNN model (converter) ──────────────────────────────────────

def step1_convert(onnx_path: Path, out_dir: Path):
    """qnn-onnx-converter: ONNX → QNN .cpp + .bin"""

    if not onnx_path.exists():
        print(f"ERROR: ONNX not found: {onnx_path}")
        print("Run: python SDXL/debug/export_taesd_to_onnx.py first")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    qairt_env = {
        "PYTHONPATH": str(QAIRT_PYTHON) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "QNN_SDK_ROOT": str(QNN_SDK),
        "TMPDIR": str(SDXL_NPU / "_qairt_tmp"),
    }
    (SDXL_NPU / "_qairt_tmp").mkdir(parents=True, exist_ok=True)

    # Use the existing repo-side QAIRT patch entrypoint so modern onnx versions
    # keep working with QAIRT 2.31's older converter expectations.
    patch_script = SCRIPT_DIR / "qnn_onnx_converter_expanddims_patch.py"
    cmd = [
        sys.executable,
        str(patch_script),
        "--input_network", str(onnx_path),
        "--output_path", str(out_dir / "model"),
        "--input_dim", "latents", "1,4,128,128",
        "--float_bitwidth", "16",
    ]

    run(cmd, "qnn-onnx-converter (TAESD FP16)", cwd=out_dir, extra_env=qairt_env)

    cpp = out_dir / "model.cpp"
    bin_ = out_dir / "model.bin"
    if not cpp.exists():
        # Folderized output
        cpp = out_dir / "model" / "model.cpp"
        bin_ = out_dir / "model" / "model.bin"

    if cpp.exists():
        print(f"\n  CPP: {cpp} ({cpp.stat().st_size/1e6:.2f} MB)")
        print(f"  BIN: {bin_} ({bin_.stat().st_size/1e6:.2f} MB)")
    else:
        print("  WARNING: expected model.cpp not found — check converter output")


# ── Step 2: Build Android .so ─────────────────────────────────────────────────

def step2_build_lib(qnn_model_dir: Path, lib_out: Path):
    """Build Android arm64-v8a .so from QNN model using the shared repo helper."""

    # Find model.cpp and model.bin
    cpp_path = qnn_model_dir / "model.cpp"
    if not cpp_path.exists():
        cpp_path = qnn_model_dir / "model" / "model.cpp"
    if not cpp_path.exists():
        # QAIRT frequently saves the generated C++ as an extensionless file named "model"
        extless = qnn_model_dir / "model"
        if extless.exists() and extless.is_file():
            cpp_path = extless
    if not cpp_path.exists():
        print(f"ERROR: model.cpp not found in {qnn_model_dir}")
        sys.exit(1)

    model_bin = qnn_model_dir / "model.bin"
    if not model_bin.exists():
        print(f"ERROR: model.bin not found in {qnn_model_dir}")
        sys.exit(1)

    lib_out.mkdir(parents=True, exist_ok=True)

    so_name = "libTAESDDecoder.so"
    so_path = lib_out / so_name
    helper_build_dir = lib_out / "_build"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_android_model_lib_windows.py"),
        "--sdk-root", str(QNN_SDK),
        "--model-cpp", str(cpp_path),
        "--model-bin", str(model_bin),
        "--ndk-root", str(NDK_ROOT),
        "--build-dir", str(helper_build_dir),
        "--lib-name", so_name,
    ]
    run(cmd, f"build Android .so: {so_name}", cwd=lib_out)

    built_so = helper_build_dir / "libs" / "arm64-v8a" / so_name
    if built_so.exists():
        shutil.copy2(built_so, so_path)

    if so_path.exists():
        print(f"  Built: {so_path} ({so_path.stat().st_size/1e6:.2f} MB)")
    else:
        print("  WARNING: .so not found after build")

    return so_path


# ── Step 3: Print phone ctxgen command ────────────────────────────────────────

def step3_phone_ctxgen(so_name: str = "libTAESDDecoder.so", backend: str = "gpu",
                       phone_base: str = "/data/local/tmp/sdxl_qnn"):
    backend = backend.lower().strip()
    if backend not in {"gpu", "htp"}:
        print(f"ERROR: unsupported backend '{backend}' (expected gpu or htp)")
        sys.exit(1)

    phone_base = phone_base.rstrip("/")
    phone_so = f"{phone_base}/model/{so_name}"
    phone_ctx = f"{phone_base}/context/taesd_decoder.serialized.bin.bin"
    ld = f"{phone_base}/lib"
    adsp = f"{ld};/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"
    backend_lib = f"{ld}/libQnnGpu.so" if backend == "gpu" else f"{ld}/libQnnHtp.so"
    backend_label = backend.upper()

    print(f"\n[step3] Phone-side context generation commands ({backend_label}):")
    print("─" * 60)
    print(f"# 1. Push .so to phone:")
    print(f"adb -s e01ad23a push {LIB_OUT}/{so_name} {phone_so}")
    if backend == "gpu":
        print(f"# 1b. Make sure GPU runtime assets are on phone:")
        print(f"#     {phone_base}/lib/libQnnGpu.so")
        print(f"#     {phone_base}/bin/qnn-gpu-target-server   (recommended when shipped by QAIRT)")
    print()
    print(f"# 2. Generate context binary (run in adb shell):")
    print(f"adb -s e01ad23a shell \\")
    print(f'  "export LD_LIBRARY_PATH={ld}:$LD_LIBRARY_PATH \\')
    print(f'   export ADSP_LIBRARY_PATH=\'{adsp}\' \\')
    print(f"   {phone_base}/bin/qnn-context-binary-generator \\")
    print(f"     --model {phone_so} \\")
    print(f"     --backend {backend_lib} \\")
    print(f"     --output_dir {phone_base}/context \\")
    print(f'     --binary_file taesd_decoder.serialized.bin"')
    print()
    print(f"# 3. Expected output: {phone_ctx}")
    print(f"# 4. Expected size: ~5-15 MB (vs full VAE ~151 MB)")
    print(f"# 5. Runtime hint: export SDXL_QNN_TAESD_BACKEND={backend}")
    print("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert TAESD XL ONNX → QNN FP16")
    ap.add_argument("--onnx", default=str(ONNX_PATH),
                    help="Path to taesd_decoder.onnx")
    ap.add_argument("--qnn-out", default=str(QNN_OUT),
                    help="QNN converter output directory")
    ap.add_argument("--lib-out", default=str(LIB_OUT),
                    help="Android .so output directory")
    ap.add_argument("--step", type=int, choices=[1, 2, 3], default=0,
                    help="Run only specific step (1=convert, 2=build-lib, 3=show-ctxgen)")
    ap.add_argument("--backend", choices=["gpu", "htp"], default="gpu",
                    help="Backend to target for the generated preview context (default: gpu)")
    ap.add_argument("--phone-base", default="/data/local/tmp/sdxl_qnn",
                    help="Phone-side base directory where preview assets are deployed")
    a = ap.parse_args()

    onnx_path = Path(a.onnx)
    qnn_out   = Path(a.qnn_out)
    lib_out   = Path(a.lib_out)

    print("[TAESD QNN FP16 Conversion]")
    print(f"  ONNX:    {onnx_path}")
    print(f"  QNN out: {qnn_out}")
    print(f"  Lib out: {lib_out}")
    print(f"  Backend: {a.backend}")

    if a.step in (0, 1):
        print("\n[STEP 1/3] ONNX → QNN model (FP16)")
        step1_convert(onnx_path, qnn_out)

    if a.step in (0, 2):
        print("\n[STEP 2/3] Build Android arm64 .so")
        step2_build_lib(qnn_out, lib_out)

    if a.step in (0, 3):
        print("\n[STEP 3/3] Phone context binary")
        step3_phone_ctxgen(backend=a.backend, phone_base=a.phone_base)

    print("\n[Done]")
    if a.step == 0:
        print("TAESD QNN pipeline complete.")
        print("After deploying to phone → runtime will auto-pick the TAESD QNN preview artifacts when available.")
        print(f"If needed, force the backend with: export SDXL_QNN_TAESD_BACKEND={a.backend}")
        print("Then run with --preview flag!")


if __name__ == "__main__":
    main()
