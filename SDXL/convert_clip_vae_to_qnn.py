#!/usr/bin/env python3
"""
Convert CLIP-L, CLIP-G, VAE decoder ONNX → QNN model libs for phone deployment.

Pre-processing:
  - CLIP-G: consolidate external data into single ONNX
  - CLIP-L/G: convert int64 inputs to int32 (QNN doesn't support int64)
  - VAE: rewrite InstanceNorm → GroupNorm (opset 18)

Conversion:
  - FP16 by default (no quantization needed for these small models)

Usage:
  python NPU/convert_clip_vae_to_qnn.py --component clip_l
  python NPU/convert_clip_vae_to_qnn.py --component clip_g
  python NPU/convert_clip_vae_to_qnn.py --component vae
  python NPU/convert_clip_vae_to_qnn.py --component all
  python NPU/convert_clip_vae_to_qnn.py --component all --start-from 2
"""
import argparse, os, subprocess, sys, shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

ROOT         = Path(r"D:\platform-tools")
SDXL_NPU     = ROOT / "sdxl_npu"
ONNX_SRC     = SDXL_NPU / "onnx_clip_vae"

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
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
        env=env, timeout=3600,
    )
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited with code {result.returncode}")
        sys.exit(1)
    print(f"  [OK] {label}")


# ── Pre-processing ──

def consolidate_external_data(src_onnx: Path, dst_onnx: Path):
    """Load ONNX with external data and save with single external data file.
    
    For models >2GB, we use onnx.save with external data to avoid protobuf limits.
    """
    if dst_onnx.exists():
        print(f"[skip] Consolidated ONNX already exists: {dst_onnx}")
        return
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"[consolidate] Loading {src_onnx} with external data...")
    model = onnx.load(str(src_onnx), load_external_data=True)
    print(f"[consolidate] Saving consolidated model to {dst_onnx}...")
    # Save with single external data file to avoid protobuf 2GB limit
    ext_data_name = dst_onnx.stem + ".data"
    onnx.save(model, str(dst_onnx),
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=ext_data_name,
              size_threshold=1024)
    proto_mb = dst_onnx.stat().st_size / (1024 * 1024)
    data_path = dst_onnx.parent / ext_data_name
    data_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0
    print(f"[consolidate] Done: proto={proto_mb:.1f} MB, data={data_mb:.1f} MB")


def convert_int64_inputs_to_int32(src_onnx: Path, dst_onnx: Path):
    """Convert int64 graph inputs to int32 for QNN compatibility.
    
    Handles both regular and external-data ONNX files.
    """
    if dst_onnx.exists():
        print(f"[skip] Int32-converted ONNX already exists: {dst_onnx}")
        return
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"[int64→int32] Loading {src_onnx}...")

    # Detect external data
    proto_only = onnx.load(str(src_onnx), load_external_data=False)
    has_ext = any(
        t.data_location == TensorProto.EXTERNAL
        for t in proto_only.graph.initializer
    )
    
    # Load full model
    model = onnx.load(str(src_onnx), load_external_data=has_ext)

    changed = False
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if tt.elem_type == TensorProto.INT64:
            print(f"  Converting input '{inp.name}' from INT64 to INT32")
            tt.elem_type = TensorProto.INT32
            changed = True

    if not changed:
        print("  No INT64 inputs found, copying as-is.")
        if has_ext:
            # Copy all files in source directory
            for f in src_onnx.parent.iterdir():
                if f.name.startswith(src_onnx.stem):
                    shutil.copy2(f, dst_onnx.parent / f.name)
        else:
            shutil.copy2(src_onnx, dst_onnx)
        return

    # Also convert any Cast nodes that produce INT64 to INT32
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.INT64:
                    attr.i = TensorProto.INT32

    # Convert INT64 initializers/constants to INT32
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            new_init = numpy_helper.from_array(arr.astype(np.int32), name=init.name)
            init.CopyFrom(new_init)

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.INT64:
                    arr = numpy_helper.to_array(attr.t)
                    new_t = numpy_helper.from_array(arr.astype(np.int32))
                    attr.t.CopyFrom(new_t)

    # Save — use external data for large models
    if has_ext:
        ext_data_name = dst_onnx.stem + ".data"
        onnx.save(model, str(dst_onnx),
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=ext_data_name,
                  size_threshold=1024)
        proto_mb = dst_onnx.stat().st_size / (1024 * 1024)
        data_path = dst_onnx.parent / ext_data_name
        data_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0
        print(f"[int64→int32] Done: {dst_onnx.name} (proto={proto_mb:.1f} MB, data={data_mb:.1f} MB)")
    else:
        # Clear external data location flags
        for tensor in model.graph.initializer:
            tensor.data_location = TensorProto.DEFAULT
        onnx.save(model, str(dst_onnx))
        size_mb = dst_onnx.stat().st_size / (1024 * 1024)
        print(f"[int64→int32] Done: {dst_onnx.name} ({size_mb:.1f} MB)")


def rewrite_vae_instancenorm(src_onnx: Path, dst_onnx: Path):
    """Rewrite InstanceNorm → GroupNorm in VAE for QNN compatibility."""
    if dst_onnx.exists():
        print(f"[skip] VAE GroupNorm rewrite already exists: {dst_onnx}")
        return
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON, str(SDXL_NPU / "rewrite_onnx_instancenorm_to_groupnorm.py"),
        "--input", str(src_onnx),
        "--output", str(dst_onnx),
        "--target-opset", "18",
    ], "VAE InstanceNorm → GroupNorm rewrite")


# ── QNN conversion ──

def qnn_convert_fp16(input_onnx: Path, qnn_out: Path, component_name: str):
    """Run QNN converter in FP16 mode with monkey-patches."""
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
        "--input_network", str(input_onnx),
        "--output_path", str(qnn_out / "model"),
        "--float_bitwidth", "16",
    ]

    run(cmd, f"QNN ONNX → model FP16 ({component_name})", extra_env=qairt_env)


# ── Android model lib build ──

def build_android_lib(qnn_out: Path, lib_dir: Path, lib_name: str):
    """Build Android .so from QNN model."""
    final_so = lib_dir / "libs" / "arm64-v8a" / lib_name
    if final_so.exists():
        print(f"[skip] Android lib already exists: {final_so}")
        return
    lib_dir.mkdir(parents=True, exist_ok=True)

    model_cpp = qnn_out / "model"
    model_bin = qnn_out / "model.bin"

    if not model_cpp.exists() and (qnn_out / "model.cpp").exists():
        model_cpp = qnn_out / "model.cpp"

    run([
        PYTHON, str(SDXL_NPU / "build_android_model_lib_windows.py"),
        "--sdk-root", QNN_SDK,
        "--model-cpp", str(model_cpp),
        "--model-bin", str(model_bin),
        "--ndk-root", NDK_ROOT,
        "--build-dir", str(lib_dir),
        "--lib-name", lib_name,
    ], f"Android model lib build ({lib_name})")


# ── Component pipelines ──

def pipeline_clip_l(start_from: int):
    """CLIP-L: QNN FP16, Android lib. No int64→int32 needed (QNN handles int64 natively)."""
    component = "clip_l"
    src_onnx = ONNX_SRC / "clip_l.onnx"
    qnn_out = SDXL_NPU / "qnn_clip_l_fp16"
    lib_dir = SDXL_NPU / "qnn_clip_l_fp16_android"
    lib_name = "libclip_l.so"

    print(f"\n{'#'*60}\n  CLIP-L Pipeline\n{'#'*60}")

    if start_from <= 2:
        qnn_convert_fp16(src_onnx, qnn_out, component)
    if start_from <= 3:
        build_android_lib(qnn_out, lib_dir, lib_name)


def pipeline_clip_g(start_from: int):
    """CLIP-G: consolidate external data → QNN FP16, Android lib.
    No int64→int32 needed (QNN handles int64 natively).
    """
    component = "clip_g"
    src_onnx = ONNX_SRC / "clip_g.onnx"
    prep_dir = SDXL_NPU / "onnx_clip_g_prepared"
    consolidated = prep_dir / "clip_g.onnx"
    qnn_out = SDXL_NPU / "qnn_clip_g_fp16"
    lib_dir = SDXL_NPU / "qnn_clip_g_fp16_android"
    lib_name = "libclip_g.so"

    print(f"\n{'#'*60}\n  CLIP-G Pipeline\n{'#'*60}")

    if start_from <= 1:
        consolidate_external_data(src_onnx, consolidated)
    if start_from <= 2:
        qnn_convert_fp16(consolidated, qnn_out, component)
    if start_from <= 3:
        build_android_lib(qnn_out, lib_dir, lib_name)


def pipeline_vae(start_from: int):
    """VAE: InstanceNorm→GroupNorm, QNN FP16, Android lib."""
    component = "vae"
    src_onnx = ONNX_SRC / "vae_decoder.onnx"
    prep_dir = SDXL_NPU / "onnx_vae_prepared"
    prepared = prep_dir / "vae_decoder.onnx"
    qnn_out = SDXL_NPU / "qnn_vae_fp16"
    lib_dir = SDXL_NPU / "qnn_vae_fp16_android"
    lib_name = "libvae_decoder.so"

    print(f"\n{'#'*60}\n  VAE Decoder Pipeline\n{'#'*60}")

    if start_from <= 1:
        rewrite_vae_instancenorm(src_onnx, prepared)
    if start_from <= 2:
        qnn_convert_fp16(prepared, qnn_out, component)
    if start_from <= 3:
        build_android_lib(qnn_out, lib_dir, lib_name)


def main():
    ap = argparse.ArgumentParser(description="CLIP/VAE → QNN pipeline")
    ap.add_argument("--component", choices=["clip_l", "clip_g", "vae", "all"], default="all")
    ap.add_argument("--start-from", type=int, default=1, choices=[1, 2, 3],
                    help="1=preprocess, 2=QNN convert, 3=Android lib")
    args = ap.parse_args()

    components = {
        "clip_l": pipeline_clip_l,
        "clip_g": pipeline_clip_g,
        "vae": pipeline_vae,
    }

    if args.component == "all":
        for name, fn in components.items():
            fn(args.start_from)
    else:
        components[args.component](args.start_from)

    print(f"\n{'='*60}\n  All done!\n{'='*60}")


if __name__ == "__main__":
    main()
