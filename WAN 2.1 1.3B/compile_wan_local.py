#!/usr/bin/env python3
"""
Локальная компиляция WAN transformer через QAIRT SDK.

Этот скрипт компилирует ONNX модель в QNN context binary локально,
минуя AI Hub, что дает больше контроля и детальные логи.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ONNX_PATH = Path(r"D:\platform-tools\wan21_13b_work\onnx\wan_t2v_1p3b_832x480_17f_seq128\transformer\model.onnx")
DEFAULT_OUTPUT_DIR = Path(r"D:\platform-tools\wan21_13b_work\qnn_local")
DEFAULT_QAIRT_ROOT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130")


def find_qnn_converter(qairt_root: Path) -> Path:
    """Находит qnn-onnx-converter в QAIRT SDK."""
    converter_paths = [
        qairt_root / "bin" / "x86_64-windows-msvc" / "qnn-onnx-converter",
        qairt_root / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter",
        qairt_root / "bin" / "x86_64-windows-msvc" / "qnn-onnx-converter.exe",
        qairt_root / "bin" / "qnn-onnx-converter.exe",
    ]
    
    for path in converter_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"qnn-onnx-converter not found in {qairt_root}. "
        "Please install QAIRT SDK or specify correct path with --qairt-root"
    )


def compile_local(
    onnx_path: Path,
    output_dir: Path,
    qairt_root: Path,
    *,
    precision: str = "fp16",
    device: str = "htp",
) -> None:
    """Компилирует ONNX в QNN context binary локально."""
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    converter = find_qnn_converter(qairt_root)
    print(f"Using QNN converter: {converter}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = onnx_path.stem
    output_path = output_dir / model_name
    
    # Базовые параметры конвертации
    cmd = [
        "python",
        str(converter),
        "--input_network", str(onnx_path),
        "--output_path", str(output_path),
        "--input_layout", "hidden_states", "NONTRIVIAL",
        "--input_layout", "temb", "NONTRIVIAL",
        "--input_layout", "timestep_proj", "NONTRIVIAL",
        "--input_layout", "encoder_hidden_states", "NONTRIVIAL",
    ]
    
    # Precision
    if precision == "fp16":
        cmd.extend(["--float_bitwidth", "16"])
    
    # Backend config для HTP
    if device == "htp":
        backend_config = {
            "backend": "htp",
            "vtcm_mb": 8,
            "hvx_threads": 2,
            "perf_profile": "burst",
            "precision": "fp16",
        }
        config_path = output_dir / "htp_backend_config.json"
        with open(config_path, "w") as f:
            json.dump(backend_config, f, indent=2)
        
        cmd.extend([
            "--backend_config", str(config_path),
        ])
    
    print(f"\nCompiling ONNX to QNN context binary...")
    print(f"Command: {' '.join(cmd)}")
    print(f"\nThis may take 10-30 minutes for large models...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\n=== STDOUT ===")
    print(result.stdout)
    
    if result.stderr:
        print("\n=== STDERR ===")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ Compilation failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Проверяем созданные файлы
    cpp_file = Path(str(output_path) + ".cpp")
    bin_file = Path(str(output_path) + ".bin")
    
    if cpp_file.exists() and bin_file.exists():
        print(f"\n✅ Compilation successful!")
        print(f"  C++ file: {cpp_file} ({cpp_file.stat().st_size / 1e6:.1f} MB)")
        print(f"  Binary: {bin_file} ({bin_file.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"\n⚠️ Compilation completed but expected files not found")
        print(f"  Looking for: {cpp_file}, {bin_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile WAN ONNX locally with QAIRT SDK")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help="Input ONNX model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT, help="QAIRT SDK root")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="Model precision")
    parser.add_argument("--device", choices=["htp", "cpu"], default="htp", help="Target device")
    
    args = parser.parse_args()
    
    compile_local(
        args.onnx,
        args.output,
        args.qairt_root,
        precision=args.precision,
        device=args.device,
    )


if __name__ == "__main__":
    main()
