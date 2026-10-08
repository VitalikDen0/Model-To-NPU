#!/usr/bin/env python3
"""
Компиляция WAN 2.1 через AI Hub для Snapdragon NPU.

AI Hub лучше справляется со сложными моделями, чем локальный QAIRT SDK.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_ONNX_PATH = Path(r"D:\platform-tools\wan21_13b_work\onnx\wan_t2v_1p3b_832x480_17f_seq128\transformer\model.onnx")
DEFAULT_OUTPUT_DIR = Path(r"D:\platform-tools\wan21_13b_work\qnn_aihub")


def _configure_aihub_token() -> None:
    """Propagate an already-provided token into the legacy AIHUB_TOKEN env name.

    The preferred setup remains the user's configured `~/.qai_hub/client.ini`
    or an environment variable set outside the repository.
    """
    for key in ("AIHUB_TOKEN", "QAI_HUB_API_TOKEN", "QAI_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            os.environ["AIHUB_TOKEN"] = value
            return


def _total_model_size_bytes(onnx_path: Path) -> int:
    total = onnx_path.stat().st_size
    external_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if external_data.exists():
        total += external_data.stat().st_size
    return total


def prepare_for_aihub(onnx_path: Path, output_dir: Path) -> dict | None:
    """Подготавливает ONNX модель для загрузки в AI Hub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем размер модели
    size_mb = _total_model_size_bytes(onnx_path) / 1e6
    size_gb = size_mb / 1024
    
    print(f"ONNX model: {onnx_path}")
    print(f"Size: {size_gb:.2f} GB ({size_mb:.1f} MB)")
    
    if size_gb > 2.0:
        print("\n⚠️  WARNING: Model is very large (>2 GB)")
        print("AI Hub may reject it or take very long to compile")
        print("Consider using smaller resolution or fewer frames")
        return None
    
    # Сохраняем метаданные
    metadata = {
        "model_path": str(onnx_path),
        "model_name": "wan_t2v_1p3b_transformer",
        "size_mb": size_mb,
        "target_device": "Snapdragon 8 Elite",
        "compile_options": {
            "target_runtime": "qnn_context_binary",
            "quantization": "fp16",
        }
    }
    
    metadata_path = output_dir / "aihub_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved: {metadata_path}")
    print("\nNext steps:")
    print("1. Upload ONNX to AI Hub manually (model too large for CLI)")
    print("2. Use AI Hub web interface to compile for Snapdragon 8 Elite")
    print("3. Download compiled context binary")
    
    return metadata


def compile_via_aihub_cli(onnx_path: Path, output_dir: Path) -> bool:
    """Пытается скомпилировать через AI Hub CLI (может не сработать для больших моделей)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Attempting to compile via AI Hub CLI...")
    print("Note: This may fail for large models (>2 GB)")
    
    # Устанавливаем токен только из внешней конфигурации/окружения.
    _configure_aihub_token()
    
    # Команда компиляции
    cmd = [
        "python", "-m", "qai_hub",
        "compile",
        "--model-path", str(onnx_path),
        "--device", "Snapdragon 8 Elite",
        "--output-dir", str(output_dir),
        "--target-runtime", "qnn",
        "--quantization", "fp16",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Compilation successful!")
            print(f"Output: {output_dir}")
            return True
        else:
            print(f"\n❌ Compilation failed with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # Сохраняем лог ошибки
            error_log = output_dir / "aihub_error.log"
            with open(error_log, "w") as f:
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            print(f"Error log saved: {error_log}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception during compilation: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile WAN 2.1 via AI Hub")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH, help="ONNX model path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--prepare", action="store_true", help="Prepare metadata for manual upload")
    parser.add_argument("--compile", action="store_true", help="Try to compile via CLI (may fail for large models)")
    
    args = parser.parse_args()
    
    if not args.onnx_path.exists():
        print(f"Error: ONNX model not found: {args.onnx_path}")
        print("Run export_wan_to_onnx.py first")
        sys.exit(1)
    
    if args.prepare or (not args.compile):
        metadata = prepare_for_aihub(args.onnx_path, args.output_dir)
        if metadata is None:
            print("\n⚠️  Model too large for AI Hub")
            print("WAN 2.1 1.3B may be too complex for current NPU compilation tools")
            sys.exit(1)
    
    if args.compile:
        success = compile_via_aihub_cli(args.onnx_path, args.output_dir)
        if not success:
            print("\n⚠️  AI Hub CLI compilation failed")
            print("Try manual upload via AI Hub web interface")
            sys.exit(1)


if __name__ == "__main__":
    main()
