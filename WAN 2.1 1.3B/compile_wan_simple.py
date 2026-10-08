#!/usr/bin/env python3
"""
Упрощенный скрипт компиляции WAN через QAIRT SDK с правильной настройкой окружения.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ONNX_PATH = Path(__file__).parent / "output" / "wan_t2v_1p3b_832x480_17f_seq128" / "transformer" / "model.onnx"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output" / "qnn_local"
DEFAULT_QAIRT_ROOT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130")


def compile_wan_simple(
    onnx_path: Path,
    output_dir: Path,
    qairt_root: Path,
) -> None:
    """Компилирует WAN ONNX через QAIRT SDK с правильной настройкой окружения."""
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = onnx_path.stem
    output_path = output_dir / model_name
    
    # Команда компиляции через PowerShell с envsetup
    envsetup_script = qairt_root / "bin" / "envsetup.ps1"
    converter_script = qairt_root / "bin" / "x86_64-windows-msvc" / "qnn-onnx-converter"
    
    # Формируем команду для PowerShell (без backend_config, он не поддерживается в converter)
    # Устанавливаем TMPDIR для Windows
    ps_command = f"""
$env:TMPDIR = '{output_dir}';
$env:TEMP = '{output_dir}';
$env:TMP = '{output_dir}';
. '{envsetup_script}';
python '{converter_script}' `
    --input_network '{onnx_path}' `
    --output_path '{output_path}' `
    --input_layout hidden_states NONTRIVIAL `
    --input_layout temb NONTRIVIAL `
    --input_layout timestep_proj NONTRIVIAL `
    --input_layout encoder_hidden_states NONTRIVIAL `
    --float_bitwidth 16
"""
    
    print(f"Compiling WAN ONNX to QNN context binary...")
    print(f"ONNX: {onnx_path}")
    print(f"Output: {output_path}")
    print(f"\nThis may take 10-30 minutes for large models...\n")
    
    # Запускаем через PowerShell
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_command],
        capture_output=True,
        text=True,
    )
    
    print("=== STDOUT ===")
    print(result.stdout)
    
    if result.stderr:
        print("\n=== STDERR ===")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ Compilation failed with exit code {result.returncode}")
        
        # Сохраняем лог
        log_path = output_dir / "compile_error.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Command:\n{ps_command}\n\n")
            f.write(f"STDOUT:\n{result.stdout}\n\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        print(f"Error log saved to: {log_path}")
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
    parser = argparse.ArgumentParser(description="Compile WAN ONNX with QAIRT SDK (simplified)")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help="Input ONNX model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT, help="QAIRT SDK root")
    
    args = parser.parse_args()
    
    compile_wan_simple(args.onnx, args.output, args.qairt_root)


if __name__ == "__main__":
    main()
