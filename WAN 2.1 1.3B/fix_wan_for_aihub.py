#!/usr/bin/env python3
"""
Исправление WAN ONNX модели для успешной компиляции через AI Hub.

Проблемы, которые могут вызывать exit code 15:
1. Слишком большая модель (8.3 GB external data)
2. Неоптимальные паттерны операций
3. Проблемы с dynamic shapes
"""
from __future__ import annotations

import argparse
from pathlib import Path
import onnx
from onnx import shape_inference
import sys


def optimize_for_aihub(input_path: Path, output_path: Path) -> None:
    """Оптимизирует ONNX модель для AI Hub."""
    print(f"Loading model from {input_path}")
    model = onnx.load(str(input_path), load_external_data=True)
    
    print(f"Original model:")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Opset: {model.opset_import[0].version}")
    
    # Проверяем, что все shapes статические
    print("\nChecking input shapes:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.HasField('dim_value') else 'dynamic' 
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
        if 'dynamic' in shape:
            print(f"    WARNING: Dynamic shape detected!")
    
    # Shape inference
    print("\nRunning shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
        print("  Shape inference successful")
    except Exception as e:
        print(f"  Shape inference failed: {e}")
        print("  Continuing without shape inference...")
    
    # Сохраняем с правильными настройками для AI Hub
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_suffix(output_path.suffix + '.data')
    
    print(f"\nSaving optimized model to {output_path}")
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,  # AI Hub требует inline для маленьких тензоров
        convert_attribute=False,
    )
    
    print(f"Saved:")
    print(f"  Model: {output_path}")
    print(f"  External data: {external_data_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB + {external_data_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize WAN ONNX for AI Hub")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"D:\platform-tools\wan21_13b_work\onnx\wan_t2v_1p3b_832x480_17f_seq128\transformer\model.onnx"),
        help="Input ONNX model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\platform-tools\wan21_13b_work\onnx_optimized\wan_t2v_1p3b_832x480_17f_seq128\transformer\model.onnx"),
        help="Output ONNX model"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    optimize_for_aihub(args.input, args.output)
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
