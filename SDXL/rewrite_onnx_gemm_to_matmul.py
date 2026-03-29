#!/usr/bin/env python3
"""Rewrite ONNX Gemm nodes into MatMul (+ Add) for QAIRT/QNN compatibility.

Why:
- QAIRT 2.44 reproducibly fails even on tiny valid Gemm models with a permute error.
- Equivalent MatMul (+ Add) graphs convert successfully.

This utility rewrites every Gemm node while preserving math for the common linear-layer case:
  Y = A * B (+ C)
with support for transB=1 by materializing a transposed weight initializer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def rewrite_gemm_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    initializer_map = {init.name: init for init in model.graph.initializer}
    rewritten = 0
    new_nodes = []

    for node in model.graph.node:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        trans_a = attrs.get("transA", 0)
        trans_b = attrs.get("transB", 0)

        if alpha != 1.0:
            raise ValueError(f"Unsupported Gemm alpha != 1.0 at node {node.name}")
        if beta != 1.0 and len(node.input) >= 3 and node.input[2]:
            raise ValueError(f"Unsupported Gemm beta != 1.0 at node {node.name}")
        if trans_a not in (0, 1):
            raise ValueError(f"Unsupported Gemm transA={trans_a} at node {node.name}")
        if trans_b not in (0, 1):
            raise ValueError(f"Unsupported Gemm transB={trans_b} at node {node.name}")

        input_a = node.input[0]
        input_b = node.input[1]
        input_c = node.input[2] if len(node.input) >= 3 and node.input[2] else None
        current_a = input_a
        current_b = input_b

        if trans_a == 1:
            transpose_a_out = f"{node.name}_transpose_a_out"
            new_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[current_a],
                    outputs=[transpose_a_out],
                    name=f"{node.name}_TransposeA",
                    perm=[1, 0],
                )
            )
            current_a = transpose_a_out

        if trans_b == 1:
            if current_b in initializer_map:
                weight_arr = numpy_helper.to_array(initializer_map[current_b])
                transposed_name = f"{current_b}__gemm_transposed"
                if transposed_name not in initializer_map:
                    transposed_init = numpy_helper.from_array(weight_arr.T.copy(), transposed_name)
                    model.graph.initializer.append(transposed_init)
                    initializer_map[transposed_name] = transposed_init
                current_b = transposed_name
            else:
                transpose_b_out = f"{node.name}_transpose_b_out"
                new_nodes.append(
                    helper.make_node(
                        "Transpose",
                        inputs=[current_b],
                        outputs=[transpose_b_out],
                        name=f"{node.name}_TransposeB",
                        perm=[1, 0],
                    )
                )
                current_b = transpose_b_out

        matmul_out = node.output[0] if not input_c else f"{node.output[0]}__matmul"
        new_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[current_a, current_b],
                outputs=[matmul_out],
                name=f"{node.name}_MatMul",
            )
        )

        if input_c:
            new_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[matmul_out, input_c],
                    outputs=list(node.output),
                    name=f"{node.name}_Add",
                )
            )

        rewritten += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite ONNX Gemm nodes into MatMul (+ Add)")
    ap.add_argument("--input", required=True, help="Input ONNX model path")
    ap.add_argument("--output", required=True, help="Output ONNX model path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_path), load_external_data=True)
    model, rewritten = rewrite_gemm_nodes(model)

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_path.name + ".data",
        size_threshold=1024,
    )

    print(f"[ok] rewritten Gemm nodes: {rewritten}")
    print(f"[ok] model: {out_path}")
    print(f"[ok] external data: {out_path}.data")


if __name__ == "__main__":
    main()
