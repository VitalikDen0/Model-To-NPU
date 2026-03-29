#!/usr/bin/env python3
"""Rewrite ONNX Reshape nodes to use static constant target shapes.

Why:
- QAIRT 2.44 fails on several exported Diffusers/UNet reshape patterns, including both
    `... -> Shape(original) -> Reshape(...)` and constant-shape forms that use `0` / `-1`
    placeholders.
- Replacing every Reshape target shape with the exact inferred static output shape preserves
    math for fixed-shape NPU exports and removes converter-side shape inference blockers.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def build_maps(model: onnx.ModelProto):
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    return producer


def prune_unreachable_nodes(model: onnx.ModelProto) -> int:
    producer = build_maps(model)
    required_values = {output.name for output in model.graph.output}
    required_nodes = set()

    changed = True
    while changed:
        changed = False
        for value_name in list(required_values):
            node = producer.get(value_name)
            if node is None or node.name in required_nodes:
                continue
            required_nodes.add(node.name)
            required_values.update(inp for inp in node.input if inp)
            changed = True

    kept_nodes = [node for node in model.graph.node if node.name in required_nodes]
    pruned = len(model.graph.node) - len(kept_nodes)
    if pruned > 0:
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
    return pruned


def _collect_static_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True, verbose=0)
    static_shapes: dict[str, list[int]] = {}

    for value_info in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        dims: list[int] = []
        all_static = True
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            else:
                all_static = False
                break

        if all_static:
            static_shapes[value_info.name] = dims

    return static_shapes


def rewrite_reshape_nodes_to_static_shapes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    static_shapes = _collect_static_shapes(model)

    rewritten = 0
    new_nodes = []

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        static_shape = static_shapes.get(node.output[0])
        if static_shape is None:
            new_nodes.append(node)
            continue

        const_name = f"{node.name}__static_shape"
        const_output = f"{const_name}_output_0"
        shape_tensor = numpy_helper.from_array(np.asarray(static_shape, dtype=np.int64), name=const_output)
        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[const_output],
            name=const_name,
            value=shape_tensor,
        )

        rewritten_node = copy.deepcopy(node)
        rewritten_node.input[1] = const_output

        new_nodes.append(const_node)
        new_nodes.append(rewritten_node)
        rewritten += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite ONNX Reshape nodes to static constants")
    ap.add_argument("--input", required=True, help="Input ONNX model path")
    ap.add_argument("--output", required=True, help="Output ONNX model path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_path), load_external_data=True)
    model, rewritten = rewrite_reshape_nodes_to_static_shapes(model)
    pruned = prune_unreachable_nodes(model)

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_path.name + ".data",
        size_threshold=1024,
    )

    print(f"[ok] rewritten Reshape nodes to static targets: {rewritten}")
    print(f"[ok] pruned unreachable nodes: {pruned}")
    print(f"[ok] model: {out_path}")
    print(f"[ok] external data: {out_path}.data")


if __name__ == "__main__":
    main()
