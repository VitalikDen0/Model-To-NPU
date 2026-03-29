from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data


def get_attr_float(node: onnx.NodeProto, name: str, default: float) -> float:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return default


def load_constant_array(node: onnx.NodeProto) -> np.ndarray | None:
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
    return None


def build_maps(model: onnx.ModelProto):
    producer = {}
    consumers = {}
    node_by_name = {}
    for node in model.graph.node:
        node_by_name[node.name] = node
        for out in node.output:
            if out:
                producer[out] = node
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return producer, consumers, node_by_name


def prune_unreachable_nodes(model: onnx.ModelProto) -> int:
    producer, _, _ = build_maps(model)
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


def ensure_default_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == "":
            if opset.version < version:
                opset.version = version
            return
    model.opset_import.append(helper.make_opsetid("", version))


def rewrite(model: onnx.ModelProto, target_opset: int) -> tuple[onnx.ModelProto, int]:
    producer, consumers, _ = build_maps(model)
    initializer_map = {init.name: init for init in model.graph.initializer}
    const_values = {}
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            arr = load_constant_array(node)
            if arr is not None:
                const_values[node.output[0]] = arr

    replacements = {}
    skip_names = set()
    new_initializers = []
    rewrite_count = 0

    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            continue

        pre = producer.get(node.input[0])
        if pre is None or pre.op_type != "Reshape" or len(pre.input) < 2:
            continue
        reshape_const = const_values.get(pre.input[1])
        if reshape_const is None:
            continue
        reshape_vals = tuple(int(x) for x in reshape_const.reshape(-1).tolist())
        if len(reshape_vals) != 3 or reshape_vals[0] != 0 or reshape_vals[2] != -1 or reshape_vals[1] <= 0:
            continue
        group_count = int(reshape_vals[1])

        inst_scale = const_values.get(node.input[1])
        inst_bias = const_values.get(node.input[2])
        if inst_scale is None or inst_bias is None:
            continue
        if not np.allclose(inst_scale, 1.0) or not np.allclose(inst_bias, 0.0):
            continue

        post_nodes = consumers.get(node.output[0], [])
        if len(post_nodes) != 1 or post_nodes[0].op_type != "Reshape":
            continue
        post = post_nodes[0]

        mul_nodes = consumers.get(post.output[0], [])
        if len(mul_nodes) != 1 or mul_nodes[0].op_type != "Mul":
            continue
        mul = mul_nodes[0]

        add_nodes = consumers.get(mul.output[0], [])
        if len(add_nodes) != 1 or add_nodes[0].op_type != "Add":
            continue
        add = add_nodes[0]

        mul_const_name = mul.input[1] if mul.input[0] == post.output[0] else mul.input[0]
        add_const_name = add.input[1] if add.input[0] == mul.output[0] else add.input[0]
        if mul_const_name not in initializer_map or add_const_name not in initializer_map:
            continue

        gamma = numpy_helper.to_array(initializer_map[mul_const_name]).astype(np.float32).reshape(-1)
        beta = numpy_helper.to_array(initializer_map[add_const_name]).astype(np.float32).reshape(-1)
        if gamma.shape != beta.shape or gamma.ndim != 1:
            continue
        num_channels = int(gamma.shape[0])
        if num_channels % group_count != 0:
            continue

        epsilon = get_attr_float(node, "epsilon", 1e-5)
        gn_weight_name = f"{add.name}_groupnorm_weight"
        gn_bias_name = f"{add.name}_groupnorm_bias"
        new_initializers.append(numpy_helper.from_array(gamma, name=gn_weight_name))
        new_initializers.append(numpy_helper.from_array(beta, name=gn_bias_name))

        gn_name = f"{add.name}_GroupNormalization"
        gn_node = helper.make_node(
            "GroupNormalization",
            inputs=[pre.input[0], gn_weight_name, gn_bias_name],
            outputs=list(add.output),
            num_groups=group_count,
            epsilon=epsilon,
            name=gn_name,
        )
        replacements[add.name] = gn_node
        skip_names.update({pre.name, node.name, post.name, mul.name})
        rewrite_count += 1

    if rewrite_count == 0:
        return model, 0

    rewritten_nodes = []
    for node in model.graph.node:
        if node.name in skip_names:
            continue
        replacement = replacements.get(node.name)
        if replacement is not None:
            rewritten_nodes.append(replacement)
        else:
            rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    ensure_default_opset(model, target_opset)
    return model, rewrite_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite SDXL GroupNorm-like InstanceNormalization patterns to direct ONNX GroupNormalization")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-opset", type=int, default=21)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_path), load_external_data=True)
    model, count = rewrite(model, target_opset=args.target_opset)
    if count == 0:
        raise SystemExit("No rewrite candidates found")

    pruned = prune_unreachable_nodes(model)

    data_name = out_path.with_suffix(out_path.suffix + ".data").name
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    onnx.save_model(model, str(out_path))
    print(f"[ok] rewrote {count} GroupNorm-like patterns -> {out_path}")
    print(f"[ok] pruned unreachable nodes: {pruned}")


if __name__ == "__main__":
    main()
