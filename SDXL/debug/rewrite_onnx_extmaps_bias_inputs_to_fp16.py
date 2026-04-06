from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data


FP16_CAST_TO = TensorProto.FLOAT16


def build_consumers(model: onnx.ModelProto):
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def replace_input_name(model: onnx.ModelProto, old_name: str, new_name: str) -> int:
    replaced = 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name
                replaced += 1
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite extmaps resnet bias inputs to fp16 and remove trivial Cast nodes")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_path), load_external_data=True)
    bias_inputs = {vi.name for vi in model.graph.input if vi.name.startswith("resnet_bias_")}

    for value_info in model.graph.input:
        if value_info.name in bias_inputs:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    consumers = build_consumers(model)
    kept_nodes = []
    removed_casts = 0
    rewired_edges = 0

    for node in model.graph.node:
        if (
            node.op_type == "Cast"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.input[0] in bias_inputs
        ):
            cast_to = None
            for attr in node.attribute:
                if attr.name == "to":
                    cast_to = attr.i
                    break
            if cast_to == FP16_CAST_TO:
                rewired_edges += replace_input_name(model, node.output[0], node.input[0])
                removed_casts += 1
                continue
        kept_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    data_name = out_path.with_suffix(out_path.suffix + ".data").name
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    onnx.save_model(model, str(out_path))
    print(f"[ok] bias inputs converted to fp16: {len(bias_inputs)}")
    print(f"[ok] removed Cast nodes: {removed_casts}")
    print(f"[ok] rewired edges: {rewired_edges}")
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
