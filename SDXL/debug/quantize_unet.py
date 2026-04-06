#!/usr/bin/env python3
"""Quantize SDXL UNet ONNX models to INT8 (gentle QDQ approach).

Sensitive layers (conv_in, conv_out, first/last blocks) are excluded
from quantization to preserve image quality.

Usage (WSL):
  python3 quantize_unet.py \
    --onnx-dir /mnt/d/platform-tools/sdxl_npu/onnx_export \
    --calibration-dir /mnt/d/platform-tools/sdxl_npu/calibration_data \
    --output-dir /mnt/d/platform-tools/sdxl_npu/onnx_quantized \
    --resolution 1024x1024,832x1216,1216x832
"""
import argparse, gc, os, sys, time
import numpy as np


def detect_model_dtype(model_path):
    """Check if model inputs are fp16 or fp32."""
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    for inp in model.graph.input:
        dt = inp.type.tensor_type.elem_type
        if dt == onnx.TensorProto.FLOAT16:
            del model
            return "float16"
    del model
    return "float32"


def get_sensitive_nodes(model_path):
    """Identify nodes to exclude from quantization for quality."""
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    exclude = []
    for node in model.graph.node:
        name = node.name.lower()
        # Exclude boundary convolutions — most sensitive to quantization
        if any(k in name for k in [
            "conv_in", "conv_out", "conv_norm_out",
            "time_embed", "add_embedding",
        ]):
            exclude.append(node.name)
    n_total = len(model.graph.node)
    del model
    gc.collect()
    return exclude, n_total


def maybe_promote_groupnorm_model_to_opset21(model):
    """Promote default opset to 21 when GroupNormalization uses per-channel affine params.

    The ONNX GroupNormalization contract changed between opset 18 and 21.
    Our rewritten UNet uses per-channel gamma/beta tensors (length == C), which
    is semantically opset-21 style. If such nodes remain under opset 18, ORT can
    load the model but fail at runtime with internal broadcast errors like
    `32 by 320`.
    """
    import onnx
    from onnx import numpy_helper

    default_opset = None
    for opset in model.opset_import:
        if opset.domain == "":
            default_opset = opset
            break
    if default_opset is None:
        default_opset = model.opset_import.add()
        default_opset.domain = ""
        default_opset.version = 21
        return 0
    if default_opset.version >= 21:
        return 0

    initializer_map = {init.name: init for init in model.graph.initializer}
    promote_count = 0
    for node in model.graph.node:
        if node.op_type != "GroupNormalization" or len(node.input) < 3:
            continue

        num_groups = None
        for attr in node.attribute:
            if attr.name == "num_groups":
                num_groups = int(attr.i)
                break
        if num_groups is None:
            continue

        scale_init = initializer_map.get(node.input[1])
        bias_init = initializer_map.get(node.input[2])
        if scale_init is None or bias_init is None:
            continue

        scale = numpy_helper.to_array(scale_init).reshape(-1)
        bias = numpy_helper.to_array(bias_init).reshape(-1)
        if scale.ndim != 1 or bias.ndim != 1 or scale.shape != bias.shape:
            continue

        if int(scale.shape[0]) != num_groups:
            promote_count += 1

    if promote_count:
        default_opset.version = 21
    return promote_count


class UNetCalibrationReader:
    """CalibrationDataReader for UNet INT8 quantization."""

    def __init__(self, npz_path, dtype="float32", max_samples=200):
        # mmap_mode reduces peak RAM and avoids loading full calibration tensors into memory
        data = np.load(npz_path, mmap_mode="r")
        n = min(data["sample"].shape[0], max_samples)
        self.cast = np.float16 if dtype == "float16" else np.float32

        self.sample = data["sample"]
        self.timestep = data["timestep"]
        self.hs = data["encoder_hidden_states"]
        self.te = data["text_embeds"]
        self.ti = data["time_ids"]
        self.n = n
        self.idx = 0
        print(f"  CalibReader: {n} samples loaded ({dtype})", flush=True)

    def get_next(self):
        if self.idx >= self.n:
            return None
        i = self.idx
        self.idx += 1
        return {
            "sample": np.asarray(self.sample[i:i+1], dtype=self.cast),
            "timestep": np.asarray(self.timestep[i:i+1], dtype=self.cast),
            "encoder_hidden_states": np.asarray(self.hs[i:i+1], dtype=self.cast),
            "text_embeds": np.asarray(self.te[i:i+1], dtype=self.cast),
            "time_ids": np.asarray(self.ti[i:i+1], dtype=self.cast),
        }

    def rewind(self):
        self.idx = 0


def convert_fp16_model_to_fp32(input_path, output_path):
    """Convert fp16 ONNX model to fp32 for quantization compatibility."""
    import onnx
    from onnx import numpy_helper, TensorProto

    def _convert_tensor_proto_to_fp32(tensor_proto):
        if tensor_proto.data_type != TensorProto.FLOAT16:
            return tensor_proto
        arr = numpy_helper.to_array(tensor_proto).astype(np.float32)
        return numpy_helper.from_array(arr, tensor_proto.name)

    print("  Converting fp16 → fp32...", flush=True)
    model = onnx.load(input_path)

    # Convert initializers
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(_convert_tensor_proto_to_fp32(init))

    # Convert tensor attributes (e.g. Constant nodes carrying fp16 literals)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                attr.t.CopyFrom(_convert_tensor_proto_to_fp32(attr.t))
            elif attr.type == onnx.AttributeProto.TENSORS:
                for i, tensor in enumerate(attr.tensors):
                    if tensor.data_type == TensorProto.FLOAT16:
                        attr.tensors[i].CopyFrom(_convert_tensor_proto_to_fp32(tensor))

    # Convert graph inputs
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT

    # Convert graph outputs
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = TensorProto.FLOAT

    # Convert value_info
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    # Fix Cast nodes targeting fp16
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT

    promoted_groupnorm_nodes = maybe_promote_groupnorm_model_to_opset21(model)
    if promoted_groupnorm_nodes:
        print(f"  Promoted default opset to 21 for {promoted_groupnorm_nodes} GroupNormalization node(s)", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save_model(
        model, output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path).replace(".onnx", "_weights.bin"),
        size_threshold=1024,
    )
    del model
    gc.collect()
    print(f"  Saved fp32 model → {output_path}", flush=True)


def prepare_model_for_ort_quantization(input_path, output_path):
    """Normalize ONNX model for ORT quantization/runtime.

    - converts fp16 tensors / metadata to fp32 when present
    - promotes default opset to 21 when GroupNormalization nodes use per-channel affine params

    Returns True when a normalized copy was written, False when the original model can be used as-is.
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    def _convert_tensor_proto_to_fp32(tensor_proto):
        if tensor_proto.data_type != TensorProto.FLOAT16:
            return tensor_proto
        arr = numpy_helper.to_array(tensor_proto).astype(np.float32)
        return numpy_helper.from_array(arr, tensor_proto.name)

    model = onnx.load(input_path)
    changed = False

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(_convert_tensor_proto_to_fp32(init))
            changed = True

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                attr.t.CopyFrom(_convert_tensor_proto_to_fp32(attr.t))
                changed = True
            elif attr.type == onnx.AttributeProto.TENSORS:
                for i, tensor in enumerate(attr.tensors):
                    if tensor.data_type == TensorProto.FLOAT16:
                        attr.tensors[i].CopyFrom(_convert_tensor_proto_to_fp32(tensor))
                        changed = True

    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
                value.type.tensor_type.elem_type = TensorProto.FLOAT
                changed = True

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
                changed = True

    promoted_groupnorm_nodes = maybe_promote_groupnorm_model_to_opset21(model)
    if promoted_groupnorm_nodes:
        print(f"  Promoted default opset to 21 for {promoted_groupnorm_nodes} GroupNormalization node(s)", flush=True)
        changed = True

    if not changed:
        del model
        gc.collect()
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save_model(
        model, output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path).replace(".onnx", "_weights.bin"),
        size_threshold=1024,
    )
    del model
    gc.collect()
    print(f"  Saved ORT-normalized model → {output_path}", flush=True)
    return True


def quantize_one(
    onnx_dir,
    model_name,
    calib_npz,
    output_dir,
    max_samples=200,
    skip_fp16_convert=False,
    calibration_method="percentile",
    per_channel=True,
    quant_format="qdq",
):
    """Quantize a single UNet ONNX to INT8 with gentle settings."""
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader,
        QuantType, QuantFormat, CalibrationMethod,
    )

    input_model = os.path.join(onnx_dir, model_name)
    if not os.path.exists(input_model):
        print(f"  ERROR: model not found: {input_model}", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)
    out_name = model_name.replace(".onnx", "_int8.onnx")
    output_model = os.path.join(output_dir, out_name)

    if os.path.exists(output_model):
        print(f"  skip (exists): {output_model}", flush=True)
        return

    # Check model dtype
    dtype = detect_model_dtype(input_model)
    print(f"  Model dtype: {dtype}", flush=True)

    # If fp16 and not skipping conversion, convert to fp32
    actual_input = input_model
    tmp_fp32 = None
    if dtype == "float16" and not skip_fp16_convert:
        tmp_fp32 = os.path.join(output_dir, model_name.replace(".onnx", "_fp32_tmp.onnx"))
        convert_fp16_model_to_fp32(input_model, tmp_fp32)
        actual_input = tmp_fp32
        dtype = "float32"
    elif dtype == "float16" and skip_fp16_convert:
        print("  WARNING: model is fp16 but --skip-fp16-convert set", flush=True)

    # Identify sensitive nodes
    exclude, n_total = get_sensitive_nodes(actual_input)
    print(f"  Nodes: {n_total} total, excluding {len(exclude)} sensitive", flush=True)

    # Create calibration reader
    # Inherit from CalibrationDataReader for proper type
    class Reader(CalibrationDataReader):
        def __init__(self):
            self._inner = UNetCalibrationReader(calib_npz, dtype, max_samples)
        def get_next(self):
            return self._inner.get_next()
        def rewind(self):
            self._inner.rewind()

    reader = Reader()

    print(f"  Quantizing → {output_model}", flush=True)
    t0 = time.time()

    method_map = {
        "percentile": CalibrationMethod.Percentile,
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
    }
    format_map = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator,
    }
    method = method_map[calibration_method]
    qformat = format_map[quant_format]

    extra_options: dict[str, object] = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    if calibration_method == "percentile":
        extra_options["CalibPercentile"] = 99.99
        extra_options["CalibMovingAverage"] = True

    quantize_static(
        model_input=actual_input,
        model_output=output_model,
        calibration_data_reader=reader,
        quant_format=qformat,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=method,
        extra_options=extra_options,
        nodes_to_exclude=exclude,
        use_external_data_format=True,
    )

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s", flush=True)

    # Cleanup temp fp32 model
    if tmp_fp32 and os.path.exists(tmp_fp32):
        os.unlink(tmp_fp32)
        weights_bin = tmp_fp32.replace(".onnx", "_weights.bin")
        if os.path.exists(weights_bin):
            os.unlink(weights_bin)


def main():
    ap = argparse.ArgumentParser(description="Quantize SDXL UNet to INT8 (gentle)")
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--calibration-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resolution", default="1024x1024,832x1216,1216x832")
    ap.add_argument("--max-samples", type=int, default=200,
                    help="Max calibration samples to use")
    ap.add_argument("--skip-fp16-convert", action="store_true",
                    help="Model is already fp32, skip conversion")
    ap.add_argument("--calibration-method", default="percentile",
                    choices=["percentile", "minmax", "entropy"],
                    help="Calibration method (minmax is more memory-friendly)")
    ap.add_argument("--no-per-channel", action="store_true",
                    help="Disable per-channel weight quantization to reduce RAM usage")
    ap.add_argument("--quant-format", default="qdq", choices=["qdq", "qoperator"],
                    help="Quantization format. qoperator may reduce RAM on some ORT builds")
    a = ap.parse_args()

    for r in a.resolution.split(","):
        ww, hh = r.strip().split("x")
        model_name = f"unet_{ww}x{hh}.onnx"
        calib = os.path.join(a.calibration_dir, f"{ww}x{hh}", "calibration.npz")
        out = os.path.join(a.output_dir, f"{ww}x{hh}")

        print(f"\n=== Quantizing {model_name} ===", flush=True)

        if not os.path.exists(calib):
            print(f"  ERROR: calibration data not found: {calib}", flush=True)
            continue

        quantize_one(
            a.onnx_dir,
            model_name,
            calib,
            out,
            a.max_samples,
            a.skip_fp16_convert,
            a.calibration_method,
            not a.no_per_channel,
            a.quant_format,
        )
        gc.collect()

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
