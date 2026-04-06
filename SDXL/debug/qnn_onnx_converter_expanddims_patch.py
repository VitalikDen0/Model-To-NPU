#!/usr/bin/env python3
from __future__ import annotations

import sys
import traceback
import types
import os

import numpy as np
import onnx

if not hasattr(onnx, "mapping"):
    import onnx._mapping as _onnx_mapping

    onnx.mapping = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE={
            key: value.np_dtype for key, value in _onnx_mapping.TENSOR_TYPE_MAP.items()
        }
    )

from qti.aisw.converters import onnx as onnx_frontend
from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
from qti.aisw.converters.backend.qnn_quantizer import QnnQuantizer
from qti.aisw.converters.common.arch_linter.arch_linter import ArchLinter
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
try:
    from qti.aisw.converters.common.graph_optimizer import GraphOptimizer
except ImportError:
    GraphOptimizer = None
from qti.aisw.converters.common.model_validator import Validator
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
from qti.aisw.converters.common.utils.converter_utils import log_error, log_info, log_warning
from qti.aisw.converters.onnx import data_translations as onnx_data_translations
from qti.aisw.converters.onnx.util import extract_attributes
from qti.aisw.converters.qnn_backend import qnn_definitions
from qti.aisw.converters.qnn_backend import qnn_translations
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory


TRACE_BACKEND_FILTER = os.environ.get("QAIRT_BACKEND_TRACE_FILTER", "").strip()


def _dtype_text(dtype) -> str:
    try:
        return hex(int(dtype))
    except Exception:
        return repr(dtype)


def _safe_tensor_info(backend, tensor_name: str):
    try:
        return backend.retrieve_tensor_info(tensor_name)
    except Exception:
        return None


_orig_backend_add_node = QnnConverterBackend.add_node
_orig_eltwise_add_op_to_backend = qnn_translations.QnnElementwiseTranslation.add_op_to_backend
_orig_groupnorm_add_op_to_backend = qnn_translations.QnnGroupNormTranslation.add_op_to_backend
_orig_reshape_add_op_to_backend = qnn_translations.QnnReshapeTranslation.add_op_to_backend
_orig_layernorm_add_op_to_backend = qnn_translations.QnnLayernormTranslation.add_op_to_backend
_orig_transpose_add_op_to_backend = qnn_translations.QnnTransposeTranslation.add_op_to_backend
_orig_stridedslice_add_op_to_backend = qnn_translations.QnnStridedSliceTranslation.add_op_to_backend
_orig_elementwiseneuron_add_op_to_backend = qnn_translations.QnnElementwiseNeuronTranslation.add_op_to_backend
_orig_matmul_add_op_to_backend = qnn_translations.QnnMatMulTranslation.add_op_to_backend
_orig_softmax_add_op_to_backend = qnn_translations.QnnSoftmaxTranslation.add_op_to_backend
_orig_convolution_add_op_to_backend = qnn_translations.QnnConvolutionTranslation.add_op_to_backend
_orig_concat_add_op_to_backend = qnn_translations.QnnConcatTranslation.add_op_to_backend
_orig_resize_add_op_to_backend = qnn_translations.QnnResizeTranslation.add_op_to_backend


def _traced_backend_add_node(self, node_name, node_type, input_names, outputs_info, tensor_params={}, scalar_params={},
                             macs=0, **kwargs):
    if TRACE_BACKEND_FILTER and TRACE_BACKEND_FILTER in (node_name or ""):
        print(f"[backend-trace] node={node_name} type={node_type}", file=sys.stderr, flush=True)
        for idx, input_name in enumerate(input_names):
            tensor_info = _safe_tensor_info(self, input_name)
            if tensor_info is None:
                print(f"[backend-trace]   input[{idx}] name={input_name} info=<missing>", file=sys.stderr, flush=True)
                continue
            print(
                f"[backend-trace]   input[{idx}] name={input_name} dtype={_dtype_text(tensor_info.get('data_type'))} "
                f"dims={tensor_info.get('dims')} axis={tensor_info.get('axis_format')} src_axis={tensor_info.get('src_axis_format')}",
                file=sys.stderr,
                flush=True,
            )
        for idx, output_info in enumerate(outputs_info):
            print(
                f"[backend-trace]   output[{idx}] name={output_info.get('name')} dtype={_dtype_text(output_info.get('data_type'))} "
                f"dims={output_info.get('dims')} axis={output_info.get('axis_format')} src_axis={output_info.get('src_axis_format')}",
                file=sys.stderr,
                flush=True,
            )
    return _orig_backend_add_node(self, node_name, node_type, input_names, outputs_info, tensor_params, scalar_params,
                                  macs)


def _qnn_dtype_name(dtype) -> str:
    if dtype == op_adapter.ir_graph.QNN_DATATYPE_FLOAT_16:
        return "float16"
    if dtype == op_adapter.ir_graph.QNN_DATATYPE_FLOAT_32:
        return "float32"
    raise ValueError(f"Unsupported dtype for cast patch: {dtype}")


def _tensor_numel_from_info(tensor_info) -> int | None:
    dims = tensor_info.get("dims")
    if dims is None:
        return None
    try:
        numel = 1
        for dim in dims:
            dim_value = int(dim)
            if dim_value < 0:
                return None
            numel *= dim_value
        return numel
    except Exception:
        return None


def _maybe_insert_float_mismatch_cast(node, graph, backend):
    if node.op.type not in qnn_translations.QnnElementwiseTranslation.BINARY_ELTWISE:
        return
    if len(node.input_names) != 2:
        return

    info0 = backend.retrieve_tensor_info(node.input_names[0])
    info1 = backend.retrieve_tensor_info(node.input_names[1])
    dtype0 = info0["data_type"]
    dtype1 = info1["data_type"]

    float_pair = {op_adapter.ir_graph.QNN_DATATYPE_FLOAT_16, op_adapter.ir_graph.QNN_DATATYPE_FLOAT_32}
    if {dtype0, dtype1} != float_pair:
        return

    output_dtype = graph.get_buffer(node.output_names[0]).dtype
    if output_dtype not in float_pair:
        numel0 = _tensor_numel_from_info(info0)
        numel1 = _tensor_numel_from_info(info1)
        if numel0 == 1 and numel1 != 1:
            output_dtype = dtype1
        elif numel1 == 1 and numel0 != 1:
            output_dtype = dtype0
        else:
            return

    for cast_input_idx, source_dtype in enumerate((dtype0, dtype1)):
        if source_dtype == output_dtype:
            continue
        cast_input_name = node.input_names[cast_input_idx]
        cast_op_name = cast_input_name + f"_cast_to_{_qnn_dtype_name(output_dtype)}"
        if not graph.has_buffer(cast_op_name):
            cast_op = op_adapter.CastOp(
                cast_op_name,
                from_type=_qnn_dtype_name(source_dtype),
                to_type=_qnn_dtype_name(output_dtype),
            )
            if not graph.has_buffer(cast_input_name):
                raise KeyError(f"Graph has no buffer {cast_input_name}, referred to as input for {node.op.name}")
            buf = graph.get_buffer(cast_input_name)
            idx_to_insert = graph.nodes_in_order.index(buf.producer) + 1
            cast_node = graph.add(cast_op, input_names=[cast_input_name], output_names=[cast_op_name], idx=idx_to_insert)
            buf.consumers.remove(node)
            cast_outputs_info = backend.get_outputs_info(
                cast_node,
                graph,
                tensor_data_type=output_dtype,
                check_encodings=True,
            )
            backend.add_node(cast_node.op.name, qnn_definitions.QNN_OP_CAST, cast_node.input_names, cast_outputs_info)
        if graph.has_buffer(cast_op_name):
            graph.get_buffer(cast_op_name).consumers.add(node)
        node.input_names[cast_input_idx] = cast_op_name


def _patched_eltwise_add_op_to_backend(self, node, graph, backend, **kwargs):
    _maybe_insert_float_mismatch_cast(node, graph, backend)

    try:
        input_dtypes = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]
        float_pair = {op_adapter.ir_graph.QNN_DATATYPE_FLOAT_16, op_adapter.ir_graph.QNN_DATATYPE_FLOAT_32}
        unique_input_dtypes = set(input_dtypes)
        if unique_input_dtypes and unique_input_dtypes.issubset(float_pair) and len(unique_input_dtypes) == 1:
            chosen_dtype = next(iter(unique_input_dtypes))
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = chosen_dtype
    except Exception:
        pass

    return _orig_eltwise_add_op_to_backend(self, node, graph, backend, **kwargs)


def _patched_groupnorm_add_op_to_backend(self, node, graph, backend, **kwargs):
    num_groups = np.uint32(node.op.group)
    num_channels = graph.get_buffer(node.input_names[0]).shape[-1]
    input_shapes = graph.get_input_shapes(node)
    if num_channels % num_groups != 0:
        raise ValueError("Node {}: number of groups must be a divisor of the number of channels {}. Got {}".format(
            node.op.name, num_channels, num_groups))

    if len(input_shapes) > 1 and input_shapes[1][0] != num_channels:
        raise ValueError("Node {}: Weight input shape must be equal to number of channels. Expected {} but got {} ".format(
            node.op.name, num_channels, input_shapes[1][0]))

    if len(input_shapes) > 2 and input_shapes[2][0] != num_channels:
        raise ValueError("Node {}: Bias input shape must be equal to number of channels. Expected {} but got {} ".format(
            node.op.name, num_channels, input_shapes[2][0]))

    scalar_params = {}
    scalar_params.update({qnn_translations.ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON:
                              (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))})
    scalar_params.update({qnn_translations.ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP:
                              (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], num_groups)})

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = graph.get_buffer(node.output_names[0]).dtype

    backend.add_node(
        node.op.name,
        node.op.type,
        input_names=node.input_names,
        outputs_info=backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params=scalar_params,
        macs=node.op.macs,
    )


def _patched_reshape_add_op_to_backend(self, node, graph, backend, **kwargs):
    tensor_params = {}
    if getattr(backend, 'serialize_with_suppl_attr', False):
        shape_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, qnn_translations.ir_graph.IR_OP_RESHAPE_PARAM_SHAPE)
        shape_tensor_info = backend.create_tensor_info(
            shape_tensor_name,
            qnn_definitions.QNN_TENSOR_TYPE_STATIC,
            [len(node.op.shape)],
            qnn_translations.ir_graph.QNN_DATATYPE_INT_32,
            data=node.op.shape,
        )
        tensor_params.update({
            qnn_translations.ir_graph.IR_OP_RESHAPE_PARAM_SHAPE:
                (shape_tensor_info, qnn_translations.ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })

    output_dtype = None
    if len(node.input_names) == 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_RESHAPE,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        tensor_params=tensor_params,
    )


def _patched_layernorm_add_op_to_backend(self, node, graph, backend, **kwargs):
    axes_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, qnn_definitions.QNN_OP_LAYER_NORM_PARAM_AXES)
    axes = node.op.axes
    axes_tensor_info = backend.create_tensor_info(
        axes_tensor_name,
        qnn_definitions.QNN_TENSOR_TYPE_STATIC,
        [len(axes)],
        qnn_translations.ir_graph.QNN_DATATYPE_UINT_32,
        data=np.asarray(axes, dtype=np.uint32),
    )

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_definitions.QNN_OP_LAYER_NORM,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        tensor_params={qnn_definitions.QNN_OP_LAYER_NORM_PARAM_AXES: axes_tensor_info},
        scalar_params={
            qnn_definitions.QNN_OP_LAYER_NORM_PARAM_EPSILON:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))
        },
        macs=node.op.macs,
    )


def _patched_transpose_add_op_to_backend(self, node, graph, backend, **kwargs):
    perm_order = np.asarray(node.op.perm, dtype=np.uint32)
    perm_order_name = backend.create_unique_qnn_tensor_name(node.op.name, qnn_translations.ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM)
    perm_order_info = backend.create_tensor_info(
        perm_order_name,
        qnn_definitions.QNN_TENSOR_TYPE_STATIC,
        [len(perm_order)],
        qnn_translations.ir_graph.QNN_DATATYPE_UINT_32,
        data=perm_order,
    )

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_TRANSPOSE,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        tensor_params={qnn_translations.ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM: perm_order_info},
    )


def _patched_stridedslice_add_op_to_backend(self, node, graph, backend, **kwargs):
    ranges_name = backend.create_unique_qnn_tensor_name(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES,
    )
    ranges_info = backend.create_tensor_info(
        ranges_name,
        qnn_definitions.QNN_TENSOR_TYPE_STATIC,
        list(node.op.ranges.shape),
        qnn_translations.ir_graph.QNN_DATATYPE_INT_32,
        data=node.op.ranges,
    )

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params={
            qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.begin_mask)),
            qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_END_MASK:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.end_mask)),
            qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.shrink_axes)),
            qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.new_axes_mask)),
        },
        tensor_params={qnn_translations.ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES: ranges_info},
    )


def _patched_elementwiseneuron_add_op_to_backend(self, node, graph, backend, **kwargs):
    neuron_scalar_params = {}
    operation = node.op.operation
    if operation == qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX:
        neuron_scalar_params = {
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.min_value)),
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.max_value)),
        }
    elif operation == qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU:
        neuron_scalar_params = {
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.alpha)),
        }
    elif operation == qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID:
        neuron_scalar_params = {
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.alpha)),
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta)),
        }
    elif operation == qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS:
        neuron_scalar_params = {
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta)),
            qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.threshold)),
        }
    neuron_scalar_params.update({
        qnn_translations.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION:
            (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(operation))
    })

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        node.op.type,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params=neuron_scalar_params,
    )


def _patched_matmul_add_op_to_backend(self, node, graph, backend, **kwargs):
    c_ir_graph = getattr(backend, 'c_ir_graph', None)
    if c_ir_graph is not None and c_ir_graph.get_output_tensor(node.input_names[1]) is not None:
        input_encoding = c_ir_graph.get_output_tensor(node.input_names[1]).get_encoding()
        input_encoding_enc_info = input_encoding.encInfo
        input_bw = input_encoding_enc_info.bw
        if input_bw != 8:
            input_name = node.input_names[1]
            convert_name = node.input_names[1] + "_converted_QNN_DATATYPE_UFIXED_POINT_8"
            convert_op = op_adapter.ConvertOp(convert_name, to_type=qnn_translations.ir_graph.QNN_DATATYPE_UFIXED_POINT_8)
            if graph.has_buffer(convert_name):
                convert_buffer = graph.buffers[convert_name]
                consumer = graph.nodes_by_name[node.op.name]
                convert_buffer.consumers.add(consumer)
                node.input_names[1] = convert_name
                input_buffer = graph.buffers[input_name]
                input_buffer.consumers.remove(consumer)
            else:
                consumers = []
                for consumer in graph.buffers[input_name].consumers:
                    if consumer.op.type == node.op.type and consumer.input_names[1] == input_name:
                        consumers.append(consumer.op.name)
                graph.inject(convert_op, input_name, convert_name, consumer_names=consumers)
                convert_node = graph.nodes_by_name[convert_name]
                producer_encoding = backend.get_producer_encoding(convert_node, graph)
                quant_params, producer_tensor_encoding = backend.get_qnn_quant_params(producer_encoding)

                quant_params['scale_offset']['scale'] = quant_params['scale_offset']['scale'] * 256.0
                quant_params['scale_offset']['offset'] = round(quant_params['scale_offset']['offset'] / 256.0)

                output_matmul_name = node.output_names[0]
                outputs_info = backend.get_outputs_info(
                    convert_node,
                    graph,
                    tensor_data_type=qnn_translations.ir_graph.QNN_DATATYPE_UFIXED_POINT_8,
                    original_output=output_matmul_name,
                )
                outputs_info[0]['data_type'] = qnn_translations.ir_graph.QNN_DATATYPE_UFIXED_POINT_8
                outputs_info[0]['quant_params'] = quant_params

                backend.add_node(
                    convert_node.op.name,
                    qnn_definitions.QNN_OP_CONVERT,
                    convert_node.input_names,
                    outputs_info,
                    scalar_params={
                        qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA:
                            (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(convert_node.op.dynamic_input_data)),
                        qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA:
                            (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(convert_node.op.dynamic_output_data)),
                    },
                )

    output_dtype = None
    try:
        input_dtypes = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names[:2]]
        float_pair = {qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_16, qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_32}
        unique_input_dtypes = set(input_dtypes)
        if unique_input_dtypes and unique_input_dtypes.issubset(float_pair) and len(unique_input_dtypes) == 1:
            output_dtype = next(iter(unique_input_dtypes))
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = output_dtype
    except Exception:
        output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_MAT_MUL,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params={
            qnn_translations.ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], node.op.transpose_in0),
            qnn_translations.ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], node.op.transpose_in1),
        },
        macs=node.op.macs,
    )


def _patched_softmax_add_op_to_backend(self, node, graph, backend, **kwargs):
    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_SOFTMAX,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params={
            qnn_translations.ir_graph.QNN_OP_SOFTMAX_PARAM_AXIS:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis)),
            qnn_translations.ir_graph.QNN_OP_SOFTMAX_PARAM_BETA:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta)),
        },
    )


def _patched_convolution_add_op_to_backend(self, node, graph, backend, **kwargs):
    conv_type, tensor_params, scalar_params = self.get_conv_params(backend, graph, node)

    output_dtype = None
    try:
        input_dtypes = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]
        float_pair = {qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_16, qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_32}
        unique_input_dtypes = set(input_dtypes)
        if unique_input_dtypes and unique_input_dtypes.issubset(float_pair) and len(unique_input_dtypes) == 1:
            output_dtype = next(iter(unique_input_dtypes))
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = output_dtype
    except Exception:
        output_dtype = None

    outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=output_dtype)
    self.squash_relu(node, graph, backend, outputs_info)
    backend.add_node(
        node.op.name,
        conv_type,
        input_names=node.input_names,
        outputs_info=outputs_info,
        tensor_params=tensor_params,
        scalar_params=scalar_params,
        macs=node.op.macs,
    )


def _patched_concat_add_op_to_backend(self, node, graph, backend, **kwargs):
    output_dtype = None
    try:
        input_dtypes = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]
        float_pair = {qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_16, qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_32}
        unique_input_dtypes = set(input_dtypes)
        if unique_input_dtypes and unique_input_dtypes.issubset(float_pair):
            output_dtype = (
                qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_32
                if qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_32 in unique_input_dtypes
                else qnn_translations.ir_graph.QNN_DATATYPE_FLOAT_16
            )

            for cast_input_idx, source_dtype in enumerate(input_dtypes):
                if source_dtype == output_dtype:
                    continue
                cast_input_name = node.input_names[cast_input_idx]
                cast_op_name = cast_input_name + f"_cast_to_{_qnn_dtype_name(output_dtype)}"
                if not graph.has_buffer(cast_op_name):
                    cast_op = op_adapter.CastOp(
                        cast_op_name,
                        from_type=_qnn_dtype_name(source_dtype),
                        to_type=_qnn_dtype_name(output_dtype),
                    )
                    if not graph.has_buffer(cast_input_name):
                        raise KeyError(f"Graph has no buffer {cast_input_name}, referred to as input for {node.op.name}")
                    buf = graph.get_buffer(cast_input_name)
                    idx_to_insert = 0
                    if buf.producer is not None:
                        idx_to_insert = graph.nodes_in_order.index(buf.producer) + 1
                    cast_node = graph.add(cast_op, input_names=[cast_input_name], output_names=[cast_op_name], idx=idx_to_insert)
                    if node in buf.consumers:
                        buf.consumers.remove(node)
                    cast_outputs_info = backend.get_outputs_info(
                        cast_node,
                        graph,
                        tensor_data_type=output_dtype,
                        check_encodings=True,
                    )
                    backend.add_node(cast_node.op.name, qnn_definitions.QNN_OP_CAST, cast_node.input_names, cast_outputs_info)
                if graph.has_buffer(cast_op_name):
                    graph.get_buffer(cast_op_name).consumers.add(node)
                node.input_names[cast_input_idx] = cast_op_name

            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = output_dtype
    except Exception:
        output_dtype = None

    backend.add_node(
        node.op.name,
        qnn_translations.ir_graph.QNN_OP_CONCAT,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params={
            qnn_translations.ir_graph.QNN_OP_CONCAT_PARAM_AXIS:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))
        },
    )


def _patched_resize_add_op_to_backend(self, node, graph, backend, **kwargs):
    scalar_params = {}
    output_shape = graph.get_output_buffers(node)[0].shape
    op_type = qnn_translations.ir_graph.QNN_OP_RESIZE
    backend_name = getattr(backend, 'backend_name', '')

    if len(output_shape) in [3, 5] or node.op.interpolation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC or \
            backend_name == "LPAI":
        scalar_params.update({
            qnn_translations.ir_graph.QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], node.op.exclude_outside),
            qnn_translations.ir_graph.QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], node.op.transformation_mode),
            qnn_translations.ir_graph.QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], node.op.interpolation_mode),
        })
        if node.op.interpolation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST:
            scalar_params.update({
                qnn_translations.ir_graph.QNN_OP_RESIZE_PARAM_NEAREST_MODE:
                    (qnn_translations.numpy_dtype_to_qnn[np.dtype('uint32')], node.op.nearest_mode)
            })
        elif node.op.interpolation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC:
            scalar_params.update({
                qnn_translations.ir_graph.QNN_OP_RESIZE_PARAM_CUBIC_COEFF:
                    (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], node.op.cubic_coeff)
            })
    elif len(output_shape) == 4:
        qnn_resize_op = self.ir_consts_to_qnn[node.op.interpolation_mode]
        op_type = qnn_resize_op["qnn_type"]
        align_corners = node.op.transformation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
        half_pixel_centers = node.op.transformation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL
        _, output_height, output_width, _ = output_shape
        if node.op.transformation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL and \
                (output_height > 1 and output_width > 1):
            half_pixel_centers = True

        scalar_params.update({
            qnn_resize_op["align_corners"]:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], align_corners),
            qnn_resize_op["half_pixel_centers"]:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], half_pixel_centers),
        })
        if node.op.interpolation_mode == qnn_translations.ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR:
            scalar_params.update({
                qnn_translations.ir_graph.QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS:
                    (qnn_translations.numpy_dtype_to_qnn[np.dtype('bool')], node.op.antialias)
            })
    else:
        raise ValueError(f"Node {node.op.name}: Expected ResizeOp with output rank 3/4/5, but got {len(output_shape)}")

    if backend.serialize_with_suppl_attr:
        scalar_params.update({
            qnn_translations.ir_graph.IR_OP_RESIZE_PARAM_SCALE_WIDTH:
                (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_width),
                 qnn_translations.ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        if len(output_shape) >= 4:
            scalar_params.update({
                qnn_translations.ir_graph.IR_OP_RESIZE_PARAM_SCALE_HEIGHT:
                    (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_height),
                     qnn_translations.ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
            })
        if len(output_shape) == 5:
            scalar_params.update({
                qnn_translations.ir_graph.IR_OP_RESIZE_PARAM_SCALE_DEPTH:
                    (qnn_translations.numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_depth),
                     qnn_translations.ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
            })

    output_dtype = None
    if len(node.input_names) >= 1:
        try:
            input_dtype = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            output_dtype = input_dtype
            if graph.has_buffer(node.output_names[0]):
                graph.get_buffer(node.output_names[0]).dtype = input_dtype
        except Exception:
            output_dtype = None

    backend.add_node(
        node.op.name,
        op_type,
        node.input_names,
        backend.get_outputs_info(node, graph, tensor_data_type=output_dtype),
        scalar_params=scalar_params,
        macs=node.op.macs,
    )


def _install_eltwise_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnElementwiseTranslation):
            bound_method = types.MethodType(_patched_eltwise_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnElementwiseTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnElementwiseTranslation instance not found in translation bank")


def _install_groupnorm_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnGroupNormTranslation):
            bound_method = types.MethodType(_patched_groupnorm_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnGroupNormTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnGroupNormTranslation instance not found in translation bank")


def _install_reshape_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnReshapeTranslation):
            bound_method = types.MethodType(_patched_reshape_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnReshapeTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnReshapeTranslation instance not found in translation bank")


def _install_layernorm_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnLayernormTranslation):
            bound_method = types.MethodType(_patched_layernorm_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnLayernormTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnLayernormTranslation instance not found in translation bank")


def _install_transpose_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnTransposeTranslation):
            bound_method = types.MethodType(_patched_transpose_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnTransposeTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnTransposeTranslation instance not found in translation bank")


def _install_stridedslice_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnStridedSliceTranslation):
            bound_method = types.MethodType(_patched_stridedslice_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnStridedSliceTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnStridedSliceTranslation instance not found in translation bank")


def _install_elementwiseneuron_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnElementwiseNeuronTranslation):
            bound_method = types.MethodType(_patched_elementwiseneuron_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnElementwiseNeuronTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnElementwiseNeuronTranslation instance not found in translation bank")


def _install_matmul_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnMatMulTranslation):
            bound_method = types.MethodType(_patched_matmul_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnMatMulTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnMatMulTranslation instance not found in translation bank")


def _install_softmax_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnSoftmaxTranslation):
            bound_method = types.MethodType(_patched_softmax_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnSoftmaxTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnSoftmaxTranslation instance not found in translation bank")


def _install_convolution_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnConvolutionTranslation):
            bound_method = types.MethodType(_patched_convolution_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnConvolutionTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnConvolutionTranslation instance not found in translation bank")


def _install_concat_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnConcatTranslation):
            bound_method = types.MethodType(_patched_concat_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnConcatTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnConcatTranslation instance not found in translation bank")


def _install_resize_translation_patch():
    patched = False
    seen_ids = set()
    for translation_obj in qnn_translations.QnnTranslations.translations.values():
        if id(translation_obj) in seen_ids:
            continue
        seen_ids.add(id(translation_obj))
        if isinstance(translation_obj, qnn_translations.QnnResizeTranslation):
            bound_method = types.MethodType(_patched_resize_add_op_to_backend, translation_obj)
            translation_obj.add_op_to_backend = bound_method
            translation_obj.register_method("add_op_to_backend", bound_method)
            patched = True
    if patched:
        print("[patch] QnnResizeTranslation add_op_to_backend rebound in translation bank")
    else:
        print("[patch] WARNING: QnnResizeTranslation instance not found in translation bank")


class ONNXtoQNNArgParser(ArgParserWrapper):
    def __init__(self):
        super().__init__(
            formatter_class=CustomHelpFormatter,
            conflict_handler="resolve",
            parents=[p for p in [
                onnx_frontend.OnnxConverterFrontend.ArgParser(),
                IROptimizations.ArgParser(),
                QnnQuantizer.ArgParser(),
                QnnConverterBackend.ArgParser(),
                ArchLinter.ArgParser(),
                GraphOptimizer.ArgParser() if GraphOptimizer is not None else None,
            ] if p is not None],
        )
        self.add_optional_argument(
            "--validate_models",
            action="store_true",
            help="Validate the original onnx model against optimized onnx model.",
        )
        self.parser.description = "Patched script to convert ONNX model into QNN using ExpandDims/Squeeze IR ops"


def _patched_unsqueeze_extract_parameters(self, src_op, converter_context):
    graph = converter_context.ir_graph
    params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
    axes = []
    if len(src_op.input) > 1:
        axes_input = str(src_op.input[1])
        axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
    elif "axes" in params:
        axes = params.axes

    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate unsqueeze dims {axes} for Unsqueeze op {src_op.name}")

    input_name = str(src_op.input[0])
    const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
    if const_input_op is not None:
        w = const_input_op.tensor
        shape = [] if converter_context.weights.was_scalar(input_name) else w.shape
        output_shape = onnx_data_translations.OnnxUnsqueezeTranslation._get_unsqueezed_shape(shape, axes)
        w = np.reshape(w, output_shape)
        output_name = str(src_op.output[0])
        converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
        if graph.has_quantization_param(const_input_op.name):
            graph.copy_quantization_param(const_input_op.name, output_name, input_name, output_name)
        elif graph.user_quantization_overrides:
            encoding = graph.get_overridden_encoding(const_input_op.name, False)
            if encoding is not None:
                graph.set_overridden_encoding(output_name, encoding, False)
        return None

    if len(src_op.input) > 1:
        converter_context.update_weights_trace_info_for_op(src_op.name, str(src_op.input[1]))

    input_shape = graph.get_buffer(input_name).shape[:]
    input_rank = len(input_shape)

    def map_axis_to_qnn(axis: int) -> int:
        if axis < 0:
            axis = axis + input_rank + 1
        if axis == input_rank:
            return input_rank - 1
        return axis

    qnn_axes = [map_axis_to_qnn(int(axis)) for axis in axes]
    return op_adapter.ExpandDimsOp(src_op.name, axes=qnn_axes)


def _patched_squeeze_extract_parameters(self, src_op, converter_context):
    graph = converter_context.ir_graph
    input_name = str(src_op.input[0])
    params = extract_attributes(src_op, schema=self.op_schema())

    axes = []
    if len(src_op.input) > 1:
        axes_input = str(src_op.input[1])
        axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
    elif "axes" in params:
        axes = params.axes

    const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
    if const_input_op is not None:
        output_name = str(src_op.output[0])
        w = converter_context.weights.fetch(input_name)
        if not len(axes):
            axes = [i for i, s in enumerate(w.shape) if s == 1]
        output_shape = onnx_data_translations.OnnxSqueezeTranslation._get_squeezed_shape(w.shape, axes)
        w = np.reshape(w, output_shape)
        was_scalar = False
        if not w.shape:
            was_scalar = True
            w = w.reshape(1)
        converter_context.insert_weights(output_name, w, was_scalar, [src_op.name], src_op.input)
        return None

    input_buf = graph.get_buffer(input_name)
    input_shape = input_buf.shape[:]
    if not len(axes):
        axes = [i for i, s in enumerate(input_shape) if s == 1]
    if not all(x < len(input_shape) for x in axes):
        raise ValueError(f"Squeeze op {src_op.name} dims {axes} greater than input rank {len(input_shape)}")
    if not all((input_shape[x] == 1) for x in axes):
        raise ValueError(f"input_shape[axes] all point to dims==1 for Squeeze op {src_op.name}")
    return op_adapter.SqueezeOp(src_op.name, axes=axes)


def apply_patches():
    onnx_data_translations.OnnxUnsqueezeTranslation.extract_parameters = _patched_unsqueeze_extract_parameters
    onnx_data_translations.OnnxSqueezeTranslation.extract_parameters = _patched_squeeze_extract_parameters
    QnnConverterBackend.add_node = _traced_backend_add_node
    qnn_translations.QnnElementwiseTranslation.add_op_to_backend = _patched_eltwise_add_op_to_backend
    qnn_translations.QnnGroupNormTranslation.add_op_to_backend = _patched_groupnorm_add_op_to_backend
    qnn_translations.QnnReshapeTranslation.add_op_to_backend = _patched_reshape_add_op_to_backend
    qnn_translations.QnnLayernormTranslation.add_op_to_backend = _patched_layernorm_add_op_to_backend
    qnn_translations.QnnTransposeTranslation.add_op_to_backend = _patched_transpose_add_op_to_backend
    qnn_translations.QnnStridedSliceTranslation.add_op_to_backend = _patched_stridedslice_add_op_to_backend
    qnn_translations.QnnElementwiseNeuronTranslation.add_op_to_backend = _patched_elementwiseneuron_add_op_to_backend
    qnn_translations.QnnMatMulTranslation.add_op_to_backend = _patched_matmul_add_op_to_backend
    qnn_translations.QnnSoftmaxTranslation.add_op_to_backend = _patched_softmax_add_op_to_backend
    qnn_translations.QnnConvolutionTranslation.add_op_to_backend = _patched_convolution_add_op_to_backend
    qnn_translations.QnnConcatTranslation.add_op_to_backend = _patched_concat_add_op_to_backend
    qnn_translations.QnnResizeTranslation.add_op_to_backend = _patched_resize_add_op_to_backend
    _install_eltwise_translation_patch()
    _install_groupnorm_translation_patch()
    _install_reshape_translation_patch()
    _install_layernorm_translation_patch()
    _install_transpose_translation_patch()
    _install_stridedslice_translation_patch()
    _install_elementwiseneuron_translation_patch()
    _install_matmul_translation_patch()
    _install_softmax_translation_patch()
    _install_convolution_translation_patch()
    _install_concat_translation_patch()
    _install_resize_translation_patch()
    print("[patch] ONNX Unsqueeze/Squeeze now map to ExpandDims/Squeeze IR ops for dynamic tensors")
    if TRACE_BACKEND_FILTER:
        print(f"[patch] Backend tracing active filter={TRACE_BACKEND_FILTER}")


def main():
    apply_patches()
    parser = ONNXtoQNNArgParser()
    args = parser.parse_args()

    try:
        validator = None
        if args.validate_models:
            if args.converter_op_package_lib:
                log_warning("Model is having custom ops skipping validation.")
                args.validate_models = False
            else:
                validator = Validator()

        converter = onnx_frontend.OnnxConverterFrontend(args, custom_op_factory=QnnCustomOpFactory(), validator=validator)
        ir_graph = converter.convert()

        args.perform_axes_to_spatial_first_order = True
        args.squash_box_decoder = True
        args.match_caffe_ssd_to_tf = True
        args.adjust_nms_features_dims = True
        args.extract_color_transform = True
        args.preprocess_roi_pool_inputs = True
        args.unroll_lstm_time_steps = True
        args.expand_gru_op_structure = True
        args.unroll_gru_time_steps = True
        args.inject_cast_for_gather = True
        args.force_prune_cast_ops = False
        args.align_matmul_ranks = True
        args.handle_gather_negative_indices = True

        optimizer = IROptimizations(args)
        optimized_graph = optimizer.optimize(ir_graph)

        backend = QnnConverterBackend(args)
        backend.save(optimized_graph)

        if args.arch_checker:
            log_warning("WARNING: --arch_checker from the conversion tool will be deprecated.")

        if args.validate_models:
            try:
                results = validator.validate()
                for result in results:
                    log_info(result)
            except Exception as e:
                log_warning(f"Model conversion is completed but validation failed: {e}")

    except Exception as e:
        log_error(f"Encountered Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
