import sys
import onnx
from onnx import TensorProto, numpy_helper
from pathlib import Path
import numpy as np

def sanitize_onnx_int64_to_int32(model_path: Path, output_path: Path, external_data_name: str):
    """
    Sanitize an ONNX model by converting all INT64 tensors, initializers,
    inputs, outputs, value_infos, and Cast operations to INT32.
    This completely prevents QNN parser integer alignment / pointer overflow bugs on Windows.
    """
    print(f"[sanitize] Loading ONNX model from {model_path}...")
    model = onnx.load(str(model_path), load_external_data=True)
    
    # 1. Convert initializers
    print("[sanitize] Converting INT64 initializers to INT32...")
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            arr_i32 = arr.astype(np.int32)
            new_init = numpy_helper.from_array(arr_i32, name=init.name)
            init.CopyFrom(new_init)
            
    # 2. Convert inputs
    print("[sanitize] Converting INT64 inputs to INT32...")
    for input_val in model.graph.input:
        if input_val.type.tensor_type.elem_type == TensorProto.INT64:
            input_val.type.tensor_type.elem_type = TensorProto.INT32
            
    # 3. Convert outputs
    print("[sanitize] Converting INT64 outputs to INT32...")
    for output_val in model.graph.output:
        if output_val.type.tensor_type.elem_type == TensorProto.INT64:
            output_val.type.tensor_type.elem_type = TensorProto.INT32
            
    # 4. Convert value_info
    print("[sanitize] Converting INT64 value_infos to INT32...")
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.INT64:
            vi.type.tensor_type.elem_type = TensorProto.INT32
            
    # 5. Convert Cast node targets
    print("[sanitize] Converting Cast node targets (INT64 -> INT32)...")
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.INT64:
                    attr.i = TensorProto.INT32
                    
    # Save the sanitized model back
    print(f"[sanitize] Saving sanitized ONNX to {output_path} with size_threshold=10240...")
    onnx.save(
        model, 
        str(output_path), 
        save_as_external_data=True,
        all_tensors_to_one_file=True, 
        location=external_data_name,
        size_threshold=10240
    )
    print("[sanitize] Done!")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python sanitize_onnx_types.py <input.onnx> <output.onnx> <external_data_name>")
        sys.exit(1)
    sanitize_onnx_int64_to_int32(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
