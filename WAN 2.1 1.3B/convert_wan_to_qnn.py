#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_EXPORT_MANIFEST = Path(__file__).parent / "output" / "wan_t2v_1p3b_832x480_17f_seq128" / "export_manifest.json"
DEFAULT_QNN_ROOT = Path(__file__).parent / "output" / "qnn"
DEFAULT_SDK_ROOT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130")
DEFAULT_NDK_ROOT = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")


def _resolve_helper_path(file_name: str) -> Path:
    candidates = [
        ROOT_DIR / "SDXL" / "debug" / file_name,
        Path(r"D:\platform-tools\GitHub\SDXL\debug") / file_name,
        Path(r"D:\platform-tools\sdxl_npu") / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required helper script not found: {file_name}")


PATCHED_CONVERTER = _resolve_helper_path("qnn_onnx_converter_expanddims_patch.py")
ANDROID_LIB_BUILDER = _resolve_helper_path("build_android_model_lib_windows.py")

# QAIRT 2.44 libPyIrGraph.pyd is compiled against numpy 1.x ABI.
# numpy 2.x changed the PyArrayObject struct layout so IrStaticTensor
# reads garbage data for ALL parameters stored via np.array → IrStaticTensor.
# We use a dedicated venv with numpy 1.26 for the converter subprocess.
_CONVERTER_VENV_PYTHON = Path(r"D:\platform-tools\sdxl_npu\converter_venv\Scripts\python.exe")
CONVERTER_PYTHON = str(_CONVERTER_VENV_PYTHON) if _CONVERTER_VENV_PYTHON.exists() else sys.executable


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run(cmd: list[str], label: str, *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    print(f"\n{'=' * 80}\n[{label}]\n{'=' * 80}")
    print(" ".join(str(c) for c in cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _find_model_cpp_and_bin(qnn_dir: Path) -> tuple[Path, Path]:
    candidates = [
        (qnn_dir / "model.cpp", qnn_dir / "model.bin"),
        (qnn_dir / "model" / "model.cpp", qnn_dir / "model" / "model.bin"),
        (qnn_dir / "model", qnn_dir / "model.bin"),
    ]
    for cpp_path, bin_path in candidates:
        if cpp_path.exists() and bin_path.exists():
            return cpp_path, bin_path
    raise FileNotFoundError(f"Could not find model.cpp/model.bin under {qnn_dir}")


def _component_ids(component: str) -> tuple[str, str]:
    if component == "transformer":
        return "WanTransformer3D", "noise_pred"
    if component == "vae_decoder":
        return "WanVaeDecoder", "video"
    raise ValueError(f"Unsupported component: {component}")


def _tensor_shape_map(model: Any) -> dict[str, list[int | str]]:
    result: dict[str, list[int | str]] = {}
    for value_info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        result[value_info.name] = dims
    return result


def _fix_negative_axes(onnx_path: Path) -> None:
    """Rewrite negative axes in ReduceMean/Softmax/etc. to positive values.

    QAIRT 2.44 ReduceOp::canonicalizeOp does not handle negative axis
    values — the int64 -1 gets misinterpreted as a huge unsigned int.
    This pre-pass resolves every negative axis using the static input rank
    inferred from shape_inference.
    """
    import struct

    import numpy as np
    import onnx
    from onnx import numpy_helper, shape_inference

    model = onnx.load(str(onnx_path), load_external_data=False)
    inferred = shape_inference.infer_shapes(model)
    shape_map = _tensor_shape_map(inferred)

    # --- fix Reduce* ops whose axes arrive via the second input tensor ---
    reduce_op_types = {
        "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
        "ReduceProd", "ReduceL1", "ReduceL2",
    }
    init_by_name: dict[str, Any] = {init.name: init for init in model.graph.initializer}
    data_path = onnx_path.parent / "model.onnx.data"
    has_external = data_path.exists()
    fixed_reduce = 0

    # Build set of initializer names used by non-Reduce nodes to detect sharing
    non_reduce_init_users: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in reduce_op_types:
            for inp in node.input:
                if inp in init_by_name:
                    non_reduce_init_users.add(inp)

    # Track which (name, positive_axes_tuple) combos already have new initializers
    created_inits: dict[tuple[str, tuple[int, ...]], str] = {}

    def _read_int64_init(init: Any) -> list[int] | None:
        if init.external_data and has_external:
            offset = length = 0
            for ext in init.external_data:
                if ext.key == "offset":
                    offset = int(ext.value)
                if ext.key == "length":
                    length = int(ext.value)
            with open(data_path, "rb") as f:
                f.seek(offset)
                raw = f.read(length)
            return [struct.unpack("<q", raw[i * 8 : (i + 1) * 8])[0] for i in range(length // 8)]
        elif init.raw_data:
            n = len(init.raw_data) // 8
            return [struct.unpack("<q", init.raw_data[i * 8 : (i + 1) * 8])[0] for i in range(n)]
        return None

    def _make_int64_init(name: str, values: list[int]) -> Any:
        new_init = onnx.TensorProto()
        new_init.name = name
        new_init.data_type = onnx.TensorProto.INT64
        new_init.dims.extend([len(values)])
        new_init.raw_data = np.array(values, dtype=np.int64).tobytes()
        return new_init

    for node in model.graph.node:
        if node.op_type not in reduce_op_types:
            continue
        if len(node.input) < 2:
            continue
        axes_name = node.input[1]
        init = init_by_name.get(axes_name)
        if init is None:
            continue

        axes = _read_int64_init(init)
        if axes is None or not any(a < 0 for a in axes):
            continue

        data_input = node.input[0]
        input_shape = shape_map.get(data_input)
        if input_shape is None:
            continue
        rank = len(input_shape)

        new_axes = [(a % rank) for a in axes]
        new_axes_key = (axes_name, tuple(new_axes))

        # If the initializer is shared with non-Reduce ops, create a private copy
        is_shared = axes_name in non_reduce_init_users
        if is_shared:
            if new_axes_key in created_inits:
                node.input[1] = created_inits[new_axes_key]
            else:
                private_name = f"{axes_name}_reduce_r{rank}"
                new_init = _make_int64_init(private_name, new_axes)
                model.graph.initializer.append(new_init)
                init_by_name[private_name] = new_init
                created_inits[new_axes_key] = private_name
                node.input[1] = private_name
        else:
            if new_axes_key not in created_inits:
                new_init = _make_int64_init(axes_name, new_axes)
                for idx, existing in enumerate(model.graph.initializer):
                    if existing.name == axes_name:
                        del model.graph.initializer[idx]
                        break
                model.graph.initializer.append(new_init)
                init_by_name[axes_name] = new_init
                created_inits[new_axes_key] = axes_name

        fixed_reduce += 1

    # --- fix Softmax / LogSoftmax negative axis attribute ---
    fixed_softmax = 0
    for node in model.graph.node:
        if node.op_type not in ("Softmax", "LogSoftmax"):
            continue
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                data_input = node.input[0]
                input_shape = shape_map.get(data_input)
                if input_shape is None:
                    continue
                rank = len(input_shape)
                attr.i = int(attr.i) % rank
                fixed_softmax += 1

    if fixed_reduce or fixed_softmax:
        onnx.save(model, str(onnx_path))
        print(f"[patch] fixed negative axes: {fixed_reduce} Reduce* nodes, {fixed_softmax} Softmax nodes")
    else:
        print("[patch] no negative axes found; skipping")


def _patch_transformer_onnx_for_qnn(onnx_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file missing for patching: {onnx_path}")

    model = onnx.load(str(onnx_path), load_external_data=False)
    inferred = shape_inference.infer_shapes(model)
    shape_map = _tensor_shape_map(inferred)
    target_node = None
    transpose_node = None
    for node in model.graph.node:
        if node.name == "/transformer/Reshape":
            target_node = node
        elif node.name == "/transformer/Transpose":
            transpose_node = node

    if target_node is None:
        print("[patch] transformer reshape node not found; skipping ONNX pre-patch")
        return
    if transpose_node is None:
        print("[patch] transformer transpose node not found; skipping ONNX pre-patch")
        return

    input_shape = shape_map.get(target_node.input[0])
    if not input_shape or len(input_shape) != 5 or not all(isinstance(v, int) for v in input_shape):
        print(f"[patch] transformer reshape input shape not fully static: {input_shape}; skipping")
        return

    batch, channels, frames, height, width = (int(v) for v in input_shape)
    static_shape = np.asarray([batch, channels, frames * height * width], dtype=np.int64)
    static_name = "/transformer/StaticReshape_qnn_shape"

    to_delete = [idx for idx, init in enumerate(model.graph.initializer) if init.name == static_name]
    for idx in reversed(to_delete):
        del model.graph.initializer[idx]
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([batch, frames * height * width, channels], dtype=np.int64), name=static_name))

    if len(target_node.input) > 1:
        del target_node.input[1:]
    del target_node.attribute[:]
    target_node.op_type = "Transpose"
    target_node.attribute.append(helper.make_attribute("perm", [0, 2, 3, 4, 1]))

    if len(transpose_node.input) > 1:
        del transpose_node.input[1:]
    transpose_node.input.append(static_name)
    del transpose_node.attribute[:]
    transpose_node.op_type = "Reshape"

    onnx.save(model, str(onnx_path))
    print(
        f"[patch] rewrote patch embedding path to Transpose(N,D,H,W,C) -> Reshape([batch,tokens,channels]); "
        f"logical token shape was {static_shape.tolist()}"
    )


def convert_component(
    *,
    component: str,
    run_tag: str,
    onnx_path: Path,
    metadata_path: Path,
    qnn_root: Path,
    sdk_root: Path,
    ndk_root: Path,
    float_bitwidth: int,
) -> dict[str, Any]:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not PATCHED_CONVERTER.exists():
        raise FileNotFoundError(f"Patched converter entrypoint not found: {PATCHED_CONVERTER}")
    if not ANDROID_LIB_BUILDER.exists():
        raise FileNotFoundError(f"Android lib builder not found: {ANDROID_LIB_BUILDER}")

    metadata = _load_json(metadata_path)
    _fix_negative_axes(onnx_path)
    patch_embedding_mode = str(metadata.get("config", {}).get("patch_embedding_mode", "conv3d"))
    if component == "transformer" and patch_embedding_mode == "conv3d":
        _patch_transformer_onnx_for_qnn(onnx_path)
    elif component == "transformer":
        print(f"[info] transformer export uses patch_embedding_mode={patch_embedding_mode}; skipping legacy ONNX pre-patch")

    lib_base_name, default_output_name = _component_ids(component)
    component_tag = f"{run_tag}_{component}_fp{float_bitwidth}"
    component_root = qnn_root / component_tag
    qnn_model_dir = component_root / "qnn_model"
    android_lib_dir = component_root / "android_lib"
    android_build_dir = android_lib_dir / "_build"
    qairt_tmp = qnn_root / "_qairt_tmp"
    qairt_tmp.mkdir(parents=True, exist_ok=True)
    qnn_model_dir.mkdir(parents=True, exist_ok=True)
    android_lib_dir.mkdir(parents=True, exist_ok=True)

    qairt_env = {
        "PYTHONPATH": str(sdk_root / "lib" / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "QNN_SDK_ROOT": str(sdk_root),
        "TMPDIR": str(qairt_tmp),
        "TEMP": str(qairt_tmp),
        "TMP": str(qairt_tmp),
    }

    _run(
        [
            CONVERTER_PYTHON,
            str(PATCHED_CONVERTER),
            "--input_network",
            str(onnx_path),
            "--output_path",
            str(qnn_model_dir / "model"),
            "--float_bitwidth",
            str(float_bitwidth),
        ],
        f"qnn-onnx-converter: {component}",
        cwd=qnn_model_dir,
        extra_env=qairt_env,
    )

    model_cpp, model_bin = _find_model_cpp_and_bin(qnn_model_dir)
    so_name = f"lib{lib_base_name}_{run_tag}_fp{float_bitwidth}.so"

    _run(
        [
            sys.executable,
            str(ANDROID_LIB_BUILDER),
            "--sdk-root",
            str(sdk_root),
            "--model-cpp",
            str(model_cpp),
            "--model-bin",
            str(model_bin),
            "--ndk-root",
            str(ndk_root),
            "--build-dir",
            str(android_build_dir),
            "--lib-name",
            so_name,
        ],
        f"build Android lib: {component}",
        cwd=android_lib_dir,
    )

    built_so = android_build_dir / "libs" / "arm64-v8a" / so_name
    if not built_so.exists():
        raise FileNotFoundError(f"Android library missing after build: {built_so}")

    final_so = android_lib_dir / so_name
    shutil.copy2(built_so, final_so)

    context_binary_file_arg = f"{Path(so_name).stem[3:]}.serialized.bin"
    context_binary_output = f"{context_binary_file_arg}.bin"

    return {
        "component": component,
        "onnx": str(onnx_path),
        "metadata": str(metadata_path),
        "qnn_model_dir": str(qnn_model_dir),
        "model_cpp": str(model_cpp),
        "model_bin": str(model_bin),
        "android_lib": str(final_so),
        "android_build_dir": str(android_build_dir),
        "lib_base_name": lib_base_name,
        "output_name": metadata.get("output_names", [default_output_name])[0],
        "input_names": metadata.get("input_names", []),
        "shapes": metadata.get("shapes", {}),
        "dtypes": metadata.get("dtypes", {}),
        "config": metadata.get("config", {}),
        "patch_embedding_mode": patch_embedding_mode,
        "context_binary_file_arg": context_binary_file_arg,
        "context_binary_output": context_binary_output,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Wan ONNX exports to QNN model libs for Android phone experiments.")
    ap.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    ap.add_argument("--qnn-root", type=Path, default=DEFAULT_QNN_ROOT)
    ap.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    ap.add_argument("--ndk-root", type=Path, default=DEFAULT_NDK_ROOT)
    ap.add_argument("--float-bitwidth", type=int, default=16)
    ap.add_argument("--component", choices=["all", "transformer", "vae"], default="all")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.export_manifest.exists():
        raise SystemExit(f"Export manifest not found: {args.export_manifest}")
    if not args.sdk_root.exists():
        raise SystemExit(f"QNN SDK root not found: {args.sdk_root}")
    if not args.ndk_root.exists():
        raise SystemExit(f"Android NDK root not found: {args.ndk_root}")

    export_manifest = _load_json(args.export_manifest)
    run_tag = export_manifest["run_tag"]
    components = export_manifest.get("components", {})

    manifest: dict[str, Any] = {
        "run_tag": run_tag,
        "export_manifest": str(args.export_manifest),
        "sdk_root": str(args.sdk_root),
        "ndk_root": str(args.ndk_root),
        "float_bitwidth": args.float_bitwidth,
        "components": {},
    }

    requested = []
    if args.component in {"all", "transformer"}:
        requested.append(("transformer", "transformer"))
    if args.component in {"all", "vae"}:
        requested.append(("vae_decoder", "vae"))

    for export_component_name, user_component_name in requested:
        if export_component_name not in components:
            raise SystemExit(f"Component missing in export manifest: {export_component_name}")
        item = components[export_component_name]
        result = convert_component(
            component=export_component_name,
            run_tag=run_tag,
            onnx_path=Path(item["onnx"]),
            metadata_path=Path(item["metadata"]),
            qnn_root=args.qnn_root,
            sdk_root=args.sdk_root,
            ndk_root=args.ndk_root,
            float_bitwidth=args.float_bitwidth,
        )
        manifest["components"][user_component_name] = result
        print(f"[ok] {export_component_name}: {result['android_lib']}")

    manifest_path = args.qnn_root / f"{run_tag}_qnn_manifest.json"
    _save_json(manifest_path, manifest)
    print(f"[ok] qnn manifest: {manifest_path}")


if __name__ == "__main__":
    main()
