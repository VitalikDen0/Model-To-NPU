#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


DEFAULT_EXPORT_MANIFEST = Path(
    r"D:\platform-tools\wan21_13b_work\onnx\wan_t2v_1p3b_832x480_17f_seq128\export_manifest.json"
)
DEFAULT_AIHUB_ROOT = Path(r"D:\platform-tools\wan21_13b_work\aihub")
DEFAULT_CONTEXT_ROOT = Path(r"D:\platform-tools\wan21_13b_work\aihub_context")
DEFAULT_DEVICE_NAME = "Snapdragon 8 Elite QRD"
DEFAULT_COMPILE_OPTIONS = (
    "--target_runtime qnn_context_binary "
    "--qnn_options default_graph_htp_precision=FLOAT16"
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_component_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)).strip("_") or "component"


def _default_qnn_manifest(export_manifest_path: Path, export_manifest: dict[str, Any]) -> Path:
    run_tag = str(export_manifest.get("run_tag") or export_manifest_path.parent.name)
    qnn_root = export_manifest_path.parents[2] / "qnn"
    return qnn_root / f"{run_tag}_qnn_manifest.json"


def _discover_external_data_locations(onnx_path: Path) -> list[str]:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx package is required for AI Hub package preparation") from exc

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    locations: set[str] = set()

    def _collect_tensor_locations(tensor: Any) -> None:
        for entry in getattr(tensor, "external_data", []):
            if entry.key == "location" and entry.value:
                locations.add(entry.value)

    for tensor in model.graph.initializer:
        _collect_tensor_locations(tensor)

    for sparse in model.graph.sparse_initializer:
        _collect_tensor_locations(sparse.values)
        _collect_tensor_locations(sparse.indices)

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                _collect_tensor_locations(attr.t)
            elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                _collect_tensor_locations(attr.sparse_tensor.values)
                _collect_tensor_locations(attr.sparse_tensor.indices)
    return sorted(locations)


def _link_or_copy(src: Path, dst: Path, *, prefer_copy: bool = False) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if prefer_copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def _prepare_package_dir(source_onnx: Path, package_dir: Path, *, force: bool) -> dict[str, Any]:
    if force and package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    external_locations = _discover_external_data_locations(source_onnx)
    created: list[dict[str, str]] = []

    onnx_target = package_dir / "model.onnx"
    created.append({
        "kind": "onnx",
        "source": str(source_onnx),
        "target": str(onnx_target),
        "mode": _link_or_copy(source_onnx, onnx_target),
    })

    for location in external_locations:
        source_file = source_onnx.parent / location
        if not source_file.exists():
            raise FileNotFoundError(f"External ONNX data missing: {source_file}")
        target_file = package_dir / location
        created.append({
            "kind": "external_data",
            "source": str(source_file),
            "target": str(target_file),
            "mode": _link_or_copy(source_file, target_file, prefer_copy=True),
        })

    return {
        "package_dir": str(package_dir),
        "model": str(onnx_target),
        "external_data": external_locations,
        "created_files": created,
    }


def prepare_packages(
    export_manifest_path: Path,
    aihub_root: Path,
    *,
    components: set[str] | None,
    force: bool,
) -> Path:
    export_manifest = _load_json(export_manifest_path)
    run_tag = str(export_manifest.get("run_tag") or export_manifest_path.parent.name)
    available_components = export_manifest.get("components", {})
    if not isinstance(available_components, dict) or not available_components:
        raise SystemExit("export manifest has no components")

    selected = sorted(name for name in available_components.keys() if components is None or name in components)
    if not selected:
        raise SystemExit(f"No requested components found in export manifest. Available: {sorted(available_components.keys())}")

    run_root = aihub_root / run_tag
    package_root = run_root / "packages"
    package_manifest_path = run_root / "package_manifest.json"

    package_manifest: dict[str, Any] = {
        "run_tag": run_tag,
        "export_manifest": str(export_manifest_path),
        "components": {},
    }

    for component_name in selected:
        component_info = available_components[component_name]
        if not isinstance(component_info, dict):
            continue
        source_onnx = Path(str(component_info.get("onnx", "")))
        if not source_onnx.exists():
            raise FileNotFoundError(f"ONNX missing for component {component_name}: {source_onnx}")

        package_dir = package_root / f"{_safe_component_name(component_name)}.onnx"
        prepared = _prepare_package_dir(source_onnx, package_dir, force=force)
        package_manifest["components"][component_name] = {
            "source_onnx": str(source_onnx),
            "metadata": str(component_info.get("metadata", "")),
            **prepared,
        }

    _save_json(package_manifest_path, package_manifest)
    print(f"[prepare] package manifest: {package_manifest_path}")
    for component_name, info in package_manifest["components"].items():
        print(f"  - {component_name}: {info['package_dir']}")
    return package_manifest_path


def _load_package_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Package manifest not found: {path}. Run --prepare first.")
    manifest = _load_json(path)
    components = manifest.get("components", {})
    if not isinstance(components, dict) or not components:
        raise SystemExit(f"Package manifest is empty: {path}")
    return manifest


def submit_compile_jobs(
    package_manifest_path: Path,
    *,
    device_name: str,
    compile_options: str,
    components: set[str] | None,
) -> None:
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise SystemExit("qai_hub package is required for --compile") from exc

    manifest = _load_package_manifest(package_manifest_path)
    run_root = package_manifest_path.parent
    jobs_root = run_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    run_tag = str(manifest.get("run_tag") or run_root.name)

    for component_name, info in manifest["components"].items():
        if components is not None and component_name not in components:
            continue
        package_dir = Path(str(info["package_dir"]))
        if not package_dir.exists():
            raise FileNotFoundError(f"AI Hub package directory is missing: {package_dir}")
        job_name = f"WAN_{run_tag}_{_safe_component_name(component_name)}"
        print(f"[compile] submitting {component_name}")
        print(f"  device: {device_name}")
        print(f"  model : {package_dir}")
        print(f"  opts  : {compile_options}")
        job = hub.submit_compile_job(
            model=str(package_dir),
            device=hub.Device(device_name),
            name=job_name,
            options=compile_options,
        )
        job_id = job.job_id
        job_file = jobs_root / f"{_safe_component_name(component_name)}.txt"
        job_file.write_text(job_id, encoding="utf-8")
        print(f"  job_id: {job_id}")
        print(f"  url   : https://app.aihub.qualcomm.com/jobs/{job_id}/")


def check_status(package_manifest_path: Path, *, components: set[str] | None) -> None:
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise SystemExit("qai_hub package is required for --status") from exc

    manifest = _load_package_manifest(package_manifest_path)
    jobs_root = package_manifest_path.parent / "jobs"

    for component_name in manifest["components"].keys():
        if components is not None and component_name not in components:
            continue
        job_file = jobs_root / f"{_safe_component_name(component_name)}.txt"
        if not job_file.exists():
            print(f"[status] {component_name}: no job id")
            continue
        job_id = job_file.read_text(encoding="utf-8").strip()
        job = hub.get_job(job_id)
        status = job.get_status()
        print(f"[status] {component_name}: {status.code}")
        if getattr(status, "message", None):
            print(f"  message: {status.message}")


def _resolve_context_output_name(component_name: str, qnn_manifest: dict[str, Any] | None) -> str:
    if qnn_manifest is not None:
        components = qnn_manifest.get("components", {})
        if isinstance(components, dict):
            info = components.get(component_name)
            if isinstance(info, dict):
                context_name = str(info.get("context_binary_output", "")).strip()
                if context_name:
                    return context_name
    return f"{_safe_component_name(component_name)}.bin"


def download_contexts(
    package_manifest_path: Path,
    *,
    context_root: Path,
    qnn_manifest_path: Path | None,
    components: set[str] | None,
) -> None:
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise SystemExit("qai_hub package is required for --download") from exc

    manifest = _load_package_manifest(package_manifest_path)
    jobs_root = package_manifest_path.parent / "jobs"
    run_tag = str(manifest.get("run_tag") or package_manifest_path.parent.name)

    qnn_manifest: dict[str, Any] | None = None
    if qnn_manifest_path is not None and qnn_manifest_path.exists():
        qnn_manifest = _load_json(qnn_manifest_path)

    out_dir = context_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for component_name in manifest["components"].keys():
        if components is not None and component_name not in components:
            continue
        job_file = jobs_root / f"{_safe_component_name(component_name)}.txt"
        if not job_file.exists():
            print(f"[download] {component_name}: no job id")
            continue
        job_id = job_file.read_text(encoding="utf-8").strip()
        job = hub.get_job(job_id)
        status = job.get_status()
        if not status.success:
            print(f"[download] {component_name}: not complete ({status.code})")
            if getattr(status, "message", None):
                print(f"  message: {status.message}")
            continue
        file_name = _resolve_context_output_name(component_name, qnn_manifest)
        out_path = out_dir / file_name
        print(f"[download] {component_name} -> {out_path}")
        download_target_model = getattr(job, "download_target_model", None)
        if not callable(download_target_model):
            raise RuntimeError(f"AI Hub job object for {component_name} has no download_target_model()")
        download_target_model(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  done: {size_mb:.1f} MB")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare and compile Wan ONNX components through Qualcomm AI Hub")
    ap.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    ap.add_argument("--package-manifest", type=Path,
                    help="Optional explicit package manifest path; defaults to <aihub-root>/<run_tag>/package_manifest.json")
    ap.add_argument("--qnn-manifest", type=Path,
                    help="Optional QNN manifest used to derive final downloaded context file names")
    ap.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    ap.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    ap.add_argument("--device", default=DEFAULT_DEVICE_NAME)
    ap.add_argument("--compile-options", default=DEFAULT_COMPILE_OPTIONS)
    ap.add_argument("--component", action="append", default=[],
                    help="Component to operate on (repeatable). Default: all components in export manifest")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--all", action="store_true", help="Prepare + compile")
    ap.add_argument("--force", action="store_true", help="Rebuild AI Hub package directories")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not any([args.prepare, args.compile, args.status, args.download, args.all]):
        raise SystemExit("Choose at least one action: --prepare / --compile / --status / --download / --all")

    if not args.export_manifest.exists():
        raise SystemExit(f"Export manifest not found: {args.export_manifest}")

    export_manifest = _load_json(args.export_manifest)
    run_tag = str(export_manifest.get("run_tag") or args.export_manifest.parent.name)
    package_manifest_path = args.package_manifest or (args.aihub_root / run_tag / "package_manifest.json")
    qnn_manifest_path = args.qnn_manifest or _default_qnn_manifest(args.export_manifest, export_manifest)
    selected_components = set(args.component) if args.component else None

    if args.prepare or args.all:
        package_manifest_path = prepare_packages(
            args.export_manifest,
            args.aihub_root,
            components=selected_components,
            force=args.force,
        )

    if args.compile or args.all:
        submit_compile_jobs(
            package_manifest_path,
            device_name=args.device,
            compile_options=args.compile_options,
            components=selected_components,
        )

    if args.status:
        check_status(package_manifest_path, components=selected_components)

    if args.download:
        download_contexts(
            package_manifest_path,
            context_root=args.context_root,
            qnn_manifest_path=qnn_manifest_path,
            components=selected_components,
        )


if __name__ == "__main__":
    main()