#!/usr/bin/env python3
from __future__ import annotations

"""Единый статус/деплой helper для WAN 2.1, Flux и SD3.5.

Скрипт специально не пытается "угадывать" готовность по наличию папки.
Он проверяет реальные рабочие артефакты:

- WAN 2.1: QNN manifest + model.bin + Android .so + deploy helper.
- Flux: реальные `model.onnx` и/или готовые `*.bin` context binary.
- SD3.5: фактическое содержимое скачанных model dirs, а не только факт создания каталога.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_DIR = REPO_ROOT / "WAN 2.1 1.3B"
FLUX_DIR = REPO_ROOT / "Flux.2"
SD35_DIR = REPO_ROOT / "SD3.5"
DEFAULT_ADB = Path(r"D:\platform-tools\adb.exe")


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run(cmd: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _find_adb(explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if DEFAULT_ADB.exists():
        candidates.append(DEFAULT_ADB)
    which = shutil.which("adb")
    if which:
        candidates.append(Path(which))
    for candidate in candidates:
        result = _run([str(candidate), "version"], timeout=15)
        if result.returncode == 0:
            return candidate
    return None


def _ready_devices(adb_path: Path | None) -> list[str]:
    if adb_path is None:
        return []
    result = _run([str(adb_path), "devices"], timeout=30)
    if result.returncode != 0:
        return []
    devices: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _glob_sorted(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob(pattern))


def _rglob_sorted(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def _format_path(path: Path | None) -> str:
    return str(path) if path is not None else "missing"


def _wan_status() -> dict[str, Any]:
    manifest_candidates = _glob_sorted(WAN_DIR / "output", "*_qnn_manifest.json") + _glob_sorted(
        Path(r"D:\platform-tools\wan21_13b_work\qnn"),
        "*_qnn_manifest.json",
    )
    deploy_script = WAN_DIR / "deploy_wan_basic_debug.py"
    status: dict[str, Any] = {
        "model": "wan21",
        "label": "WAN 2.1",
        "state": "missing",
        "summary": "QNN manifest не найден",
        "recommended_variant": "IPostYellow/Wan2.1-T2V-1.3B-INT8-Diffusers",
        "size_class": "самый компактный Wan-кандидат, который сейчас реально ведётся в репо (но это всё ещё тяжёлый T2V 1.3B)",
        "next_host_step": "Подготовить уменьшённый AI Hub export профиля transformer, например через `export_wan_to_onnx.py --profile aihub-compact --component transformer`.",
        "details": [],
        "paths": {
            "deploy_script": str(deploy_script),
        },
        "ready_for_phone_deploy": False,
    }

    manifest_path = manifest_candidates[0] if manifest_candidates else None
    status["paths"]["qnn_manifest"] = _format_path(manifest_path)
    if manifest_path is None:
        status["details"].append("Нет готового WAN QNN manifest — сначала нужен локальный convert_to_qnn path.")
        return status

    manifest = _load_json(manifest_path)
    transformer = manifest.get("components", {}).get("transformer", {})
    model_bin = Path(str(transformer.get("model_bin", "")))
    android_lib = Path(str(transformer.get("android_lib", "")))
    run_tag = str(manifest.get("run_tag", "")).strip()

    context_roots = [
        Path(r"D:\platform-tools\wan21_13b_work\aihub_context") / run_tag,
        WAN_DIR / "output" / "qnn_local",
    ]
    context_dir = _first_existing(context_roots)
    context_files = [] if context_dir is None else _rglob_sorted(context_dir, "*.bin*")

    status["paths"].update(
        {
            "model_bin": str(model_bin),
            "android_lib": str(android_lib),
            "context_dir": _format_path(context_dir),
        }
    )

    details = status["details"]
    if model_bin.exists():
        details.append("Локальный WAN QNN `model.bin` существует.")
    else:
        details.append("Локальный WAN QNN `model.bin` отсутствует.")
    if android_lib.exists():
        details.append("Android runtime library `.so` для transformer уже собрана.")
    else:
        details.append("Android runtime library `.so` для transformer отсутствует.")
    if context_files:
        details.append(f"Найдены готовые context binary: {len(context_files)} шт.")
    else:
        details.append("Готовых WAN context binary пока нет; текущий путь — on-device context generation через deploy helper.")
    if deploy_script.exists():
        details.append("Есть dedicated deploy helper: `WAN 2.1 1.3B/deploy_wan_basic_debug.py`.")
    else:
        details.append("WAN deploy helper отсутствует.")

    ready = model_bin.exists() and android_lib.exists() and deploy_script.exists()
    status["ready_for_phone_deploy"] = ready
    if ready:
        status["state"] = "ready-for-phone-deploy"
        status["summary"] = "WAN локально дошёл до QNN model.bin + Android .so; нужен телефон для on-device context generation и runtime smoke"
    else:
        status["state"] = "partial"
        status["summary"] = "WAN частично готов, но не все локальные артефакты присутствуют"
    return status


def _flux_status() -> dict[str, Any]:
    onnx_dir = _first_existing([
        Path(r"D:\platform-tools\flux_work\onnx_ready\FLUX.1-schnell-onnx"),
        FLUX_DIR / "onnx_ready" / "FLUX.1-schnell-onnx",
    ])
    model_dir = _first_existing([
        Path(r"D:\platform-tools\flux_work\models"),
        FLUX_DIR / "models",
    ])
    qnn_dir = _first_existing([
        Path(r"D:\platform-tools\flux_work\qnn_context"),
        FLUX_DIR / "qnn_context",
    ])

    onnx_files = [] if onnx_dir is None else _rglob_sorted(onnx_dir, "model.onnx")
    qnn_bins = [] if qnn_dir is None else _glob_sorted(qnn_dir, "*.bin")
    downloaded_models = [] if model_dir is None else [p.name for p in sorted(model_dir.iterdir()) if p.is_dir()]

    status: dict[str, Any] = {
        "model": "flux",
        "label": "Flux",
        "state": "missing",
        "summary": "Flux assets не найдены",
        "recommended_variant": "black-forest-labs/FLUX.1-schnell",
        "size_class": "самый компактный Flux-вариант, на который реально завязаны текущие helper-скрипты",
        "next_host_step": "Докачать реальные веса или готовый ONNX snapshot; сейчас локально лежат только README/metadata без model.onnx и без весов.",
        "details": [],
        "paths": {
            "models_dir": _format_path(model_dir),
            "onnx_dir": _format_path(onnx_dir),
            "qnn_dir": _format_path(qnn_dir),
        },
        "ready_for_phone_deploy": False,
    }

    details = status["details"]
    if downloaded_models:
        details.append(f"Скачанные model dirs: {', '.join(downloaded_models)}")
    else:
        details.append("Полных Flux model dirs не найдено.")

    if onnx_dir is not None and not onnx_files:
        details.append("Папка FLUX.1-schnell-onnx существует, но реальных `model.onnx` внутри нет.")
    elif onnx_files:
        details.append(f"Найдены реальные Flux ONNX файлы: {len(onnx_files)} шт.")
    else:
        details.append("Flux ONNX артефакты отсутствуют.")

    if qnn_bins:
        details.append(f"Найдены Flux context binaries: {len(qnn_bins)} шт.")
        status["state"] = "compiled-contexts-present"
        status["summary"] = "Flux context binaries есть, но unified deploy/runtime path в репо ещё не оформлен"
    elif downloaded_models or onnx_dir is not None:
        status["state"] = "incomplete-download-or-export"
        status["summary"] = "Flux скачан/начат, но готовых ONNX или QNN context binaries для телефона нет"
    return status


def _sd35_variant_state(variant_dir: Path) -> str:
    if not variant_dir.exists() or not variant_dir.is_dir():
        return "missing"
    entries = {p.name for p in variant_dir.iterdir() if p.name != ".cache"}
    required_markers = {"scheduler", "text_encoder", "transformer", "vae"}
    if required_markers.issubset(entries):
        return "substantial"
    if entries:
        return "partial"
    return "missing"


def _sd35_status() -> dict[str, Any]:
    models_dir = _first_existing([
        Path(r"D:\platform-tools\sd35_work\models"),
        SD35_DIR / "models",
    ])
    medium_dir = None if models_dir is None else models_dir / "stable-diffusion-3.5-medium"
    large_dir = None if models_dir is None else models_dir / "stable-diffusion-3.5-large-turbo"

    medium_state = _sd35_variant_state(medium_dir) if medium_dir is not None else "missing"
    large_state = _sd35_variant_state(large_dir) if large_dir is not None else "missing"

    status: dict[str, Any] = {
        "model": "sd35",
        "label": "SD3.5",
        "state": "missing",
        "summary": "SD3.5 artifacts не найдены",
        "recommended_variant": "stabilityai/stable-diffusion-3.5-medium",
        "size_class": "самая маленькая осмысленная ветка SD3.5 из текущих скриптов (~5 GB против ~11.9 GB у large-turbo)",
        "next_host_step": "Доскачать именно medium-вариант полностью; large-turbo сейчас только увеличивает риск и не даёт phone-first преимущества.",
        "details": [],
        "paths": {
            "models_dir": _format_path(models_dir),
            "medium_dir": _format_path(medium_dir),
            "large_turbo_dir": _format_path(large_dir),
        },
        "ready_for_phone_deploy": False,
    }

    details = status["details"]
    details.append(f"medium: {medium_state}")
    details.append(f"large-turbo: {large_state}")

    if medium_state == "substantial" or large_state == "substantial":
        status["state"] = "downloaded-not-exported"
        status["summary"] = "SD3.5 веса на диске есть, но export/compile/deploy pipeline ещё не готов"
    elif medium_state == "partial" or large_state == "partial":
        status["state"] = "partial-download"
        status["summary"] = "SD3.5 скачан частично: есть только куски snapshot, без export/QNN артефактов"
    return status


def _collect_statuses() -> dict[str, dict[str, Any]]:
    return {
        "wan21": _wan_status(),
        "flux": _flux_status(),
        "sd35": _sd35_status(),
    }


def _print_statuses(statuses: dict[str, dict[str, Any]], *, adb_path: Path | None, devices: list[str]) -> None:
    print("=" * 72)
    print("Model-to-NPU status")
    print("=" * 72)
    print(f"ADB: {adb_path if adb_path else 'not found'}")
    print(f"Ready devices: {', '.join(devices) if devices else 'none'}")
    for key in ("wan21", "flux", "sd35"):
        item = statuses[key]
        print("\n" + "-" * 72)
        print(f"{item['label']}")
        print("-" * 72)
        print(f"state   : {item['state']}")
        print(f"summary : {item['summary']}")
        print(f"variant : {item.get('recommended_variant', 'n/a')}")
        print(f"size    : {item.get('size_class', 'n/a')}")
        print(f"next    : {item.get('next_host_step', 'n/a')}")
        for name, value in item["paths"].items():
            print(f"{name:<11}: {value}")
        if item.get("details"):
            print("details :")
            for line in item["details"]:
                print(f"  - {line}")


def _deploy_wan21(status: dict[str, Any], *, adb_path: Path | None, serial: str | None, dry_run: bool) -> bool:
    deploy_script = Path(status["paths"]["deploy_script"])
    manifest_path = Path(status["paths"]["qnn_manifest"])
    if not status["ready_for_phone_deploy"]:
        print("[WAN 2.1] not ready for phone deploy:")
        for line in status.get("details", []):
            print(f"  - {line}")
        return False
    if adb_path is None and not dry_run:
        print("[WAN 2.1] adb not found")
        return False

    cmd = [sys.executable, str(deploy_script), "--manifest", str(manifest_path)]
    if adb_path is not None:
        cmd += ["--adb", str(adb_path)]
    if serial:
        cmd += ["--serial", serial]

    if dry_run:
        print("[WAN 2.1] dry-run deploy command:")
        print("  " + " ".join(cmd))
        return True

    print("[WAN 2.1] deploying via dedicated helper...")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def _deploy_flux(status: dict[str, Any], *, dry_run: bool) -> bool:
    print("[Flux] unified phone deploy is not ready yet.")
    for line in status.get("details", []):
        print(f"  - {line}")
    if dry_run:
        return True
    return False


def _deploy_sd35(status: dict[str, Any], *, dry_run: bool) -> bool:
    print("[SD3.5] unified phone deploy is not ready yet.")
    for line in status.get("details", []):
        print(f"  - {line}")
    if dry_run:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Status/deploy helper for WAN 2.1, Flux and SD3.5")
    parser.add_argument("--status", action="store_true", help="Show actual readiness status")
    parser.add_argument("--json", action="store_true", help="Print status as JSON")
    parser.add_argument("--all", action="store_true", help="Deploy all models that have deploy paths")
    parser.add_argument("--model", choices=["wan21", "flux", "sd35"], help="Deploy a specific model")
    parser.add_argument("--dry-run", action="store_true", help="Print deploy action without executing it")
    parser.add_argument("--adb", type=str, default=None, help="Explicit adb path")
    parser.add_argument("--serial", type=str, default=None, help="Explicit device serial")
    args = parser.parse_args()

    adb_path = _find_adb(args.adb)
    devices = _ready_devices(adb_path)
    statuses = _collect_statuses()

    if args.json:
        print(json.dumps({"adb": str(adb_path) if adb_path else None, "devices": devices, "models": statuses}, ensure_ascii=False, indent=2))
        return

    if args.status or (not args.all and not args.model):
        _print_statuses(statuses, adb_path=adb_path, devices=devices)
        if not args.all and not args.model:
            return

    targets = [args.model] if args.model else ["wan21", "flux", "sd35"]
    success: list[str] = []
    failed: list[str] = []
    for target in targets:
        if target == "wan21":
            ok = _deploy_wan21(statuses[target], adb_path=adb_path, serial=args.serial, dry_run=args.dry_run)
        elif target == "flux":
            ok = _deploy_flux(statuses[target], dry_run=args.dry_run)
        else:
            ok = _deploy_sd35(statuses[target], dry_run=args.dry_run)
        (success if ok else failed).append(statuses[target]["label"])

    print("\n" + "=" * 72)
    if success:
        print(f"OK     : {', '.join(success)}")
    if failed:
        print(f"FAILED : {', '.join(failed)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
