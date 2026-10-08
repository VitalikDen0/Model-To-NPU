from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class ModelPreset:
    preset: str
    repo_id: str
    category: str
    step_reduction: str
    resolution_note: str
    readiness: str
    phone_priority: str
    notes: str


PRESETS: dict[str, ModelPreset] = {
    "int8-diffusers": ModelPreset(
        preset="int8-diffusers",
        repo_id="IPostYellow/Wan2.1-T2V-1.3B-INT8-Diffusers",
        category="quantized diffusers baseline",
        step_reduction="No confirmed few-step reduction",
        resolution_note="Start at 480p; treat 720p as phase 2",
        readiness="Best start-now candidate",
        phone_priority="high",
        notes=(
            "Public diffusers-style repo, about 15.4 GB on the Hugging Face tree page. "
            "It does not solve the step-count problem, but it is the most actionable public 1.3B branch found so far."
        ),
    ),
    "official-diffusers": ModelPreset(
        preset="official-diffusers",
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        category="official diffusers baseline",
        step_reduction="Standard inference",
        resolution_note="Officially 480p-first; 720p possible but less stable",
        readiness="Stable fallback baseline",
        phone_priority="high",
        notes=(
            "Use this when you want the clean official reference path. "
            "This is the baseline to compare against before claiming any speed or quality win elsewhere."
        ),
    ),
    "worstcoder-converted": ModelPreset(
        preset="worstcoder-converted",
        repo_id="worstcoder/Wan",
        category="converted official pth assets for rCM/TurboDiffusion-style tooling",
        step_reduction="Not few-step by itself",
        resolution_note="Contains 1.3B / 14B / VAE / text encoder assets",
        readiness="Useful prep-assets repo, not a proven low-step drop-in",
        phone_priority="medium",
        notes=(
            "This repo appears to host converted official checkpoints and related assets. "
            "Treat it as support material for the rCM branch, not as proof that a public ready-to-run few-step 1.3B student is already available."
        ),
    ),
    "lightx2v-distill-reference": ModelPreset(
        preset="lightx2v-distill-reference",
        repo_id="lightx2v/Wan2.1-Distill-Models",
        category="4-step distillation reference",
        step_reduction="4-step",
        resolution_note="Public tree is mainly 14B / I2V-oriented for this search",
        readiness="Useful reference, weak as direct 1.3B start",
        phone_priority="low",
        notes=(
            "Important reference for what a real 4-step Wan family looks like, but it did not surface a clean public 1.3B Lightning-style path in this pass."
        ),
    ),
    "maty-rcm-watch": ModelPreset(
        preset="maty-rcm-watch",
        repo_id="maty0505/Wan1.3B-rCM-8step-iter20000",
        category="watchlist / unverified",
        step_reduction="Claimed 8-step in the name",
        resolution_note="Unknown",
        readiness="Do not trust yet",
        phone_priority="low",
        notes=(
            "The public tree looked effectively empty / unverified during this pass. "
            "Keep it on a watchlist, not on the critical path."
        ),
    ),
}

PRESET_ORDER = [
    "int8-diffusers",
    "official-diffusers",
    "worstcoder-converted",
    "lightx2v-distill-reference",
    "maty-rcm-watch",
]

DEFAULT_DOWNLOAD_PRESET = "int8-diffusers"


def _print_block(title: str, value: str) -> None:
    print(f"{title:<16}: {value}")


def _preset_to_dict(preset: ModelPreset) -> dict[str, str]:
    return asdict(preset)


def _find_adb(explicit: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    which_adb = shutil.which("adb")
    if which_adb:
        candidates.append(Path(which_adb))

    env_roots = [
        os.environ.get("ANDROID_SDK_ROOT"),
        os.environ.get("ANDROID_HOME"),
    ]
    for root in env_roots:
        if root:
            candidates.append(Path(root) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb"))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "Android" / "Sdk" / "platform-tools" / "adb.exe")

    home = Path.home()
    candidates.extend(
        [
            home / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
            home / "Android" / "Sdk" / "platform-tools" / "adb.exe",
            home / "platform-tools" / "adb.exe",
            home / "platform-tools" / "adb",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _run_capture(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return result


def _parse_adb_devices(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        entry = {
            "serial": parts[0],
            "state": parts[1],
            "raw": line,
        }
        for token in parts[2:]:
            if ":" in token:
                key, value = token.split(":", 1)
                entry[key] = value
        devices.append(entry)
    return devices


def _adb_shell(adb_path: Path, serial: str, *shell_args: str, check: bool = True) -> str:
    result = _run_capture([str(adb_path), "-s", serial, "shell", *shell_args], check=check)
    return (result.stdout or "").strip()


def _choose_device(devices: list[dict[str, str]], requested_serial: Optional[str]) -> dict[str, str]:
    ready = [device for device in devices if device.get("state") == "device"]
    if requested_serial:
        for device in ready:
            if device.get("serial") == requested_serial:
                return device
        raise SystemExit(f"Requested serial '{requested_serial}' was not found among ready devices.")

    if not ready:
        raise SystemExit("No ready adb device found.")
    if len(ready) > 1:
        serials = ", ".join(device["serial"] for device in ready)
        raise SystemExit(f"Multiple ready devices detected ({serials}). Re-run with --serial.")
    return ready[0]


def _slugify(text: str) -> str:
    return text.replace("/", "__")


def cmd_models(args: argparse.Namespace) -> int:
    payload = [_preset_to_dict(PRESETS[name]) for name in PRESET_ORDER]
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for name in PRESET_ORDER:
        preset = PRESETS[name]
        print(f"[{preset.preset}] {preset.repo_id}")
        _print_block("category", preset.category)
        _print_block("readiness", preset.readiness)
        _print_block("step", preset.step_reduction)
        _print_block("resolution", preset.resolution_note)
        _print_block("priority", preset.phone_priority)
        _print_block("notes", preset.notes)
        print()
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    recommendation = {
        "start_now": PRESETS["int8-diffusers"].repo_id,
        "fallback_baseline": PRESETS["official-diffusers"].repo_id,
        "few_step_research": "NVlabs/rcm (checkpoint availability for public 1.3B student still needs verification)",
        "resolution_strategy": "480p first, 720p second",
        "reasoning": [
            "Official Wan 2.1 1.3B documentation recommends 480p and says 720p is less stable.",
            "A clean public 1.3B Lightning/Distill drop-in was not confidently verified.",
            "INT8 Diffusers is the most actionable public 1.3B branch for immediate work.",
            "rCM is the closest conceptual match to the requested low-step path, but the public 1.3B student checkpoint situation remains unclear.",
        ],
    }
    if args.json:
        print(json.dumps(recommendation, indent=2, ensure_ascii=False))
        return 0

    print("Recommended starting path")
    print("=========================")
    print(f"Start now          : {recommendation['start_now']}")
    print(f"Fallback baseline  : {recommendation['fallback_baseline']}")
    print(f"Few-step research  : {recommendation['few_step_research']}")
    print(f"Resolution         : {recommendation['resolution_strategy']}")
    print()
    print("Why this is the current call:")
    for item in recommendation["reasoning"]:
        print(f"- {item}")
    return 0


def cmd_phone_check(args: argparse.Namespace) -> int:
    adb_path = _find_adb(args.adb)
    if not adb_path:
        raise SystemExit(
            "adb was not found. Pass --adb explicitly or install Android platform-tools."
        )

    devices_output = _run_capture([str(adb_path), "devices", "-l"], check=True).stdout
    devices = _parse_adb_devices(devices_output)
    chosen = _choose_device(devices, args.serial)
    serial = chosen["serial"]

    report = {
        "adb": str(adb_path),
        "serial": serial,
        "listed_model": chosen.get("model", ""),
        "android_model": _adb_shell(adb_path, serial, "getprop", "ro.product.model", check=False),
        "platform": _adb_shell(adb_path, serial, "getprop", "ro.board.platform", check=False),
        "android_version": _adb_shell(adb_path, serial, "getprop", "ro.build.version.release", check=False),
        "physical_size": _adb_shell(adb_path, serial, "wm", "size", check=False),
        "physical_density": _adb_shell(adb_path, serial, "wm", "density", check=False),
        "storage": _adb_shell(adb_path, serial, "df", "-h", "/data", "/sdcard", check=False),
        "wan_resolution_advice": "480p first; try 720p only after a sane 480p path exists.",
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print("Connected phone report")
    print("======================")
    for key in [
        "adb",
        "serial",
        "listed_model",
        "android_model",
        "platform",
        "android_version",
        "physical_size",
        "physical_density",
    ]:
        _print_block(key, str(report.get(key, "")))
    print("storage          :")
    print(report["storage"] or "<no storage output>")
    print()
    print(f"Wan advice        : {report['wan_resolution_advice']}")
    return 0


def _resolve_download_presets(requested: Iterable[str]) -> list[ModelPreset]:
    names = list(requested)
    if not names:
        names = [DEFAULT_DOWNLOAD_PRESET]

    resolved: list[ModelPreset] = []
    seen: set[str] = set()
    aliases = {
        "start-now": "int8-diffusers",
        "fallback-now": "official-diffusers",
    }

    for name in names:
        actual = aliases.get(name, name)
        if actual not in PRESETS:
            valid = ", ".join(sorted(list(PRESETS.keys()) + list(aliases.keys())))
            raise SystemExit(f"Unknown preset '{name}'. Valid values: {valid}")
        if actual in seen:
            continue
        seen.add(actual)
        resolved.append(PRESETS[actual])
    return resolved


def cmd_download(args: argparse.Namespace) -> int:
    presets = _resolve_download_presets(args.preset)
    destination = Path(args.dest).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = [
            {
                "preset": preset.preset,
                "repo_id": preset.repo_id,
                "local_dir": str(destination / _slugify(preset.preset)),
                "allow_patterns": args.allow_pattern,
                "ignore_patterns": args.ignore_pattern,
            }
            for preset in presets
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for download. Install it with: python -m pip install huggingface_hub"
        ) from exc

    token = os.environ.get(args.token_env) or None
    downloaded: list[dict[str, str]] = []
    for preset in presets:
        local_dir = destination / _slugify(preset.preset)
        snapshot_download(
            repo_id=preset.repo_id,
            local_dir=str(local_dir),
            token=token,
            allow_patterns=args.allow_pattern or None,
            ignore_patterns=args.ignore_pattern or None,
        )
        downloaded.append({"preset": preset.preset, "repo_id": preset.repo_id, "local_dir": str(local_dir)})

    print(json.dumps(downloaded, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Helpers for Wan 2.1 1.3B model selection, downloads, and phone probing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="Show the current Wan candidate matrix.")
    models_parser.add_argument("--json", action="store_true", help="Print the candidate matrix as JSON.")
    models_parser.set_defaults(func=cmd_models)

    recommend_parser = subparsers.add_parser("recommend", help="Print the current recommended starting path.")
    recommend_parser.add_argument("--json", action="store_true", help="Print the recommendation as JSON.")
    recommend_parser.set_defaults(func=cmd_recommend)

    phone_parser = subparsers.add_parser("phone-check", help="Probe the connected phone via adb.")
    phone_parser.add_argument("--adb", help="Path to adb or adb.exe.")
    phone_parser.add_argument("--serial", help="Explicit device serial if more than one phone is connected.")
    phone_parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    phone_parser.set_defaults(func=cmd_phone_check)

    download_parser = subparsers.add_parser("download", help="Download one or more selected Hugging Face repos.")
    download_parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help=(
            "Preset to download. Can be repeated. "
            "Valid names include int8-diffusers, official-diffusers, worstcoder-converted, "
            "lightx2v-distill-reference, maty-rcm-watch, start-now, fallback-now."
        ),
    )
    download_parser.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parent / "downloads"),
        help="Destination directory for downloads.",
    )
    download_parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional Hugging Face allow-pattern. Can be repeated.",
    )
    download_parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Optional Hugging Face ignore-pattern. Can be repeated.",
    )
    download_parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name that may contain a Hugging Face token.",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download plan without downloading anything.",
    )
    download_parser.set_defaults(func=cmd_download)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
