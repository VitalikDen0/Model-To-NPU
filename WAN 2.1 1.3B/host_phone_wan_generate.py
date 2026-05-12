#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


DEFAULT_QNN_MANIFEST = Path(r"D:\platform-tools\wan21_13b_work\qnn\wan_t2v_1p3b_832x480_17f_seq128_qnn_manifest.json")
DEFAULT_CORE_DIR = Path(r"D:\platform-tools\wan21_13b_work\official_core\official-diffusers")
DEFAULT_TEXT_ENCODER_DIR = Path(r"D:\platform-tools\wan21_13b_work\int8_text_encoder\int8-diffusers")
DEFAULT_OUTPUT_DIR = Path(r"D:\platform-tools\wan21_13b_work\outputs")
DEFAULT_SDK_ROOT = Path(r"D:\platform-tools\sdxl_npu\qairt_2.44\qairt\2.44.0.260225")
DEFAULT_NDK_ROOT = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")
DEFAULT_PHONE_BASE = "/data/local/tmp/wan21_t2v_qnn"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, low quality, ugly, deformed, "
    "disfigured, still picture, messy background"
)
QNN_RUNTIME_LIBS = [
    ("libQnnHtp.so", "lib/aarch64-android/libQnnHtp.so"),
    ("libQnnHtpNetRunExtensions.so", "lib/aarch64-android/libQnnHtpNetRunExtensions.so"),
    ("libQnnHtpPrepare.so", "lib/aarch64-android/libQnnHtpPrepare.so"),
    ("libQnnHtpProfilingReader.so", "lib/aarch64-android/libQnnHtpProfilingReader.so"),
    ("libQnnHtpV79Stub.so", "lib/aarch64-android/libQnnHtpV79Stub.so"),
    ("libQnnSystem.so", "lib/aarch64-android/libQnnSystem.so"),
    ("libQnnHtpV79Skel.so", "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"),
]


def _resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--host-vae-device=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")
QNN_RUNTIME_BINS = [
    ("qnn-net-run", "bin/aarch64-android/qnn-net-run"),
    ("qnn-context-binary-generator", "bin/aarch64-android/qnn-context-binary-generator"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _extract_manifest_resolution(manifest_payload: dict[str, Any], manifest_path: Path) -> tuple[int, int]:
    components = manifest_payload.get("components", {})
    transformer = components.get("transformer", {}) if isinstance(components, dict) else {}
    config = transformer.get("config", {}) if isinstance(transformer, dict) else {}

    width = int(config.get("width", 0) or 0)
    height = int(config.get("height", 0) or 0)
    if width > 0 and height > 0:
        return width, height

    run_tag = f"{manifest_payload.get('run_tag', '')} {manifest_path.stem}"
    match = re.search(r"(\d{3,5})x(\d{3,5})", run_tag)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(f"Could not infer resolution from manifest: {manifest_path}")


def _discover_manifest_candidates(primary_manifest: Path, manifest_dir: Path | None) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    def _add_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if not resolved.exists() or not resolved.is_file():
            return
        seen.add(resolved)
        discovered.append(resolved)

    _add_candidate(primary_manifest)

    search_roots: list[Path] = []
    if manifest_dir is not None:
        search_roots.append(manifest_dir)
    if primary_manifest.parent.exists():
        search_roots.append(primary_manifest.parent)

    for root in search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("*qnn_manifest*.json")):
            _add_candidate(candidate)
        for candidate in sorted(root.rglob("*qnn_manifest*.json")):
            _add_candidate(candidate)

    return discovered


def _select_manifest_for_resolution(
    candidates: list[Path],
    requested_width: int,
    requested_height: int,
    *,
    require_exact: bool,
) -> tuple[Path, dict[str, Any], list[str], bool]:
    entries: list[tuple[Path, dict[str, Any], int, int]] = []
    for candidate in candidates:
        try:
            payload = _load_json(candidate)
            width, height = _extract_manifest_resolution(payload, candidate)
            entries.append((candidate, payload, width, height))
        except Exception as exc:
            print(f"[warn] skip manifest {candidate}: {exc}")

    if not entries:
        raise SystemExit("No valid QNN manifest candidates were found")

    available_resolutions: list[str] = []
    seen_resolutions: set[str] = set()
    for _, _, width, height in entries:
        key = f"{width}x{height}"
        if key not in seen_resolutions:
            seen_resolutions.add(key)
            available_resolutions.append(key)

    selected = entries[0]
    is_exact_match = True

    if requested_width > 0 and requested_height > 0:
        exact_matches = [entry for entry in entries if entry[2] == requested_width and entry[3] == requested_height]
        if exact_matches:
            selected = exact_matches[0]
            is_exact_match = True
        else:
            if require_exact:
                raise SystemExit(
                    "Exact WAN manifest bucket was requested but not found: "
                    f"{requested_width}x{requested_height}; available: {', '.join(available_resolutions)}"
                )
            requested_area = requested_width * requested_height
            selected = min(
                entries,
                key=lambda entry: (
                    (entry[2] - requested_width) ** 2 + (entry[3] - requested_height) ** 2,
                    abs((entry[2] * entry[3]) - requested_area),
                ),
            )
            is_exact_match = False

    return selected[0], selected[1], available_resolutions, is_exact_match


def _slug(text: str, limit: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", text).strip("_")
    return text[:limit] or "run"


def _find_adb(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    which_adb = shutil.which("adb")
    if which_adb:
        candidates.append(Path(which_adb))

    for env_key in ("ANDROID_SDK_ROOT", "ANDROID_HOME", "LOCALAPPDATA"):
        value = os.environ.get(env_key)
        if not value:
            continue
        root = Path(value)
        if env_key == "LOCALAPPDATA":
            candidates.append(root / "Android" / "Sdk" / "platform-tools" / "adb.exe")
        else:
            candidates.append(root / "platform-tools" / "adb.exe")
            candidates.append(root / "platform-tools" / "adb")

    home = Path.home()
    candidates.extend(
        [
            home / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
            home / "Android" / "Sdk" / "platform-tools" / "adb.exe",
            home / "platform-tools" / "adb.exe",
        ]
    )

    for candidate in candidates:
        try:
            result = subprocess.run([str(candidate), "version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except Exception:
            continue
    raise SystemExit("adb not found. Pass --adb explicitly.")


def _run(cmd: list[str], *, label: str, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit={result.returncode}"
        raise RuntimeError(f"{label} failed: {detail}")
    return result


def _adb(adb: Path, serial: str, *args: str, label: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return _run([str(adb), "-s", serial, *args], label=label, timeout=timeout)


def _adb_shell(adb: Path, serial: str, command: str, *, root: bool = False, label: str, timeout: int = 600) -> str:
    if root:
        wrapped = f"su -c {shlex.quote(command)}"
        result = _run([str(adb), "-s", serial, "shell", wrapped], label=label, timeout=timeout)
    else:
        result = _run([str(adb), "-s", serial, "shell", command], label=label, timeout=timeout)
    return (result.stdout or "").strip()


def _adb_push(adb: Path, serial: str, local_path: Path, remote_path: str) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found for adb push: {local_path}")
    print(f"  push {local_path.name} -> {remote_path}")
    _adb(adb, serial, "push", str(local_path), remote_path, label=f"adb push {local_path.name}", timeout=1800)


def _adb_pull(adb: Path, serial: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _adb(adb, serial, "pull", remote_path, str(local_path), label=f"adb pull {Path(remote_path).name}", timeout=1800)


def _pick_serial(adb: Path, requested_serial: str | None) -> str:
    out = _run([str(adb), "devices", "-l"], label="adb devices", timeout=30).stdout
    ready: list[str] = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            ready.append(parts[0])
    if requested_serial:
        if requested_serial not in ready:
            raise SystemExit(f"Requested serial {requested_serial!r} not found among ready devices: {ready}")
        return requested_serial
    if not ready:
        raise SystemExit("No ready adb device found.")
    if len(ready) > 1:
        raise SystemExit(f"Multiple adb devices found: {', '.join(ready)}; pass --serial")
    return ready[0]


def _phone_paths(phone_base: str) -> dict[str, str]:
    base = phone_base.rstrip("/")
    return {
        "base": base,
        "bin": f"{base}/bin",
        "lib": f"{base}/lib",
        "model": f"{base}/model",
        "context": f"{base}/context",
        "inputs": f"{base}/inputs",
        "outputs": f"{base}/outputs",
        "work": f"{base}/work",
    }


def _ensure_phone_layout(adb: Path, serial: str, phone_base: str) -> dict[str, str]:
    paths = _phone_paths(phone_base)
    mkdirs = " && ".join(f"mkdir -p {v}" for k, v in paths.items() if k != "base")
    _adb_shell(adb, serial, mkdirs, root=True, label="prepare phone layout")
    return paths


def _find_ndk_cpp_shared(ndk_root: Path) -> Path:
    path = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "sysroot" / "usr" / "lib" / "aarch64-linux-android" / "libc++_shared.so"
    if not path.exists():
        raise FileNotFoundError(f"libc++_shared.so not found in NDK: {path}")
    return path


def _push_runtime_assets(adb: Path, serial: str, sdk_root: Path, ndk_root: Path, phone_base: str) -> None:
    paths = _phone_paths(phone_base)
    for file_name, relative in QNN_RUNTIME_LIBS:
        source = sdk_root / relative
        if not source.exists():
            raise FileNotFoundError(f"Missing QAIRT runtime library: {source}")
        _adb_push(adb, serial, source, f"{paths['lib']}/{file_name}")
    for file_name, relative in QNN_RUNTIME_BINS:
        source = sdk_root / relative
        if not source.exists():
            raise FileNotFoundError(f"Missing QAIRT runtime binary: {source}")
        remote = f"{paths['bin']}/{file_name}"
        _adb_push(adb, serial, source, remote)
        _adb_shell(adb, serial, f"chmod 755 {remote}", root=True, label=f"chmod {file_name}")
    cpp_shared = _find_ndk_cpp_shared(ndk_root)
    _adb_push(adb, serial, cpp_shared, f"{paths['lib']}/libc++_shared.so")


def _push_model_artifacts(adb: Path, serial: str, qnn_manifest: dict[str, Any], phone_base: str) -> dict[str, dict[str, str]]:
    paths = _phone_paths(phone_base)
    deployed: dict[str, dict[str, str]] = {}
    for component_name, info in qnn_manifest["components"].items():
        local_lib = Path(info["android_lib"])
        remote_lib = f"{paths['model']}/{local_lib.name}"
        _adb_push(adb, serial, local_lib, remote_lib)
        deployed[component_name] = {
            "lib": remote_lib,
            "context": f"{paths['context']}/{info['context_binary_output']}",
            "context_binary_file_arg": info["context_binary_file_arg"],
            "output_name": info["output_name"],
        }
    return deployed


def _remove_phone_model_libs(adb: Path, serial: str, deployed: dict[str, dict[str, str]]) -> None:
    libs = sorted({info["lib"] for info in deployed.values()})
    if not libs:
        return
    joined = " ".join(libs)
    _adb_shell(adb, serial, f"rm -f {joined}", root=True, label="remove temporary model libs")


def _phone_env_exports(phone_base: str) -> str:
    paths = _phone_paths(phone_base)
    return (
        f"export LD_LIBRARY_PATH={paths['lib']}:$LD_LIBRARY_PATH; "
        f"export ADSP_LIBRARY_PATH='{paths['lib']};/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp'; "
    )


def _ensure_phone_context(
    adb: Path,
    serial: str,
    phone_base: str,
    model_lib: str,
    context_binary_file_arg: str,
    expected_output_path: str,
) -> None:
    exists = _adb_shell(
        adb,
        serial,
        f"if [ -f {expected_output_path} ]; then echo present; else echo missing; fi",
        root=True,
        label=f"check context {Path(expected_output_path).name}",
    )
    if "present" in exists:
        print(f"  context already present: {expected_output_path}")
        return

    paths = _phone_paths(phone_base)
    cmd = (
        _phone_env_exports(phone_base)
        + f"{paths['bin']}/qnn-context-binary-generator "
        + f"--model {model_lib} "
        + f"--backend {paths['lib']}/libQnnHtp.so "
        + f"--output_dir {paths['context']} "
        + f"--binary_file {context_binary_file_arg}"
    )
    _adb_shell(adb, serial, cmd, root=True, label=f"generate context {Path(expected_output_path).name}", timeout=7200)

    verify = _adb_shell(
        adb,
        serial,
        f"if [ -f {expected_output_path} ]; then ls -lh {expected_output_path}; else exit 1; fi",
        root=True,
        label=f"verify context {Path(expected_output_path).name}",
    )
    print(verify)


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _prompt_clean(text: str) -> str:
    try:
        import ftfy  # type: ignore

        text = ftfy.fix_text(text)
    except Exception:
        pass
    text = html.unescape(html.unescape(text))
    return _whitespace_clean(text)


def _hf_subfolder_or_root(base_dir: Path, subfolder: str) -> tuple[Path, str | None]:
    candidate = base_dir / subfolder
    if candidate.exists():
        return base_dir, subfolder
    for child in sorted(base_dir.iterdir()) if base_dir.exists() else []:
        nested = child / subfolder
        if child.is_dir() and nested.exists():
            return child, subfolder
    return base_dir, None


def _encode_prompt(
    prompt: str,
    *,
    tokenizer_dir: Path,
    text_encoder_dir: Path,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    tok_base, tok_subfolder = _hf_subfolder_or_root(tokenizer_dir, "tokenizer")
    enc_base, enc_subfolder = _hf_subfolder_or_root(text_encoder_dir, "text_encoder")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_base),
        subfolder=tok_subfolder,
        local_files_only=True,
    )
    text_encoder = cast(Any, UMT5EncoderModel.from_pretrained(
        str(enc_base),
        subfolder=enc_subfolder,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ))
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    cleaned = [_prompt_clean(prompt)]
    text_inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        prompt_embeds = text_encoder(ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
    arr = prompt_embeds.detach().cpu().numpy().astype(np.float16, copy=False)

    del tokenizer, text_encoder, ids, mask, seq_lens, prompt_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


def _decode_video_host(
    model_dir: Path,
    vae_latents: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    except Exception:
        from diffusers import AutoencoderKLWan  # type: ignore[attr-defined]

    load_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = cast(Any, AutoencoderKLWan.from_pretrained(
        str(model_dir),
        subfolder="vae",
        local_files_only=True,
        torch_dtype=load_dtype,
    ))
    vae = vae.to(device=device, dtype=load_dtype)
    vae.eval()

    latents_t = torch.from_numpy(vae_latents).to(device=device, dtype=load_dtype)
    with torch.no_grad():
        video = vae.decode(latents_t, return_dict=False)[0]
    result = video.detach().cpu().numpy().astype(np.float32)

    del latents_t, video, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _scheduler_from_core(model_dir: Path, *, flow_shift: float) -> UniPCMultistepScheduler:
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(model_dir),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=flow_shift,
    )
    return scheduler


def _write_raw(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clear_remote_dir(adb: Path, serial: str, remote_dir: str, *, label: str) -> None:
    _adb_shell(adb, serial, f"rm -rf {remote_dir} && mkdir -p {remote_dir}", root=True, label=label)


def _run_phone_context(
    adb: Path,
    serial: str,
    *,
    phone_base: str,
    context_path: str,
    input_list_path: str,
    output_dir: str,
    timeout: int = 7200,
) -> float:
    paths = _phone_paths(phone_base)
    cmd = (
        _phone_env_exports(phone_base)
        + f"{paths['bin']}/qnn-net-run "
        + f"--retrieve_context {context_path} "
        + f"--backend {paths['lib']}/libQnnHtp.so "
        + f"--input_list {input_list_path} "
        + f"--output_dir {output_dir} "
        + "--perf_profile sustained_high_performance "
        + "--log_level warn --use_mmap --use_native_output_files"
    )
    t0 = time.time()
    _adb_shell(adb, serial, cmd, root=True, label=f"qnn-net-run {Path(context_path).name}", timeout=timeout)
    return time.time() - t0


def _find_first_raw(path: Path) -> Path:
    for candidate in sorted(path.rglob("*.raw")):
        return candidate
    raise FileNotFoundError(f"No .raw output found under {path}")


def _resolve_target_output_frames(
    source_frames: int,
    fps: int,
    min_duration_sec: float,
    target_frames: int,
    target_duration_sec: float,
) -> int:
    baseline = max(1, int(source_frames))
    min_required = max(1, int(math.ceil(max(0.0, float(min_duration_sec)) * float(fps))))
    baseline = max(baseline, min_required)

    if target_frames and target_frames > 0:
        return max(1, int(target_frames))
    if target_duration_sec and target_duration_sec > 0.0:
        return max(1, int(math.ceil(float(target_duration_sec) * float(fps))))
    return baseline


def _resize_video_frame_count(
    video_fhwc: np.ndarray,
    target_frames: int,
    *,
    upsample_fill: str,
) -> tuple[np.ndarray, str]:
    source_frames = int(video_fhwc.shape[0])
    target_frames = max(1, int(target_frames))

    if source_frames == target_frames:
        return video_fhwc, "exact"

    if target_frames < source_frames:
        indices = np.linspace(0, source_frames - 1, target_frames).round().astype(np.int64)
        return video_fhwc[indices], "downsample_linear"

    # target_frames > source_frames
    if upsample_fill == "loop":
        full_cycles = target_frames // source_frames
        remainder = target_frames % source_frames
        pieces = [video_fhwc for _ in range(full_cycles)]
        if remainder:
            pieces.append(video_fhwc[:remainder])
        return np.concatenate(pieces, axis=0), "upsample_loop"

    pad_count = target_frames - source_frames
    tail = np.repeat(video_fhwc[-1:, :, :, :], pad_count, axis=0)
    return np.concatenate([video_fhwc, tail], axis=0), "upsample_hold_last"


def _save_frames_and_video(
    video_fhwc: np.ndarray,
    output_dir: Path,
    *,
    fps: int,
    min_duration_sec: float,
    target_frames: int,
    target_duration_sec: float,
    upsample_fill: str,
) -> tuple[dict[str, str], np.ndarray]:
    source_frames = int(video_fhwc.shape[0])
    resolved_target_frames = _resolve_target_output_frames(
        source_frames,
        fps,
        min_duration_sec,
        target_frames,
        target_duration_sec,
    )
    resized, frame_strategy = _resize_video_frame_count(
        video_fhwc,
        resolved_target_frames,
        upsample_fill=upsample_fill,
    )
    output_frames = int(resized.shape[0])
    min_frames_required = max(1, int(math.ceil(max(0.0, float(min_duration_sec)) * float(fps))))

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(resized):
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:03d}.png")

    artifacts: dict[str, str] = {
        "frames_dir": str(frames_dir),
        "source_frames": str(source_frames),
        "output_frames": str(output_frames),
        "requested_target_frames": str(target_frames),
        "requested_target_duration_sec": str(target_duration_sec),
        "resolved_target_frames": str(resolved_target_frames),
        "frame_count_strategy": frame_strategy,
        "upsample_fill": upsample_fill,
        "min_required_frames": str(min_frames_required),
    }
    try:
        import imageio.v2 as imageio  # type: ignore

        mp4_path = output_dir / "result.mp4"
        imageio.mimsave(mp4_path, list(resized), fps=fps, quality=8)
        artifacts["mp4"] = str(mp4_path)
    except Exception as exc:
        artifacts["video_warning"] = f"Could not write MP4: {exc}"
    return artifacts, resized


def _estimate_memory_report(
    latent_shape: tuple[int, ...],
    output_video_shape: tuple[int, ...],
    *,
    fps: int,
) -> dict[str, int | float | list[int]]:
    latent_elems = int(np.prod(np.asarray(latent_shape, dtype=np.int64)))
    latent_fp16_bytes = latent_elems * 2
    latent_fp32_bytes = latent_elems * 4

    frames = int(output_video_shape[0])
    frame_h = int(output_video_shape[1])
    frame_w = int(output_video_shape[2])
    frame_c = int(output_video_shape[3])
    frame_rgb8_bytes = frame_h * frame_w * frame_c

    def _mib(value: int) -> float:
        return round(float(value) / (1024.0 * 1024.0), 4)

    return {
        "latent_shape": list(latent_shape),
        "output_video_shape": list(output_video_shape),
        "latent_elements": latent_elems,
        "latent_fp16_bytes": latent_fp16_bytes,
        "latent_fp16_mib": _mib(latent_fp16_bytes),
        "latent_fp32_bytes": latent_fp32_bytes,
        "latent_fp32_mib": _mib(latent_fp32_bytes),
        "frame_rgb8_bytes": frame_rgb8_bytes,
        "frame_rgb8_mib": _mib(frame_rgb8_bytes),
        "output_frames": frames,
        "output_rgb8_total_bytes": frame_rgb8_bytes * frames,
        "output_rgb8_total_mib": _mib(frame_rgb8_bytes * frames),
        "output_rgb8_bytes_per_second": frame_rgb8_bytes * int(fps),
        "output_rgb8_mib_per_second": _mib(frame_rgb8_bytes * int(fps)),
        "fps": int(fps),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Host-orchestrated Wan 2.1 1.3B generation with transformer/VAE inference executed on phone QNN/HTP.")
    ap.add_argument("--adb", type=str)
    ap.add_argument("--serial", type=str)
    ap.add_argument("--qnn-manifest", type=Path, default=DEFAULT_QNN_MANIFEST)
    ap.add_argument("--qnn-manifest-dir", type=Path, default=None,
                    help="Directory with multiple *qnn_manifest*.json buckets")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_CORE_DIR)
    ap.add_argument("--text-encoder-dir", type=Path, default=DEFAULT_TEXT_ENCODER_DIR)
    ap.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    ap.add_argument("--ndk-root", type=Path, default=DEFAULT_NDK_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--phone-base", type=str, default=DEFAULT_PHONE_BASE)
    ap.add_argument("--width", type=int, default=0,
                    help="Requested output width bucket (0 = use selected manifest default)")
    ap.add_argument("--height", type=int, default=0,
                    help="Requested output height bucket (0 = use selected manifest default)")
    ap.add_argument("--exact-resolution", action="store_true",
                    help="Fail if requested --width/--height bucket is missing")
    ap.add_argument("--vae-backend", choices=["host", "phone"], default="host")
    ap.add_argument("--host-vae-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--min-duration-sec", type=float, default=1.0)
    ap.add_argument("--target-frames", type=int, default=0,
                    help="Exact output frame count override (0 keeps shape/default flow)")
    ap.add_argument("--target-duration-sec", type=float, default=0.0,
                    help="Exact output duration override in seconds (used when --target-frames=0)")
    ap.add_argument("--upsample-fill", choices=["hold_last", "loop"], default="hold_last",
                    help="How to extend video when target frame count is higher than generated")
    ap.add_argument("--flow-shift", type=float, default=None)
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--skip-context-gen", action="store_true")
    ap.add_argument("--delete-model-libs-after-context", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if (args.width > 0) != (args.height > 0):
        raise SystemExit("Provide both --width and --height (or leave both at 0)")

    if not args.qnn_manifest.exists() and args.qnn_manifest_dir is None:
        raise SystemExit(f"QNN manifest not found: {args.qnn_manifest}")
    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")
    if not args.text_encoder_dir.exists():
        raise SystemExit(f"Text encoder dir not found: {args.text_encoder_dir}")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    manifest_candidates = _discover_manifest_candidates(args.qnn_manifest, args.qnn_manifest_dir)
    if not manifest_candidates:
        raise SystemExit("No QNN manifest candidates found")

    selected_manifest_path, qnn_manifest, available_resolutions, exact_resolution_match = _select_manifest_for_resolution(
        manifest_candidates,
        args.width,
        args.height,
        require_exact=args.exact_resolution,
    )

    transformer_info = qnn_manifest["components"]["transformer"]
    vae_info = qnn_manifest["components"].get("vae")
    if args.vae_backend == "phone" and vae_info is None:
        raise SystemExit("QNN manifest does not contain a VAE component, but --vae-backend=phone was requested")
    height = int(transformer_info["config"]["height"])
    width = int(transformer_info["config"]["width"])
    num_frames = int(transformer_info["config"]["num_frames"])
    max_seq_len = int(transformer_info["config"]["max_sequence_length"])
    latent_shape = tuple(int(v) for v in transformer_info["shapes"]["hidden_states"])
    flow_shift = args.flow_shift if args.flow_shift is not None else (3.0 if height <= 480 else 5.0)
    vae_cfg = _load_json(args.model_dir / "vae" / "config.json")

    adb = _find_adb(args.adb)
    serial = _pick_serial(adb, args.serial)
    requested_resolution = f"{args.width}x{args.height}" if args.width > 0 and args.height > 0 else f"{width}x{height}"
    resolved_resolution = f"{width}x{height}"

    print(f"Device: {serial}")
    print(f"Run tag: {qnn_manifest['run_tag']}")
    print(f"Requested resolution: {requested_resolution}")
    print(f"Resolved bucket: {resolved_resolution} ({'exact' if exact_resolution_match else 'nearest'})")
    print(f"Manifest: {selected_manifest_path}")
    if available_resolutions:
        print(f"Available buckets: {', '.join(available_resolutions)}")
    print(f"Fixed export shape: {width}x{height}, frames={num_frames}, seq={max_seq_len}")

    paths = _ensure_phone_layout(adb, serial, args.phone_base)
    deploy_manifest = {
        "components": {
            "transformer": qnn_manifest["components"]["transformer"],
        }
    }
    if args.vae_backend == "phone" and vae_info is not None:
        deploy_manifest["components"]["vae"] = vae_info

    deployed = _push_model_artifacts(adb, serial, deploy_manifest, args.phone_base)
    if not args.skip_deploy:
        _push_runtime_assets(adb, serial, args.sdk_root, args.ndk_root, args.phone_base)
    if not args.skip_context_gen:
        _ensure_phone_context(
            adb,
            serial,
            args.phone_base,
            deployed["transformer"]["lib"],
            deployed["transformer"]["context_binary_file_arg"],
            deployed["transformer"]["context"],
        )
        if args.vae_backend == "phone":
            _ensure_phone_context(
                adb,
                serial,
                args.phone_base,
                deployed["vae"]["lib"],
                deployed["vae"]["context_binary_file_arg"],
                deployed["vae"]["context"],
            )

    if args.delete_model_libs_after_context:
        _remove_phone_model_libs(adb, serial, deployed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Prompt encoder device: {device}")
    prompt_embeds = _encode_prompt(
        args.prompt,
        tokenizer_dir=args.model_dir,
        text_encoder_dir=args.text_encoder_dir,
        max_seq_len=max_seq_len,
        device=device,
        dtype=embed_dtype,
    )
    negative_prompt_embeds = None
    if args.guidance_scale > 1.0:
        negative_prompt_embeds = _encode_prompt(
            args.negative_prompt,
            tokenizer_dir=args.model_dir,
            text_encoder_dir=args.text_encoder_dir,
            max_seq_len=max_seq_len,
            device=device,
            dtype=embed_dtype,
        )

    scheduler = _scheduler_from_core(args.model_dir, flow_shift=flow_shift)
    scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = scheduler.timesteps

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(args.prompt)}"
    run_dir = args.output_dir / run_name
    host_io_dir = run_dir / "host_io"
    phone_pull_dir = run_dir / "phone_outputs"
    host_io_dir.mkdir(parents=True, exist_ok=True)
    phone_pull_dir.mkdir(parents=True, exist_ok=True)

    transformer_input_list_phone = f"{paths['work']}/transformer_input_list.txt"
    vae_input_list_phone = f"{paths['work']}/vae_input_list.txt"
    transformer_input_list_local = host_io_dir / "transformer_input_list.txt"
    vae_input_list_local = host_io_dir / "vae_input_list.txt"
    _write_text(
        transformer_input_list_local,
        f"{paths['inputs']}/hidden_states.raw {paths['inputs']}/timestep.raw {paths['inputs']}/prompt_embeds.raw\n",
    )
    _write_text(
        vae_input_list_local,
        f"{paths['inputs']}/vae_latents.raw\n",
    )
    _adb_push(adb, serial, transformer_input_list_local, transformer_input_list_phone)
    _adb_push(adb, serial, vae_input_list_local, vae_input_list_phone)

    prompt_embeds_path = host_io_dir / "prompt_embeds.raw"
    _write_raw(prompt_embeds_path, prompt_embeds.astype(np.float16, copy=False))
    _adb_push(adb, serial, prompt_embeds_path, f"{paths['inputs']}/prompt_embeds.raw")

    negative_prompt_embeds_path: Path | None = None
    if negative_prompt_embeds is not None:
        neg_path = host_io_dir / "negative_prompt_embeds.raw"
        _write_raw(neg_path, negative_prompt_embeds.astype(np.float16, copy=False))
        _adb_push(adb, serial, neg_path, f"{paths['inputs']}/negative_prompt_embeds.raw")
        negative_prompt_embeds_path = neg_path

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32)

    step_times: list[dict[str, float]] = []
    for step_index, t in enumerate(timesteps):
        print(f"\n=== denoise step {step_index + 1}/{len(timesteps)} ; t={float(t):.6f} ===")
        hidden_states_path = host_io_dir / "hidden_states.raw"
        timestep_path = host_io_dir / "timestep.raw"
        _write_raw(hidden_states_path, latents.detach().cpu().numpy().astype(np.float16, copy=False))
        _write_raw(timestep_path, np.asarray([float(t)], dtype=np.float32))
        _adb_push(adb, serial, hidden_states_path, f"{paths['inputs']}/hidden_states.raw")
        _adb_push(adb, serial, timestep_path, f"{paths['inputs']}/timestep.raw")

        transformer_remote_output_dir = f"{paths['outputs']}/transformer"
        _clear_remote_dir(adb, serial, transformer_remote_output_dir, label="clear transformer output dir")
        elapsed = _run_phone_context(
            adb,
            serial,
            phone_base=args.phone_base,
            context_path=deployed["transformer"]["context"],
            input_list_path=transformer_input_list_phone,
            output_dir=transformer_remote_output_dir,
        )
        transformer_local_output_dir = phone_pull_dir / f"step_{step_index:03d}_cond"
        _adb_pull(adb, serial, transformer_remote_output_dir, transformer_local_output_dir)
        cond_raw_path = _find_first_raw(transformer_local_output_dir)
        noise_pred = np.fromfile(cond_raw_path, dtype=np.float16).reshape(latent_shape).astype(np.float32)

        if args.guidance_scale > 1.0:
            if negative_prompt_embeds_path is None:
                raise RuntimeError("negative prompt embeddings were not prepared")
            _adb_shell(
                adb,
                serial,
                f"cp {paths['inputs']}/negative_prompt_embeds.raw {paths['inputs']}/prompt_embeds.raw",
                root=True,
                label="swap in negative prompt embeds",
            )
            _clear_remote_dir(adb, serial, transformer_remote_output_dir, label="clear transformer output dir (uncond)")
            uncond_elapsed = _run_phone_context(
                adb,
                serial,
                phone_base=args.phone_base,
                context_path=deployed["transformer"]["context"],
                input_list_path=transformer_input_list_phone,
                output_dir=transformer_remote_output_dir,
            )
            transformer_uncond_output_dir = phone_pull_dir / f"step_{step_index:03d}_uncond"
            _adb_pull(adb, serial, transformer_remote_output_dir, transformer_uncond_output_dir)
            uncond_raw_path = _find_first_raw(transformer_uncond_output_dir)
            noise_uncond = np.fromfile(uncond_raw_path, dtype=np.float16).reshape(latent_shape).astype(np.float32)
            noise_pred = noise_uncond + args.guidance_scale * (noise_pred - noise_uncond)
            _adb_shell(
                adb,
                serial,
                f"cp {paths['inputs']}/prompt_embeds.raw {paths['inputs']}/prompt_embeds.raw >/dev/null 2>&1 || true",
                root=True,
                label="noop restore cond embeds",
            )
            _adb_push(adb, serial, prompt_embeds_path, f"{paths['inputs']}/prompt_embeds.raw")
            elapsed += uncond_elapsed

        noise_pred_t = torch.from_numpy(noise_pred)
        latents = scheduler.step(noise_pred_t, t, latents, return_dict=False)[0].to(torch.float32)
        step_times.append({"step": float(step_index), "timestep": float(t), "seconds": float(elapsed)})
        print(f"  transformer seconds: {elapsed:.3f}")

    latents_mean = np.asarray(vae_cfg["latents_mean"], dtype=np.float32).reshape(1, int(vae_cfg["z_dim"]), 1, 1, 1)
    latents_std = np.asarray(vae_cfg["latents_std"], dtype=np.float32).reshape(1, int(vae_cfg["z_dim"]), 1, 1, 1)
    vae_latents = latents.detach().cpu().numpy().astype(np.float32) * latents_std + latents_mean
    if args.vae_backend == "phone":
        if vae_info is None:
            raise RuntimeError("VAE info missing from QNN manifest")
        video_shape = tuple(int(v) for v in vae_info["shapes"]["video"])
        vae_latents_path = host_io_dir / "vae_latents.raw"
        _write_raw(vae_latents_path, vae_latents.astype(np.float16, copy=False))
        _adb_push(adb, serial, vae_latents_path, f"{paths['inputs']}/vae_latents.raw")

        vae_remote_output_dir = f"{paths['outputs']}/vae"
        _clear_remote_dir(adb, serial, vae_remote_output_dir, label="clear vae output dir")
        vae_elapsed = _run_phone_context(
            adb,
            serial,
            phone_base=args.phone_base,
            context_path=deployed["vae"]["context"],
            input_list_path=vae_input_list_phone,
            output_dir=vae_remote_output_dir,
        )
        vae_local_output_dir = phone_pull_dir / "vae"
        _adb_pull(adb, serial, vae_remote_output_dir, vae_local_output_dir)
        vae_raw_path = _find_first_raw(vae_local_output_dir)
        video = np.fromfile(vae_raw_path, dtype=np.float16).reshape(video_shape).astype(np.float32)
    else:
        host_vae_device = _resolve_torch_device(args.host_vae_device)
        print(f"Host VAE device: {host_vae_device}")
        t0 = time.time()
        video = _decode_video_host(args.model_dir, vae_latents, device=host_vae_device)
        vae_elapsed = time.time() - t0

    video_fhwc = np.transpose(video[0], (1, 2, 3, 0))
    video_fhwc = np.clip(video_fhwc / 2.0 + 0.5, 0.0, 1.0)
    video_uint8 = np.clip(np.round(video_fhwc * 255.0), 0, 255).astype(np.uint8)

    artifacts, final_video_uint8 = _save_frames_and_video(
        video_uint8,
        run_dir,
        fps=args.fps,
        min_duration_sec=args.min_duration_sec,
        target_frames=args.target_frames,
        target_duration_sec=args.target_duration_sec,
        upsample_fill=args.upsample_fill,
    )
    actual_duration_sec = float(final_video_uint8.shape[0]) / float(args.fps)
    memory_report = _estimate_memory_report(
        latent_shape,
        tuple(int(v) for v in final_video_uint8.shape),
        fps=args.fps,
    )

    report = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "steps": args.steps,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": args.fps,
        "min_duration_sec": args.min_duration_sec,
        "target_frames": args.target_frames,
        "target_duration_sec": args.target_duration_sec,
        "upsample_fill": args.upsample_fill,
        "output_frames": int(final_video_uint8.shape[0]),
        "actual_duration_sec": actual_duration_sec,
        "flow_shift": flow_shift,
        "vae_backend": args.vae_backend,
        "adb_serial": serial,
        "phone_base": args.phone_base,
        "qnn_manifest": str(selected_manifest_path),
        "qnn_manifest_dir": str(args.qnn_manifest_dir) if args.qnn_manifest_dir else "",
        "requested_resolution": requested_resolution,
        "resolved_resolution": resolved_resolution,
        "manifest_resolution_exact_match": exact_resolution_match,
        "available_resolutions": available_resolutions,
        "step_times": step_times,
        "vae_seconds": vae_elapsed,
        "memory_estimate": memory_report,
        "artifacts": artifacts,
    }
    _save_json(run_dir / "run_report.json", report)

    print("\n[done]")
    print(f"Frames: {artifacts['frames_dir']}")
    if "mp4" in artifacts:
        print(f"MP4: {artifacts['mp4']}")
    if "video_warning" in artifacts:
        print(artifacts["video_warning"])
    print(
        "Memory estimate: "
        f"frame={memory_report['frame_rgb8_mib']:.3f} MiB, "
        f"output_rate={memory_report['output_rgb8_mib_per_second']:.3f} MiB/s, "
        f"latent_fp16={memory_report['latent_fp16_mib']:.3f} MiB"
    )
    print(f"Report: {run_dir / 'run_report.json'}")


if __name__ == "__main__":
    main()
