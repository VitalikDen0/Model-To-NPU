#!/data/data/com.termux/files/usr/bin/python3
"""
SDXL Lightning — standalone phone generation.
Runs entirely on phone: tokenizer + scheduler + QNN NPU inference.
No PC, no torch, no diffusers required.

Usage (in Termux):
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "1girl, anime, cherry blossoms"
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "cat on windowsill" --seed 777
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "dark castle" --cfg 2.0 --neg "blurry, bad"
"""
import argparse
import atexit
import hashlib
import json
import importlib
import math
import os
import random
import re
import select
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time

import numpy as np

# ─── Paths ───
DEFAULT_BASE_DIR = "/sdcard/Download/sdxl_qnn"
LEGACY_BASE_DIR = "/data/local/tmp/sdxl_qnn"


def _dir_is_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".sdxl_write_test")
        with open(probe, "wb") as f:
            f.write(b"ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _resolve_base_dir():
    override = os.environ.get("SDXL_QNN_BASE")
    if override:
        return override

    candidates = [
        DEFAULT_BASE_DIR,
        "/storage/emulated/0/Download/sdxl_qnn",
        LEGACY_BASE_DIR,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_BASE_DIR


def _resolve_work_dir() -> str:
    override = os.environ.get("SDXL_QNN_WORK_DIR")
    if override:
        return override

    candidates: list[str] = []
    if not DR.startswith(LEGACY_BASE_DIR):
        candidates.extend([
            "/data/data/com.termux/files/usr/tmp/sdxl_qnn_work",
            "/data/local/tmp/sdxl_qnn_work",
        ])
    candidates.append(f"{DR}/phone_gen/work")

    for candidate in candidates:
        if _dir_is_writable(candidate):
            return candidate
    return candidates[-1]


def _env_first(keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            return value
    return default


def _env_bool(keys: tuple[str, ...], default: bool) -> bool:
    raw = _env_first(keys, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(keys: tuple[str, ...], default: int = 0) -> int:
    raw = _env_first(keys, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


DR = _resolve_base_dir()
CONTEXTS = {
    "clip_l": f"{DR}/context/clip_l.serialized.bin.bin",
    "clip_g": f"{DR}/context/clip_g.serialized.bin.bin",
    "encoder": f"{DR}/context/unet_encoder_fp16.serialized.bin.bin",
    "decoder": f"{DR}/context/unet_decoder_fp16.serialized.bin.bin",
    "vae": f"{DR}/context/vae_decoder.serialized.bin.bin",
}
TOKENIZER_DIR = f"{DR}/phone_gen/tokenizer"
OUTPUT_DIR = os.environ.get("SDXL_QNN_OUTPUT_DIR", f"{DR}/outputs")
WORK_DIR = _resolve_work_dir()
QNN_NET_RUN = os.environ.get("SDXL_QNN_NET_RUN", f"{DR}/bin/qnn-net-run")
QNN_BIN_DIR = os.environ.get("SDXL_QNN_BIN_DIR", os.path.dirname(QNN_NET_RUN))
QNN_LIB = os.environ.get("SDXL_QNN_LIB_DIR", f"{DR}/lib")
QNN_MODEL_DIR = os.environ.get("SDXL_QNN_MODEL_DIR", f"{DR}/model")
PREVIEW_PNG = os.environ.get("SDXL_QNN_PREVIEW_PNG", f"{OUTPUT_DIR}/preview_current.png")
TAESD_ONNX = os.environ.get("SDXL_QNN_TAESD_ONNX", f"{DR}/phone_gen/taesd_decoder.onnx")
TAESD_CONTEXT = os.environ.get("SDXL_QNN_TAESD_CONTEXT", f"{DR}/context/taesd_decoder.serialized.bin.bin")
TAESD_MODEL = os.environ.get("SDXL_QNN_TAESD_MODEL", f"{QNN_MODEL_DIR}/libTAESDDecoder.so")
DEFAULT_QNN_EXT_LIB = f"{QNN_LIB}/libQnnHtpNetRunExtensions.so"
DEFAULT_QNN_CONFIG_PATH = f"{DR}/htp_backend_extensions_lightning.json"
QNN_CONTEXT_RUNNER = os.environ.get("SDXL_QNN_DAEMON_BIN", f"{QNN_BIN_DIR}/qnn-context-runner")
QNN_SYSTEM_LIB = os.environ.get("SDXL_QNN_SYSTEM_LIB", f"{QNN_LIB}/libQnnSystem.so")
DEFAULT_TAESD_QNN_NET_RUN = QNN_NET_RUN


def _detect_default_qnn_config() -> str:
    if os.path.exists(DEFAULT_QNN_CONFIG_PATH) and os.path.exists(DEFAULT_QNN_EXT_LIB):
        return DEFAULT_QNN_CONFIG_PATH
    return ""

# ─── Cached helpers / cross-thread logging ───

_QNN_ENV: dict = {}
_PRINT_LOCK = threading.Lock()
_ort_session = None
_ort_avail: bool | None = None
_preview_thread: threading.Thread | None = None
SHOW_TEMP = os.environ.get("SDXL_SHOW_TEMP", "0") == "1"
TEMP_POLL_INTERVAL = max(0.2, float(os.environ.get("SDXL_TEMP_INTERVAL_SEC", "1.0")))
QNN_LOG_LEVEL = os.environ.get("SDXL_QNN_LOG_LEVEL", "warn")
QNN_PROFILING_LEVEL = os.environ.get("SDXL_QNN_PROFILING_LEVEL", "").strip()
QNN_PROFILE_ARCHIVE = os.environ.get("SDXL_QNN_PROFILE_ARCHIVE", "0") == "1"
QNN_PROFILE_ARCHIVE_DIR = os.environ.get("SDXL_QNN_PROFILE_ARCHIVE_DIR", os.path.join(WORK_DIR, "qnn_profiles"))
QNN_PROFILE_VIEWER = os.environ.get("SDXL_QNN_PROFILE_VIEWER", f"{QNN_BIN_DIR}/qnn-profile-viewer").strip()
QNN_USE_MMAP = os.environ.get("SDXL_QNN_USE_MMAP", "1") == "1"
QNN_STDOUT_ECHO = os.environ.get("SDXL_QNN_STDOUT_ECHO", "0") == "1"
QNN_PERF_PROFILE = os.environ.get("SDXL_QNN_PERF_PROFILE", "sustained_high_performance").strip() or "sustained_high_performance"
QNN_CONFIG_FILE = os.environ.get("SDXL_QNN_CONFIG_FILE", _detect_default_qnn_config()).strip()
QNN_USE_DAEMON = _env_bool(("SDXL_QNN_USE_DAEMON", "QNN_USE_DAEMON"), False)
QNN_DAEMON_CONTEXT_SCOPE = _env_first(("SDXL_QNN_DAEMON_CONTEXT_SCOPE", "QNN_DAEMON_CONTEXT_SCOPE"), "unet").strip().lower() or "unet"
QNN_DAEMON_PREWARM = _env_bool(("SDXL_QNN_DAEMON_PREWARM", "QNN_DAEMON_PREWARM"), True)
QNN_DAEMON_CONFIG_FILE = _env_first(("SDXL_QNN_DAEMON_CONFIG_FILE", "QNN_DAEMON_CONFIG_FILE"), QNN_CONFIG_FILE).strip()
QNN_HVX_THREADS = max(0, _env_int(("SDXL_QNN_HVX_THREADS", "QNN_HVX_THREADS"), 0))
QNN_VTCM_MB = max(0, _env_int(("SDXL_QNN_VTCM_MB", "QNN_VTCM_MB"), 0))
QNN_CLIP_CACHE = os.environ.get("SDXL_QNN_CLIP_CACHE", "1") == "1"
QNN_CLIP_CACHE_DIR = os.environ.get("SDXL_QNN_CLIP_CACHE_DIR", f"{DR}/phone_gen/cache/clip")
PREVIEW_PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get("SDXL_QNN_PREVIEW_PNG_COMPRESS", "0"))))
FINAL_PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get("SDXL_QNN_FINAL_PNG_COMPRESS", "1"))))
STRETCH_SAMPLE_STRIDE = max(1, int(os.environ.get("SDXL_QNN_STRETCH_SAMPLE_STRIDE", "4")))
TAESD_BACKEND = os.environ.get("SDXL_QNN_TAESD_BACKEND", "gpu").strip().lower() or "gpu"
TAESD_BACKEND_LIB = os.environ.get("SDXL_QNN_TAESD_BACKEND_LIB", "").strip()
TAESD_CONFIG_FILE = os.environ.get("SDXL_QNN_TAESD_CONFIG_FILE", "").strip()
TAESD_QNN_NET_RUN = os.environ.get("SDXL_QNN_TAESD_NET_RUN", DEFAULT_TAESD_QNN_NET_RUN).strip() or QNN_NET_RUN
TAESD_FORCE_ONNX = os.environ.get("SDXL_QNN_TAESD_FORCE_ONNX", "0") == "1"
_TEMP_SENSOR_CACHE: dict[str, list[tuple[str, str]]] | None = None
_TEMP_MONITOR_STOP: threading.Event | None = None
_TEMP_MONITOR_THREAD: threading.Thread | None = None
_QNN_DAEMONS: dict[tuple[str, bool], "_QnnContextDaemon"] = {}
_EXEC_BIN_CACHE: dict[str, str] = {}
_RUNTIME_FILE_CACHE: dict[str, str] = {}
_SHARED_RUNTIME_STAGED = False
_TAESD_QNN_CHECKED = False
_TAESD_QNN_PLAN: dict | None = None
_TAESD_QNN_FAILED = False
_QNN_PROFILE_INDEX = 0
_QNN_PROFILE_LOCK = threading.Lock()
_CLIP_CACHE_NAMESPACE: str | None = None
_RESOLVED_CONFIG_CACHE: dict[tuple[str, int, int], str] = {}


def _log(line: str = "") -> None:
    with _PRINT_LOCK:
        print(line, flush=True)


def _sanitize_profile_tag(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(tag)).strip("._") or "profile"


def _artifact_signature(path: str) -> str:
    if not path or not os.path.exists(path):
        return f"missing:{path}"
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns}"


def _get_clip_cache_namespace() -> str:
    global _CLIP_CACHE_NAMESPACE
    if _CLIP_CACHE_NAMESPACE is not None:
        return _CLIP_CACHE_NAMESPACE

    parts = [
        "clip-cache-v1",
        _artifact_signature(CONTEXTS["clip_l"]),
        _artifact_signature(CONTEXTS["clip_g"]),
        _artifact_signature(f"{TOKENIZER_DIR}/vocab.json"),
        _artifact_signature(f"{TOKENIZER_DIR}/merges.txt"),
        _artifact_signature(QNN_CONFIG_FILE) if QNN_CONFIG_FILE else "no-config",
        _artifact_signature(DEFAULT_QNN_EXT_LIB),
    ]
    _CLIP_CACHE_NAMESPACE = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return _CLIP_CACHE_NAMESPACE


def _clip_cache_prefix(text: str) -> str:
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "text": text,
                "namespace": _get_clip_cache_namespace(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return os.path.join(QNN_CLIP_CACHE_DIR, cache_key)


def _write_atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def _write_atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_clip_cache(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not QNN_CLIP_CACHE:
        return None

    prefix = _clip_cache_prefix(text)
    pe_path = prefix + ".pe.raw"
    te_path = prefix + ".te.raw"
    meta_path = prefix + ".json"
    if not (os.path.exists(pe_path) and os.path.exists(te_path) and os.path.exists(meta_path)):
        return None

    try:
        pe = np.fromfile(pe_path, np.float32).reshape(1, 77, 2048)
        te = np.fromfile(te_path, np.float32).reshape(1, 1280)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("namespace") != _get_clip_cache_namespace() or meta.get("text") != text:
            return None
        return pe, te
    except Exception:
        return None


def _save_clip_cache(text: str, pe: np.ndarray, te: np.ndarray) -> None:
    if not QNN_CLIP_CACHE:
        return

    prefix = _clip_cache_prefix(text)
    _write_atomic_bytes(prefix + ".pe.raw", pe.astype(np.float32, copy=False).tobytes())
    _write_atomic_bytes(prefix + ".te.raw", te.astype(np.float32, copy=False).tobytes())
    _write_atomic_json(
        prefix + ".json",
        {
            "text": text,
            "namespace": _get_clip_cache_namespace(),
            "pe_shape": [1, 77, 2048],
            "te_shape": [1, 1280],
        },
    )


def _archive_qnn_profile(output_dir: str, profile_tag: str | None) -> dict | None:
    global _QNN_PROFILE_INDEX
    if not QNN_PROFILING_LEVEL:
        return None

    prof_log = os.path.join(output_dir, "qnn-profiling-data_0.log")
    if not os.path.exists(prof_log):
        return None

    info = {"log_path": prof_log}
    if not QNN_PROFILE_ARCHIVE:
        return info

    os.makedirs(QNN_PROFILE_ARCHIVE_DIR, exist_ok=True)
    with _QNN_PROFILE_LOCK:
        _QNN_PROFILE_INDEX += 1
        idx = _QNN_PROFILE_INDEX
    stem = f"{idx:03d}_{_sanitize_profile_tag(profile_tag or os.path.basename(output_dir))}"
    archived_log = os.path.join(QNN_PROFILE_ARCHIVE_DIR, f"{stem}.log")
    shutil.copy2(prof_log, archived_log)
    info["archived_log"] = archived_log

    if os.path.exists(QNN_PROFILE_VIEWER):
        try:
            viewer_result = subprocess.run(
                [QNN_PROFILE_VIEWER, "--input_log", prof_log],
                env=_get_qnn_env(),
                cwd=DR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            viewer_stdout = (viewer_result.stdout or "").strip()
            if viewer_result.returncode == 0 and viewer_stdout:
                viewer_path = os.path.join(QNN_PROFILE_ARCHIVE_DIR, f"{stem}.txt")
                with open(viewer_path, "w", encoding="utf-8") as f:
                    f.write(viewer_stdout)
                info["viewer_path"] = viewer_path
            elif viewer_result.stderr:
                info["viewer_error"] = viewer_result.stderr.strip()
        except Exception as e:
            info["viewer_error"] = str(e)
    return info


def _get_qnn_env() -> dict:
    if not _QNN_ENV:
        _ensure_shared_runtime_assets()
        _QNN_ENV.update(os.environ)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        staged_runtime_lib = os.path.join(WORK_DIR, "runtime", "lib")
        _QNN_ENV["LD_LIBRARY_PATH"] = (
            f"{staged_runtime_lib}:{QNN_LIB}:{QNN_BIN_DIR}:{QNN_MODEL_DIR}"
            + (f":{existing_ld}" if existing_ld else "")
        )
        _QNN_ENV["ADSP_LIBRARY_PATH"] = (
            f"{staged_runtime_lib};{QNN_LIB};/vendor/lib64/rfs/dsp;"
            f"/vendor/lib/rfsa/adsp;/vendor/dsp"
        )
    return _QNN_ENV


def _is_shared_storage_path(path: str | None) -> bool:
    if not path:
        return False
    norm = path.replace("\\", "/")
    return norm.startswith("/sdcard/") or norm.startswith("/storage/emulated/")


def _ensure_shared_runtime_assets() -> None:
    global _SHARED_RUNTIME_STAGED
    if _SHARED_RUNTIME_STAGED or not _is_shared_storage_path(QNN_LIB):
        return

    runtime_root = os.path.join(WORK_DIR, "runtime")
    staged_lib_dir = os.path.join(runtime_root, "lib")
    os.makedirs(staged_lib_dir, exist_ok=True)

    try:
        for name in os.listdir(QNN_LIB):
            src = os.path.join(QNN_LIB, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(staged_lib_dir, name)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
    except Exception as e:
        _log(f"  [QNN] runtime lib staging warning: {e}")
    _SHARED_RUNTIME_STAGED = True


def _resolve_runtime_artifact(path: str, category: str) -> str:
    cached = _RUNTIME_FILE_CACHE.get(path)
    if cached and os.path.exists(cached):
        return cached
    if not _is_shared_storage_path(path):
        _RUNTIME_FILE_CACHE[path] = path
        return path

    dst_dir = os.path.join(WORK_DIR, "runtime", category)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(path))
    try:
        needs_copy = (
            not os.path.exists(dst)
            or os.path.getsize(dst) != os.path.getsize(path)
            or os.path.getmtime(dst) < os.path.getmtime(path)
        )
    except Exception:
        needs_copy = True
    if needs_copy:
        shutil.copy2(path, dst)
        if category in ("bin", "lib"):
            os.chmod(dst, 0o755)
    _RUNTIME_FILE_CACHE[path] = dst
    return dst


def _resolve_qnn_config_path(config_path: str) -> str:
    if not config_path:
        return config_path

    cache_key = (config_path, QNN_HVX_THREADS, QNN_VTCM_MB)
    cached = _RESOLVED_CONFIG_CACHE.get(cache_key)
    if cached and os.path.exists(cached):
        return cached

    needs_override = QNN_HVX_THREADS > 0 or QNN_VTCM_MB > 0
    if not _is_shared_storage_path(config_path) and not needs_override:
        _RESOLVED_CONFIG_CACHE[cache_key] = config_path
        return config_path

    runtime_root = os.path.join(WORK_DIR, "runtime")
    os.makedirs(runtime_root, exist_ok=True)
    dst = os.path.join(runtime_root, os.path.basename(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    shared_ext_cfg = os.path.join(os.path.dirname(config_path), "htp_backend_ext_config_lightning.json")
    staged_ext_cfg = os.path.join(runtime_root, os.path.basename(shared_ext_cfg))
    ext_cfg_payload = None
    if os.path.exists(shared_ext_cfg):
        try:
            with open(shared_ext_cfg, "r", encoding="utf-8") as f:
                ext_cfg_payload = json.load(f)
        except Exception:
            ext_cfg_payload = None

    if ext_cfg_payload is not None and isinstance(ext_cfg_payload.get("graphs"), list):
        for graph in ext_cfg_payload["graphs"]:
            if not isinstance(graph, dict):
                continue
            if QNN_HVX_THREADS > 0:
                graph["hvx_threads"] = QNN_HVX_THREADS
            if QNN_VTCM_MB > 0:
                graph["vtcm_mb"] = QNN_VTCM_MB

    if ext_cfg_payload is not None:
        with open(staged_ext_cfg, "w", encoding="utf-8") as f:
            json.dump(ext_cfg_payload, f, indent=4)
    elif os.path.exists(shared_ext_cfg):
        shutil.copy2(shared_ext_cfg, staged_ext_cfg)

    shared_ext_lib = os.path.join(QNN_LIB, "libQnnHtpNetRunExtensions.so")
    staged_ext_lib = _resolve_runtime_artifact(shared_ext_lib, "lib") if os.path.exists(shared_ext_lib) else ""

    backend_ext = config_data.get("backend_extensions")
    if isinstance(backend_ext, dict):
        if staged_ext_lib:
            backend_ext["shared_library_path"] = staged_ext_lib
        if os.path.exists(staged_ext_cfg):
            backend_ext["config_file_path"] = staged_ext_cfg

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    _RESOLVED_CONFIG_CACHE[cache_key] = dst
    return dst


def _resolve_exec_binary(path: str) -> str:
    cached = _EXEC_BIN_CACHE.get(path)
    if cached and os.path.exists(cached):
        return cached

    if os.path.exists(path) and os.access(path, os.X_OK):
        _EXEC_BIN_CACHE[path] = path
        return path

    dst = os.path.join(WORK_DIR, "bin", os.path.basename(path))
    try:
        needs_copy = (
            not os.path.exists(dst)
            or os.path.getsize(dst) != os.path.getsize(path)
            or os.path.getmtime(dst) < os.path.getmtime(path)
        )
    except Exception:
        needs_copy = True
    if needs_copy:
        shutil.copy2(path, dst)
    os.chmod(dst, 0o755)
    _EXEC_BIN_CACHE[path] = dst
    return dst


def _resolve_backend_lib(backend_name_or_path: str | None) -> str:
    if not backend_name_or_path:
        return f"{QNN_LIB}/libQnnHtp.so"

    candidate = backend_name_or_path.strip()
    if os.path.isabs(candidate) or "/" in candidate or "\\" in candidate:
        return candidate

    mapping = {
        "cpu": "libQnnCpu.so",
        "dsp": "libQnnDsp.so",
        "gpu": "libQnnGpu.so",
        "hta": "libQnnHta.so",
        "htp": "libQnnHtp.so",
        "lpai": "libQnnLpai.so",
    }
    lib_name = mapping.get(candidate.lower())
    if lib_name is None:
        raise ValueError(f"unsupported QNN backend: {backend_name_or_path}")
    return f"{QNN_LIB}/{lib_name}"


def _backend_label(backend_lib: str) -> str:
    base = os.path.basename(backend_lib).lower()
    if "gpu" in base:
        return "GPU"
    if "htp" in base:
        return "HTP"
    if "dsp" in base:
        return "DSP"
    if "cpu" in base:
        return "CPU"
    if "hta" in base:
        return "HTA"
    if "lpai" in base:
        return "LPAI"
    return os.path.basename(backend_lib)


def _is_htp_backend(backend_lib: str) -> bool:
    return "htp" in os.path.basename(backend_lib).lower()


def _get_taesd_qnn_plan() -> dict | None:
    global _TAESD_QNN_CHECKED, _TAESD_QNN_PLAN
    if _TAESD_QNN_CHECKED:
        return _TAESD_QNN_PLAN

    _TAESD_QNN_CHECKED = True
    if TAESD_FORCE_ONNX or TAESD_BACKEND in ("", "off", "none", "onnx", "cpu"):
        return None

    try:
        backend_lib = TAESD_BACKEND_LIB or _resolve_backend_lib(TAESD_BACKEND)
    except Exception as e:
        _log(f"  [TAESD] invalid QNN backend '{TAESD_BACKEND}': {e}")
        return None

    if not os.path.exists(backend_lib):
        return None

    if os.path.exists(TAESD_CONTEXT):
        _TAESD_QNN_PLAN = {
            "mode": "context",
            "ctx_path": TAESD_CONTEXT,
            "backend_lib": backend_lib,
            "backend_label": _backend_label(backend_lib),
        }
        return _TAESD_QNN_PLAN

    if os.path.exists(TAESD_MODEL):
        _TAESD_QNN_PLAN = {
            "mode": "model",
            "model_path": TAESD_MODEL,
            "backend_lib": backend_lib,
            "backend_label": _backend_label(backend_lib),
        }
        return _TAESD_QNN_PLAN

    return None


def _prepare_preview_backend() -> None:
    plan = None if _TAESD_QNN_FAILED else _get_taesd_qnn_plan()
    if plan is not None:
        location = plan.get("ctx_path") or plan.get("model_path") or ""
        _log(
            f"  [TAESD] preview backend ready: QNN {plan['backend_label']} "
            f"({plan['mode']}, {os.path.basename(location)})"
        )
        return

    if os.path.exists(TAESD_ONNX):
        _get_ort_session()
        return

    if not TAESD_FORCE_ONNX and TAESD_BACKEND not in ("", "off", "none", "onnx", "cpu"):
        backend_lib = TAESD_BACKEND_LIB or TAESD_BACKEND
        _log(
            f"  [TAESD] no QNN preview artifacts for backend '{backend_lib}' — "
            "falling back to ONNX CPU if available"
        )
    else:
        _log("  [TAESD] no preview backend available")


def _get_ort_session():
    global _ort_session, _ort_avail
    if _ort_avail is False:
        return None
    if _ort_session is not None:
        return _ort_session

    if not os.path.exists(TAESD_ONNX):
        _ort_avail = False
        _log(f"  [TAESD] ONNX not found: {TAESD_ONNX}")
        _log("  [TAESD] Export: python SDXL/debug/export_taesd_to_onnx.py --validate")
        return None

    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        _ort_avail = False
        _log("  [TAESD] onnxruntime not found — run in Termux: pip install onnxruntime")
        return None

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        _ort_session = ort.InferenceSession(
            TAESD_ONNX,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _ort_avail = True
        _log("  [TAESD] onnxruntime session ready (CPU)")
    except Exception as e:
        _ort_avail = False
        _log(f"  [TAESD] ort session failed: {e}")
    return _ort_session


def _prime_ctx_bg(paths: list[str]) -> list[threading.Thread]:
    def _read(path: str):
        try:
            with open(path, "rb") as f:
                while f.read(8 * 1024 * 1024):
                    pass
        except Exception:
            pass

    threads = [
        threading.Thread(target=_read, args=(p,), daemon=True)
        for p in paths if os.path.exists(p)
    ]
    for t in threads:
        t.start()
    return threads


def _write_input_list_once(path: str, entries: list[str] | tuple[str, ...]) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(entries) + "\n")


def _write_multi_input_list_once(path: str, rows: list[list[str]] | tuple[tuple[str, ...], ...]) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


def _write_array_reuse(path: str, arr: np.ndarray, dtype=np.float32) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    target_size = len(payload)
    try:
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(payload)
            if os.fstat(f.fileno()).st_size != target_size:
                f.truncate(target_size)
    except FileNotFoundError:
        with open(path, "wb") as f:
            f.write(payload)


def _stretch_sample_view(img: np.ndarray) -> np.ndarray:
    if STRETCH_SAMPLE_STRIDE <= 1:
        return img
    if img.shape[0] < 512 or img.shape[1] < 512:
        return img
    return img[::STRETCH_SAMPLE_STRIDE, ::STRETCH_SAMPLE_STRIDE]


def _match_temp_group(zone_type: str) -> str | None:
    zt = zone_type.lower()
    if zt.startswith("cpu-") or zt.startswith("cpuss-"):
        return "CPU"
    if zt.startswith("gpuss-") or zt.startswith("gpu") or "kgsl" in zt:
        return "GPU"
    if zt.startswith("nsphvx-") or zt.startswith("nsphmx-") or zt.startswith("nsp"):
        return "NPU"
    return None


def _normalize_temp(raw_value: str) -> float | None:
    try:
        value = float(raw_value.strip())
    except Exception:
        return None
    if value <= -100:
        return None
    if abs(value) >= 1000:
        value /= 1000.0
    if value <= 0:
        return None
    return value


def _discover_temp_sensors() -> dict[str, list[tuple[str, str]]]:
    global _TEMP_SENSOR_CACHE
    if _TEMP_SENSOR_CACHE is not None:
        return _TEMP_SENSOR_CACHE

    sensors: dict[str, list[tuple[str, str]]] = {"CPU": [], "GPU": [], "NPU": []}
    try:
        entries = sorted(os.listdir("/sys/class/thermal"))
    except Exception:
        _TEMP_SENSOR_CACHE = sensors
        return sensors

    for entry in entries:
        if not entry.startswith("thermal_zone"):
            continue
        zone_dir = f"/sys/class/thermal/{entry}"
        type_path = f"{zone_dir}/type"
        temp_path = f"{zone_dir}/temp"
        try:
            with open(type_path, "r", encoding="utf-8") as f:
                zone_type = f.read().strip()
        except Exception:
            continue
        group = _match_temp_group(zone_type)
        if group:
            sensors[group].append((zone_type, temp_path))

    _TEMP_SENSOR_CACHE = sensors
    return sensors


def _phone_thermal_snapshot() -> dict[str, float]:
    sensors = _discover_temp_sensors()
    temps: dict[str, float] = {}

    for group in ("CPU", "GPU", "NPU"):
        hottest_temp: float | None = None
        for _, temp_path in sensors.get(group, []):
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    temp_c = _normalize_temp(f.read())
            except Exception:
                continue
            if temp_c is None:
                continue
            if hottest_temp is None or temp_c > hottest_temp:
                hottest_temp = temp_c
        if hottest_temp is not None:
            temps[group] = hottest_temp
    return temps


def _phone_temp_summary() -> str:
    temps = _phone_thermal_snapshot()
    parts = [
        f"{group}={temps[group]:.1f}°C"
        for group in ("CPU", "GPU", "NPU")
        if group in temps
    ]
    return " ".join(parts)


def _temp_monitor_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        summary = _phone_temp_summary()
        if summary:
            _log(f"  [TEMP] {summary}")
        if stop_event.wait(TEMP_POLL_INTERVAL):
            break


def _start_temp_monitor() -> None:
    global _TEMP_MONITOR_STOP, _TEMP_MONITOR_THREAD
    if not SHOW_TEMP:
        return
    _stop_temp_monitor()
    _TEMP_MONITOR_STOP = threading.Event()
    _TEMP_MONITOR_THREAD = threading.Thread(
        target=_temp_monitor_loop,
        args=(_TEMP_MONITOR_STOP,),
        daemon=True,
    )
    _TEMP_MONITOR_THREAD.start()


def _stop_temp_monitor(timeout: float = 2.0) -> None:
    global _TEMP_MONITOR_STOP, _TEMP_MONITOR_THREAD
    if _TEMP_MONITOR_STOP is not None:
        _TEMP_MONITOR_STOP.set()
    if _TEMP_MONITOR_THREAD is not None and _TEMP_MONITOR_THREAD.is_alive():
        _TEMP_MONITOR_THREAD.join(timeout=timeout)
    _TEMP_MONITOR_STOP = None
    _TEMP_MONITOR_THREAD = None

# ─── CLIP BPE Tokenizer (pure Python, no dependencies) ───

def _bytes_to_unicode():
    """Same mapping as openai/clip."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class CLIPTokenizer:
    """Minimal CLIP BPE tokenizer — works without transformers/tokenizers."""

    def __init__(self, vocab_path, merges_path, pad_token_id=49407):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        # Skip header line
        merges = [tuple(line.split()) for line in lines[1:]]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u0500-\u052F]+|[0-9]|[^\s\w]+""",
            re.IGNORECASE,
        )
        self.bos_id = self.encoder.get("<|startoftext|>", 49406)
        self.eos_id = self.encoder.get("<|endoftext|>", 49407)
        self.pad_id = pad_token_id

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _tokenize(self, text):
        text = text.lower().strip()
        tokens = []
        matches = self.pat.findall(text)
        for token in matches:
            if token in ("<|startoftext|>", "<|endoftext|>"):
                tokens.append(token)
                continue
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = self._bpe(encoded).split(" ")
            tokens.extend(bpe_tokens)
        return tokens

    def encode(self, text, max_length=77):
        """Encode text to token IDs with BOS/EOS/padding."""
        bpe_tokens = self._tokenize(text)
        ids = [self.bos_id]
        for t in bpe_tokens:
            if t in self.encoder:
                ids.append(self.encoder[t])
            else:
                ids.append(self.eos_id)  # unk → eos
        ids.append(self.eos_id)
        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        # Pad
        while len(ids) < max_length:
            ids.append(self.pad_id)
        return ids


# ─── Euler Discrete Scheduler (pure numpy) ───

class EulerDiscreteScheduler:
    """Minimal EulerDiscreteScheduler for SDXL Lightning."""

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085,
                 beta_end=0.012, beta_schedule="scaled_linear",
                 prediction_type="epsilon", steps_offset=1):
        self.num_train = num_train_timesteps
        self.prediction_type = prediction_type
        self.steps_offset = steps_offset

        if beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps,
                                dtype=np.float64) ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # Precompute sigmas for all timesteps
        self._all_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.timesteps = np.array([], dtype=np.int64)
        self.sigmas = np.array([], dtype=np.float32)
        self.init_noise_sigma = 1.0

    def set_timesteps(self, num_steps):
        """Set timesteps with trailing spacing (used by Lightning)."""
        # Trailing spacing: np.round(np.arange(N, 0, -N/steps)) - 1
        step_ratio = self.num_train / num_steps
        timesteps = np.round(np.arange(self.num_train, 0, -step_ratio)).astype(np.int64) - 1

        sigmas = np.array([self._all_sigmas[t] for t in timesteps], dtype=np.float64)
        sigmas = np.append(sigmas, 0.0)

        self.timesteps = timesteps
        self.sigmas = sigmas.astype(np.float32)
        self.init_noise_sigma = float(max(self.sigmas))

    def scale_model_input(self, sample, step_index):
        """Scale input by sigma (Karras schedule)."""
        sigma = self.sigmas[step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output, step_index, sample):
        """Single Euler step."""
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        if self.prediction_type == "epsilon":
            pred_x0 = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = sigma * sample + (1 - sigma**2)**0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Euler method
        d = (sample - pred_x0) / sigma
        prev_sample = sample + (sigma_next - sigma) * d
        return prev_sample


# ─── QNN Net Run helper ───


def _can_use_qnn_daemon() -> bool:
    return QNN_USE_DAEMON and os.path.exists(QNN_CONTEXT_RUNNER) and os.path.exists(QNN_SYSTEM_LIB)


def _daemon_supports_context(ctx_path: str | None) -> bool:
    if not ctx_path:
        return False
    if QNN_DAEMON_CONTEXT_SCOPE == "all":
        return True
    if QNN_DAEMON_CONTEXT_SCOPE in ("", "none", "off", "0"):
        return False

    name = os.path.basename(ctx_path)
    if QNN_DAEMON_CONTEXT_SCOPE == "unet":
        return name in {
            "unet_encoder_fp16.serialized.bin.bin",
            "unet_decoder_fp16.serialized.bin.bin",
        }

    allowed = {part.strip() for part in QNN_DAEMON_CONTEXT_SCOPE.split(",") if part.strip()}
    return name in allowed


def _prewarm_qnn_daemons(ctx_paths: list[str]) -> list[threading.Thread]:
    threads: list[threading.Thread] = []

    def _start(path: str) -> None:
        try:
            _get_qnn_daemon(path, native=False).start()
        except Exception as e:
            _log(f"  [QNN daemon] prewarm failed for {os.path.basename(path)}: {e}")

    for ctx_path in ctx_paths:
        if not _daemon_supports_context(ctx_path):
            continue
        t = threading.Thread(target=_start, args=(ctx_path,), daemon=True)
        t.start()
        threads.append(t)
    return threads


class _QnnContextDaemon:
    def __init__(self, ctx_path: str, native: bool) -> None:
        self.ctx_path = ctx_path
        self.native = native
        self.name = re.sub(r"[^0-9A-Za-z_.-]+", "_", os.path.basename(ctx_path))
        self.daemon_dir = os.path.join(WORK_DIR, "daemon")
        self.req_fifo = os.path.join(self.daemon_dir, f"{self.name}.req")
        self.rsp_fifo = os.path.join(self.daemon_dir, f"{self.name}.rsp")
        self.bootstrap_output_dir = os.path.join(self.daemon_dir, f"{self.name}.boot")
        self.proc: subprocess.Popen[str] | None = None
        self.stderr_tail: list[str] = []
        self.stderr_thread: threading.Thread | None = None

    def _capture_stderr(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        try:
            for raw_line in self.proc.stderr:
                line = raw_line.rstrip()
                if not line:
                    continue
                self.stderr_tail.append(line)
                if len(self.stderr_tail) > 40:
                    self.stderr_tail = self.stderr_tail[-40:]
                if QNN_STDOUT_ECHO:
                    _log(f"  [daemon {self.name}] {line}")
        except Exception:
            pass

    def _error_tail(self) -> str:
        if not self.stderr_tail:
            return ""
        return " | ".join(self.stderr_tail[-5:])

    def _ensure_fifo(self, path: str) -> None:
        mkfifo = getattr(os, "mkfifo", None)
        if mkfifo is None:
            raise RuntimeError("os.mkfifo is unavailable on this platform")
        if os.path.exists(path):
            if not stat.S_ISFIFO(os.stat(path).st_mode):
                os.remove(path)
        if not os.path.exists(path):
            mkfifo(path)

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        os.makedirs(self.daemon_dir, exist_ok=True)
        os.makedirs(self.bootstrap_output_dir, exist_ok=True)
        for fifo in (self.req_fifo, self.rsp_fifo):
            self._ensure_fifo(fifo)

        runner_path = _resolve_exec_binary(QNN_CONTEXT_RUNNER)
        backend_lib = _resolve_runtime_artifact(f"{QNN_LIB}/libQnnHtp.so", "lib")
        system_lib = _resolve_runtime_artifact(QNN_SYSTEM_LIB, "lib")
        daemon_config = QNN_DAEMON_CONFIG_FILE
        effective_config = _resolve_qnn_config_path(daemon_config) if daemon_config else ""

        cmd = [
            runner_path,
            "--retrieve_context", self.ctx_path,
            "--backend", backend_lib,
            "--system_library", system_lib,
            "--request_fifo", self.req_fifo,
            "--response_fifo", self.rsp_fifo,
            "--output_dir", self.bootstrap_output_dir,
            "--log_level", QNN_LOG_LEVEL,
        ]
        if effective_config:
            cmd.extend(["--config_file", effective_config])
        if self.native:
            cmd.append("--use_native_output_files")
        if QNN_PROFILING_LEVEL:
            cmd.extend(["--profiling_level", QNN_PROFILING_LEVEL])

        self.proc = subprocess.Popen(
            cmd,
            env=_get_qnn_env(),
            cwd=DR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stderr_thread = threading.Thread(target=self._capture_stderr, daemon=True)
        self.stderr_thread.start()
        time.sleep(0.15)
        if self.proc.poll() is not None:
            raise RuntimeError(f"runner exited early: {self._error_tail()}")

    def run(self, input_list_path: str, output_dir: str, timeout: float = 120.0) -> float:
        self.start()
        assert self.proc is not None
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()
        with open(self.req_fifo, "w", encoding="utf-8") as req:
            req.write(f"{input_list_path}\n{output_dir}\n")
            req.flush()

        fd = os.open(self.rsp_fifo, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            response = b""
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"runner exited: {self._error_tail()}")
                remaining = timeout - (time.time() - t0)
                if remaining <= 0:
                    raise TimeoutError(f"runner timeout after {timeout:.1f}s")
                readable, _, _ = select.select([fd], [], [], min(0.2, remaining))
                if not readable:
                    continue
                chunk = os.read(fd, 4096)
                if not chunk:
                    continue
                response += chunk
                if b"\n" not in response:
                    continue
                line = response.splitlines()[0].decode("utf-8", errors="replace").strip()
                if line == "OK":
                    return (time.time() - t0) * 1000
                raise RuntimeError(f"{line} ({self._error_tail()})")
        finally:
            os.close(fd)

    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                with open(self.req_fifo, "w", encoding="utf-8") as req:
                    req.write("__quit__\n\n")
                    req.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
        self.proc = None


def _get_qnn_daemon(ctx_path: str, native: bool) -> _QnnContextDaemon:
    key = (ctx_path, native)
    daemon = _QNN_DAEMONS.get(key)
    if daemon is None:
        daemon = _QnnContextDaemon(ctx_path, native=native)
        _QNN_DAEMONS[key] = daemon
    return daemon


def _shutdown_qnn_daemons() -> None:
    for daemon in list(_QNN_DAEMONS.values()):
        try:
            daemon.stop()
        except Exception:
            pass
    _QNN_DAEMONS.clear()


atexit.register(_shutdown_qnn_daemons)

def qnn_run(ctx_path, input_list_path, output_dir, native=False, *,
            native_input=False, backend=None, model_path=None, config_file=None,
            use_mmap=None, perf_profile=None, net_run_path=None, profile_tag=None):
    """Run QNN context on NPU via qnn-net-run."""
    if ctx_path is None and model_path is None:
        raise ValueError("qnn_run needs either ctx_path or model_path")
    if ctx_path is not None and model_path is not None:
        raise ValueError("qnn_run accepts ctx_path or model_path, not both")

    backend_lib = _resolve_backend_lib(backend or "htp")
    backend_lib = _resolve_runtime_artifact(backend_lib, "lib")
    effective_config = QNN_CONFIG_FILE if config_file is None and _is_htp_backend(backend_lib) else (config_file or "")
    effective_config = _resolve_qnn_config_path(effective_config) if effective_config else ""
    effective_mmap = QNN_USE_MMAP if use_mmap is None else use_mmap
    effective_perf = perf_profile or QNN_PERF_PROFILE
    runner_path = _resolve_exec_binary(net_run_path or QNN_NET_RUN)

    if (
        ctx_path is not None
        and model_path is None
        and net_run_path is None
        and _is_htp_backend(backend_lib)
        and _can_use_qnn_daemon()
        and _daemon_supports_context(ctx_path)
    ):
        try:
            daemon = _get_qnn_daemon(ctx_path, native=native)
            return daemon.run(input_list_path, output_dir, timeout=120.0)
        except Exception as e:
            _log(f"  [QNN daemon] fallback for {os.path.basename(ctx_path)}: {e}")
            daemon = _QNN_DAEMONS.pop((ctx_path, native), None)
            if daemon is not None:
                daemon.stop()

    os.makedirs(output_dir, exist_ok=True)
    cmd = [runner_path]
    if ctx_path is not None:
        cmd.extend(["--retrieve_context", ctx_path])
    else:
        assert model_path is not None
        model_path = _resolve_runtime_artifact(model_path, "model")
        cmd.extend(["--model", model_path])
    cmd.extend([
        "--backend", backend_lib,
        "--input_list", input_list_path,
        "--output_dir", output_dir,
        "--perf_profile", effective_perf,
        "--log_level", QNN_LOG_LEVEL,
    ])
    if QNN_PROFILING_LEVEL:
        cmd.extend(["--profiling_level", QNN_PROFILING_LEVEL])
    if effective_config:
        cmd.extend(["--config_file", effective_config])
    if effective_mmap and ctx_path is not None:
        cmd.append("--use_mmap")
    if native_input:
        cmd.append("--use_native_input_files")
    if native:
        cmd.append("--use_native_output_files")
    t0 = time.time()
    stdout_target = subprocess.PIPE if QNN_STDOUT_ECHO else subprocess.DEVNULL
    result = subprocess.run(
        cmd,
        env=_get_qnn_env(),
        cwd=DR,
        stdout=stdout_target,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    elapsed = (time.time() - t0) * 1000
    if result.returncode != 0:
        print(f"  [qnn-net-run ERROR] {result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"qnn-net-run failed: exit {result.returncode}")
    if QNN_STDOUT_ECHO and result.stdout and result.stdout.strip():
        _log(result.stdout.rstrip())
    if QNN_PROFILING_LEVEL:
        prof_info = _archive_qnn_profile(output_dir, profile_tag)
        if prof_info is not None:
            log_target = prof_info.get("archived_log") or prof_info.get("log_path")
            _log(f"  [QNN profile] {log_target}")
    return elapsed


# ─── Main generation pipeline ───

DEFAULT_NEG = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry"

def generate(prompt, seed=None, steps=8, cfg_scale=3.5, neg_prompt=None,
             stretch=True, name=None, preview=False, progressive_cfg=False):
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PREVIEW_PNG), exist_ok=True)
    use_cfg = cfg_scale > 1.0
    if neg_prompt is None:
        neg_prompt = DEFAULT_NEG if use_cfg else ""

    ctx_prime_threads = _prime_ctx_bg([
        CONTEXTS["encoder"],
        CONTEXTS["decoder"],
        CONTEXTS["vae"],
    ])
    daemon_prewarm_threads: list[threading.Thread] = []
    if _can_use_qnn_daemon() and QNN_DAEMON_PREWARM:
        daemon_prewarm_threads = _prewarm_qnn_daemons([
            CONTEXTS["encoder"],
            CONTEXTS["decoder"],
        ])

    _log(f"Prompt: {prompt}")
    _log(f"Base:   {DR}")
    _log(f"Work:   {WORK_DIR}")
    _log(f"Out:    {OUTPUT_DIR}")
    qnn_mode = (
        f"QNN:    perf={QNN_PERF_PROFILE}, mmap={'on' if QNN_USE_MMAP else 'off'}, "
        f"log={QNN_LOG_LEVEL}"
    )
    if _can_use_qnn_daemon():
        qnn_mode += f", daemon={os.path.basename(QNN_CONTEXT_RUNNER)}"
    else:
        qnn_mode += ", daemon=off"
    if QNN_CONFIG_FILE:
        qnn_mode += f", config={os.path.basename(QNN_CONFIG_FILE)}"
    if QNN_HVX_THREADS > 0:
        qnn_mode += f", hvx={QNN_HVX_THREADS}"
    if QNN_VTCM_MB > 0:
        qnn_mode += f", vtcm={QNN_VTCM_MB}"
    if QNN_PROFILING_LEVEL:
        qnn_mode += f", profiling={QNN_PROFILING_LEVEL}"
    _log(qnn_mode)
    if use_cfg:
        _log(f"Neg:    {neg_prompt[:80]}{'...' if len(neg_prompt) > 80 else ''}")
    _log(f"Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")
    t_total = time.time()
    _start_temp_monitor()

    # ── Load tokenizers ──
    tok_l = CLIPTokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt",
        pad_token_id=49407,  # CLIP-L: pad = <|endoftext|>
    )
    tok_g = CLIPTokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt",
        pad_token_id=0,  # CLIP-G: pad = "!"
    )

    # ── Setup scheduler ──
    sched = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        steps_offset=1,
    )
    sched.set_timesteps(steps)

    # ── 1. CLIP ──
    def run_clip(text, tag):
        cache_t0 = time.time()
        cached = _load_clip_cache(text)
        if cached is not None:
            pe, te = cached
            return pe, te, (time.time() - cache_t0) * 1000, 0.0, True

        ids_l = np.array(tok_l.encode(text, 77), dtype=np.float32).reshape(1, 77)
        ids_g = np.array(tok_g.encode(text, 77), dtype=np.float32).reshape(1, 77)

        cd = f"{WORK_DIR}/clip/{tag}"
        os.makedirs(cd, exist_ok=True)
        ids_l.tofile(f"{cd}/ids_l.raw")
        ids_g.tofile(f"{cd}/ids_g.raw")

        _write_input_list_once(f"{cd}/il_l.txt", [f"{cd}/ids_l.raw"])
        _write_input_list_once(f"{cd}/il_g.txt", [f"{cd}/ids_g.raw"])

        ms_l = qnn_run(CONTEXTS["clip_l"], f"{cd}/il_l.txt", f"{cd}/out_l", profile_tag=f"clip_{tag}_l")
        ms_g = qnn_run(CONTEXTS["clip_g"], f"{cd}/il_g.txt", f"{cd}/out_g", profile_tag=f"clip_{tag}_g")

        cl = np.fromfile(f"{cd}/out_l/Result_0/penultimate_hidden.raw", np.float32).reshape(1, 77, 768)
        cg = np.fromfile(f"{cd}/out_g/Result_0/penultimate_hidden.raw", np.float32).reshape(1, 77, 1280)
        te = np.fromfile(f"{cd}/out_g/Result_0/text_embeds.raw", np.float32).reshape(1, 1280)

        pe = np.concatenate([cl, cg], axis=-1)  # [1, 77, 2048]
        _save_clip_cache(text, pe, te)
        return pe, te, ms_l, ms_g, False

    pe_uncond = None
    te_uncond = None
    pe_cond, te_cond, ms_l, ms_g, clip_cond_cached = run_clip(prompt, "cond")
    cond_suffix = " (cache)" if clip_cond_cached else ""
    _log(f"[CLIP cond{cond_suffix}] L={ms_l:.0f}ms G={ms_g:.0f}ms")
    ms_clip = ms_l + ms_g
    if use_cfg:
        pe_uncond, te_uncond, ms_l2, ms_g2, clip_uncond_cached = run_clip(neg_prompt, "uncond")
        uncond_suffix = " (cache)" if clip_uncond_cached else ""
        _log(f"[CLIP uncond{uncond_suffix}] L={ms_l2:.0f}ms G={ms_g2:.0f}ms")
        ms_clip += ms_l2 + ms_g2

    for t in ctx_prime_threads:
        t.join(timeout=0.5)
    for t in daemon_prewarm_threads:
        t.join(timeout=15.0)

    if preview:
        _prepare_preview_backend()

    # ── 2. Prepare UNet inputs ──
    def push_unet_data(pe, te, tag):
        """Save encoder_hidden_states, text_embeds, time_ids for a condition."""
        d = f"{WORK_DIR}/unet/{tag}"
        os.makedirs(d, exist_ok=True)

        # encoder_hidden_states [1,77,2048] — standard layout for AI Hub split
        enc = pe.astype(np.float32)
        enc.tofile(f"{d}/enc.raw")

        # text_embeds [1,1280]
        te.astype(np.float32).tofile(f"{d}/te.raw")

        # time_ids [1,6] — fixed 1024x1024
        tid = np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=np.float32)
        tid.tofile(f"{d}/tid.raw")

    push_unet_data(pe_cond, te_cond, "cond")
    if use_cfg:
        assert pe_uncond is not None and te_uncond is not None
        push_unet_data(pe_uncond, te_uncond, "uncond")

    # ── 3. Denoise loop ──
    rng = np.random.RandomState(seed)
    # Generate latents matching torch's randn with the same seed
    latents = rng.randn(1, 4, 128, 128).astype(np.float32)
    latents = latents * sched.init_noise_sigma
    total_unet_ms = 0

    # Pre-create a single reusable work dir (avoids mkdir overhead per step)
    _ensure_unet_workdirs(use_cfg)

    cfg_cutoff = ((steps + 1) // 2) if (use_cfg and progressive_cfg) else steps
    if progressive_cfg and use_cfg:
        _log(f"  [Progressive CFG] CFG on steps 1..{cfg_cutoff}, uncond-only after")

    for si in range(steps):
        t = sched.timesteps[si]
        lat_in = sched.scale_model_input(latents, si)

        step_uses_cfg = use_cfg and (si < cfg_cutoff)

        if step_uses_cfg:
            # Batched CFG: encoder runs cond+uncond in ONE subprocess call,
            # decoder does the same — saves 2 subprocess launches per step.
            np_cond, np_uncond, ms = _run_unet_split_cfg(
                lat_in, t,
                f"{WORK_DIR}/unet/cond",
                f"{WORK_DIR}/unet/uncond",
                si,
            )
            noise_pred = np_uncond + cfg_scale * (np_cond - np_uncond)
        else:
            noise_pred, ms = _run_unet_split(lat_in, t, si, "cond")

        total_unet_ms += ms

        temp_str = ""
        if SHOW_TEMP:
            temp_summary = _phone_temp_summary()
            if temp_summary:
                temp_str = f" [{temp_summary}]"

        cfg_str = " CFG" if step_uses_cfg else ""
        _log(
            f"  [UNet {si+1}/{steps}]{cfg_str}{temp_str} "
            f"{ms:.0f}ms [{noise_pred.min():.2f}..{noise_pred.max():.2f}]"
        )

        latents = sched.step(noise_pred, si, latents)

        if preview:
            _start_bg_preview(latents.copy(), si, steps)

    if preview:
        _join_preview_thread()

    _log(f"  UNet total: {total_unet_ms:.0f}ms ({total_unet_ms/steps:.0f}ms/step)")

    # ── 4. VAE decode ──
    scaling_factor = 0.13025
    lat_sc = latents / scaling_factor

    vd = f"{WORK_DIR}/vae"
    os.makedirs(vd, exist_ok=True)
    # VAE expects NHWC
    lat_nhwc = np.transpose(lat_sc, (0, 2, 3, 1)).astype(np.float32)
    lat_nhwc.tofile(f"{vd}/lat.raw")
    _write_input_list_once(f"{vd}/il.txt", [f"{vd}/lat.raw"])

    ms_vae = qnn_run(CONTEXTS["vae"], f"{vd}/il.txt", f"{vd}/out", native=True, profile_tag="vae_final")
    _log(f"[VAE] {ms_vae:.0f}ms")

    raw = np.fromfile(f"{vd}/out/Result_0/image_native.raw", np.float16).astype(np.float32)
    expected = 1024 * 1024 * 3
    if raw.size != expected:
        raise ValueError(f"VAE output: expected {expected} elements, got {raw.size}")
    img = raw.reshape(1024, 1024, 3)
    img = np.clip(img / 2 + 0.5, 0, 1)

    if stretch:
        stretch_ref = _stretch_sample_view(img)
        lo, hi = np.percentile(stretch_ref, [0.5, 99.5])
        if hi - lo > 0.05:
            img = np.clip((img - lo) / (hi - lo), 0, 1)

    img_u8 = (img * 255).astype(np.uint8)
    _stop_temp_monitor()

    # ── 5. Save ──
    from PIL import Image
    tag = name or f"gen_s{seed}"
    out_path = f"{OUTPUT_DIR}/{tag}.png"
    Image.fromarray(img_u8).save(
        out_path,
        format="PNG",
        compress_level=FINAL_PNG_COMPRESS_LEVEL,
        optimize=False,
    )

    elapsed = time.time() - t_total
    _log(f"\n{'=' * 40}")
    _log(f"Saved: {out_path}")
    _log(f"CLIP: {ms_clip:.0f}ms | UNet: {total_unet_ms:.0f}ms | VAE: {ms_vae:.0f}ms")
    _log(f"Total: {elapsed:.1f}s")
    return out_path


def _preview_step(latents: np.ndarray, step_idx: int, total_steps: int) -> None:
    global _TAESD_QNN_FAILED

    plan = None if _TAESD_QNN_FAILED else _get_taesd_qnn_plan()
    if plan is not None:
        try:
            ms = _preview_step_qnn(latents, plan)
            _log(
                f"  [PREVIEW step {step_idx+1}/{total_steps}] "
                f"QNN {plan['backend_label']} {ms:.0f}ms"
            )
            return
        except Exception as e:
            _TAESD_QNN_FAILED = True
            _log(f"  [TAESD] QNN preview fallback to ONNX CPU: {e}")

    sess = _get_ort_session()
    if sess is None:
        return

    t0 = time.time()
    try:
        result = sess.run(None, {"latents": latents.astype(np.float32, copy=False)})
    except Exception as e:
        _log(f"  [TAESD] inference error: {e}")
        return

    _save_preview_png(result[0])

    ms = (time.time() - t0) * 1000
    _log(f"  [PREVIEW step {step_idx+1}/{total_steps}] CPU {ms:.0f}ms")


def _preview_tensor_to_hwc(out_tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(out_tensor)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"TAESD preview expects batch=1, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"TAESD preview expects rank-3/4 tensor, got {arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        img = arr
    elif arr.shape[0] in (1, 3, 4):
        img = arr.transpose(1, 2, 0)
    else:
        raise ValueError(f"TAESD preview cannot infer tensor layout from {arr.shape}")

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[-1] > 3:
        img = img[:, :, :3]
    return img.astype(np.float32, copy=False)


def _normalize_preview_image(img: np.ndarray) -> np.ndarray:
    lo = float(np.min(img))
    hi = float(np.max(img))

    if lo >= -0.05 and hi <= 1.05:
        return np.clip(img, 0.0, 1.0)

    if lo >= -1.05 and hi <= 1.05:
        return np.clip(img / 2.0 + 0.5, 0.0, 1.0)

    return np.clip(img, 0.0, 1.0)


def _save_preview_png(out_tensor: np.ndarray) -> None:
    img = _preview_tensor_to_hwc(out_tensor)
    img = _normalize_preview_image(img)
    img_u8 = (img * 255).astype(np.uint8)

    from PIL import Image

    tmp_path = PREVIEW_PNG + ".tmp"
    Image.fromarray(img_u8).save(
        tmp_path,
        format="PNG",
        compress_level=PREVIEW_PNG_COMPRESS_LEVEL,
        optimize=False,
    )
    os.replace(tmp_path, PREVIEW_PNG)
    try:
        os.chmod(PREVIEW_PNG, 0o644)
    except Exception:
        pass


def _load_qnn_raw_tensor(raw_path: str, dims: list[int] | tuple[int, ...], dtype) -> np.ndarray:
    data = np.fromfile(raw_path, dtype)
    expected = math.prod(dims)
    if data.size != expected:
        raise ValueError(
            f"TAESD QNN output size mismatch for {os.path.basename(raw_path)}: "
            f"expected {expected} elements, got {data.size}"
        )
    return data.reshape(dims).astype(np.float32, copy=False)


def _read_taesd_qnn_output(out_dir: str) -> np.ndarray:
    result_dir = f"{out_dir}/Result_0"
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"TAESD QNN output dir missing: {result_dir}")

    native_path = os.path.join(result_dir, "image_native.raw")
    native_meta_path = os.path.join(out_dir, "image_native.raw.json")
    if os.path.exists(native_path):
        dims = [1, 1024, 1024, 3]
        dtype = np.float16
        if os.path.exists(native_meta_path):
            with open(native_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_dims = meta.get("Dimensions") or meta.get("dimensions")
            if isinstance(meta_dims, list) and len(meta_dims) == 4:
                dims = [int(v) for v in meta_dims]
            dtype_name = str(meta.get("Datatype") or meta.get("datatype") or "").upper()
            if "FLOAT_32" in dtype_name:
                dtype = np.float32
        return _load_qnn_raw_tensor(native_path, dims, dtype)

    raw_path = os.path.join(result_dir, "image.raw")
    if os.path.exists(raw_path):
        dims = [1, 1024, 1024, 3]
        raw_bytes = os.path.getsize(raw_path)
        elem_count = math.prod(dims)
        if raw_bytes == elem_count * 4:
            return _load_qnn_raw_tensor(raw_path, dims, np.float32)
        if raw_bytes == elem_count * 2:
            return _load_qnn_raw_tensor(raw_path, dims, np.float16)
        raise ValueError(f"TAESD QNN output has unexpected size: {raw_bytes} bytes")

    raise FileNotFoundError(f"TAESD QNN output file missing in {result_dir}")


def _preview_step_qnn(latents: np.ndarray, plan: dict) -> float:
    base = f"{WORK_DIR}/preview_qnn"
    out_dir = f"{base}/out"
    os.makedirs(base, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    lat_path = f"{base}/latents.raw"
    il_path = f"{base}/il.txt"
    np.transpose(latents, (0, 2, 3, 1)).astype(np.float16, copy=False).tofile(lat_path)
    _write_input_list_once(il_path, [lat_path])

    ms = qnn_run(
        plan.get("ctx_path"),
        il_path,
        out_dir,
        native=True,
        native_input=True,
        backend=plan["backend_lib"],
        model_path=plan.get("model_path"),
        config_file=TAESD_CONFIG_FILE or "",
        use_mmap=(plan["mode"] == "context"),
        net_run_path=TAESD_QNN_NET_RUN,
        profile_tag="taesd_preview",
    )
    _save_preview_png(_read_taesd_qnn_output(out_dir))
    return ms


def _start_bg_preview(latents_copy: np.ndarray, step_idx: int, total_steps: int) -> None:
    global _preview_thread
    if _preview_thread and _preview_thread.is_alive():
        _preview_thread.join(timeout=0.2)
        if _preview_thread.is_alive():
            _log(f"  [PREVIEW step {step_idx+1}/{total_steps}] skipped (previous decode still running)")
            return
    _preview_thread = threading.Thread(
        target=_preview_step,
        args=(latents_copy, step_idx, total_steps),
        daemon=True,
    )
    _preview_thread.start()


def _join_preview_thread(timeout: float = 30.0) -> None:
    global _preview_thread
    if _preview_thread and _preview_thread.is_alive():
        _preview_thread.join(timeout=timeout)


def _ensure_unet_workdirs(use_cfg):
    """Pre-create all work directories so per-step mkdir is skipped."""
    for tag in (["cond", "uncond"] if use_cfg else ["cond"]):
        os.makedirs(f"{WORK_DIR}/unet/{tag}/out_enc", exist_ok=True)
        os.makedirs(f"{WORK_DIR}/unet/{tag}/out_dec", exist_ok=True)
    if use_cfg:
        os.makedirs(f"{WORK_DIR}/unet/enc_batch", exist_ok=True)
        os.makedirs(f"{WORK_DIR}/unet/dec_batch", exist_ok=True)


def _enc_dec_inputs(base, smp_path, ts_path):
    """Return (enc_entries, dec_entries_builder) for one condition."""
    enc = [
        f"{base}/enc.raw",   # [0] encoder_hidden_states
        ts_path,              # [1] timestep
        f"{base}/tid.raw",   # [2] time_ids
        f"{base}/te.raw",    # [3] text_embeds
        smp_path,             # [4] sample
    ]
    return enc


def _dec_entries_from_enc_out(base, enc_out_dir):
    """Build decoder input list from encoder output directory."""
    dec = [
        f"{base}/enc.raw",               # [0] encoder_hidden_states
        f"{enc_out_dir}/output_0.raw",    # [1] mid_out
        f"{enc_out_dir}/output_9.raw",    # [2] skip_8
        f"{enc_out_dir}/output_10.raw",   # [3] temb
    ]
    for i in range(8, 0, -1):
        dec.append(f"{enc_out_dir}/output_{i}.raw")  # skip_7→skip_0
    return dec


def _read_noise_pred(out_dec_dir, result_idx=0):
    """Read decoder noise_pred output (auto-detect float32/float16)."""
    out_path = f"{out_dec_dir}/Result_{result_idx}/output_0.raw"
    raw_bytes = os.path.getsize(out_path)
    expected_f32 = 1 * 4 * 128 * 128 * 4
    expected_f16 = 1 * 4 * 128 * 128 * 2
    if raw_bytes == expected_f32:
        d = np.fromfile(out_path, np.float32)
    elif raw_bytes == expected_f16:
        d = np.fromfile(out_path, np.float16).astype(np.float32)
    else:
        raise ValueError(f"Decoder output: unexpected {raw_bytes} bytes at {out_path}")
    return d.reshape(1, 4, 128, 128)


def _run_unet_split(latent_np, timestep, step_idx, tag):
    """Run split UNet (encoder + decoder) on NPU — single condition."""
    # Reuse a fixed work dir (step_idx ignored — files overwritten each step)
    base = f"{WORK_DIR}/unet/{tag}"
    enc_out = f"{base}/out_enc"
    dec_out = f"{base}/out_dec"

    smp_path = f"{base}/smp.raw"
    ts_path = f"{base}/ts.raw"

    _write_array_reuse(smp_path, latent_np, dtype=np.float32)
    _write_array_reuse(ts_path, np.array([float(timestep)], dtype=np.float32), dtype=np.float32)

    enc_entries = _enc_dec_inputs(base, smp_path, ts_path)
    il_enc = f"{base}/il_enc.txt"
    _write_input_list_once(il_enc, enc_entries)

    ms_enc = qnn_run(CONTEXTS["encoder"], il_enc, enc_out, profile_tag=f"unet_step{step_idx+1:02d}_{tag}_enc")

    dec_entries = _dec_entries_from_enc_out(base, f"{enc_out}/Result_0")
    il_dec = f"{base}/il_dec.txt"
    _write_input_list_once(il_dec, dec_entries)

    ms_dec = qnn_run(CONTEXTS["decoder"], il_dec, dec_out, profile_tag=f"unet_step{step_idx+1:02d}_{tag}_dec")

    return _read_noise_pred(dec_out, 0), ms_enc + ms_dec


def _run_unet_split_cfg(latent_np, timestep, cond_base, uncond_base, step_idx):
    """Optimized CFG: run cond+uncond as 2-line input_list in ONE subprocess each.

    Instead of 4 qnn-net-run calls, we make 2:
      - encoder: processes uncond then cond in one process
      - decoder: same
    Outputs land in Result_0/ (uncond) and Result_1/ (cond).
    """
    # Common files (same latent/timestep for both conditions)
    smp_path = f"{cond_base}/smp.raw"
    ts_path = f"{cond_base}/ts.raw"
    smp_np = latent_np.astype(np.float32)
    ts_np = np.array([float(timestep)], dtype=np.float32)
    _write_array_reuse(smp_path, smp_np, dtype=np.float32)
    _write_array_reuse(ts_path, ts_np, dtype=np.float32)
    # uncond uses the same values — write them there too
    _write_array_reuse(f"{uncond_base}/smp.raw", smp_np, dtype=np.float32)

    enc_out_batch = f"{WORK_DIR}/unet/enc_batch"
    dec_out_batch = f"{WORK_DIR}/unet/dec_batch"

    # ── Batched Encoder: 2 inferences in one qnn-net-run ──
    # Line 0 → uncond (Result_0), Line 1 → cond (Result_1)
    enc_uncond = _enc_dec_inputs(uncond_base, f"{uncond_base}/smp.raw", ts_path)
    enc_cond = _enc_dec_inputs(cond_base, smp_path, ts_path)

    il_enc_batch = f"{cond_base}/il_enc_batch.txt"
    _write_multi_input_list_once(il_enc_batch, [enc_uncond, enc_cond])

    # Single shared output dir; Result_0 = uncond, Result_1 = cond
    ms_enc = qnn_run(CONTEXTS["encoder"], il_enc_batch, enc_out_batch, profile_tag=f"unet_step{step_idx+1:02d}_cfg_batch_enc")

    # ── Batched Decoder: 2 inferences in one qnn-net-run ──
    dec_uncond = _dec_entries_from_enc_out(uncond_base, f"{enc_out_batch}/Result_0")
    dec_cond = _dec_entries_from_enc_out(cond_base, f"{enc_out_batch}/Result_1")

    il_dec_batch = f"{cond_base}/il_dec_batch.txt"
    _write_multi_input_list_once(il_dec_batch, [dec_uncond, dec_cond])

    dec_out_batch = f"{WORK_DIR}/unet/dec_batch"
    ms_dec = qnn_run(CONTEXTS["decoder"], il_dec_batch, dec_out_batch, profile_tag=f"unet_step{step_idx+1:02d}_cfg_batch_dec")

    np_cond = _read_noise_pred(dec_out_batch, 1)
    np_uncond = _read_noise_pred(dec_out_batch, 0)

    return np_cond, np_uncond, ms_enc + ms_dec


# ─── CLI ───

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL Lightning — Phone NPU (standalone)")
    ap.add_argument("prompt", type=str, help="Text prompt")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed (omit for a fresh random seed each run)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=3.5,
                    help="CFG scale (1.0=off, 3.5=default for NPU quality)")
    ap.add_argument("--neg", type=str, default=None,
                    help="Negative prompt (auto-set if --cfg > 1.0)")
    ap.add_argument("--no-stretch", action="store_true",
                    help="Disable contrast stretching")
    ap.add_argument("--name", type=str, default=None,
                    help="Output filename (without .png)")
    ap.add_argument("--preview", action="store_true",
                    help="Decode each step with TAESD live preview (prefers QNN GPU if deployed, otherwise ONNX CPU)")
    ap.add_argument("--prog-cfg", action="store_true",
                    help="Apply CFG only on the first ceil(steps/2) denoising steps")
    a = ap.parse_args()

    generate(
        a.prompt, seed=a.seed, steps=a.steps,
        cfg_scale=a.cfg, neg_prompt=a.neg,
        stretch=not a.no_stretch, name=a.name,
        preview=a.preview, progressive_cfg=a.prog_cfg,
    )
