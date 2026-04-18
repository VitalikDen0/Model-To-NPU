from __future__ import annotations

import ctypes
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_ACCEL_CACHE: dict[tuple[bool, str], "RuntimeAccel"] = {}
_ACCEL_LIB_ENV_KEYS = (
    "SDXL_QNN_ACCEL_LIB",
    "QNN_ACCEL_LIB",
    "SDXL_QNN_RUNTIME_ACCEL_LIB",
)


def _platform_library_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["sdxl_runtime_accel.dll", "libsdxl_runtime_accel.dll"]
    if sys.platform == "darwin":
        return ["libsdxl_runtime_accel.dylib"]
    return ["libsdxl_runtime_accel.so", "sdxl_runtime_accel.so"]


def _candidate_dirs() -> list[Path]:
    base = Path(os.path.abspath(__file__)).parent
    dirs = [
        base / "lib",
        base,
        base / "phone_gen" / "lib",
        base / "phone_gen",
        base / "NPU" / "build" / "runtime_accel" / "windows-x64",
        base / "NPU" / "build" / "runtime_accel" / "linux-x86_64",
        base / "NPU" / "build" / "runtime_accel" / "android-arm64",
    ]
    if base.name == "phone_gen":
        dirs.insert(0, base / "lib")
    seen: set[str] = set()
    out: list[Path] = []
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            out.append(directory)
    return out


def _candidate_library_paths(explicit_path: str = "") -> list[str]:
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    for key in _ACCEL_LIB_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(value)

    for directory in _candidate_dirs():
        for name in _platform_library_names():
            candidates.append(str(directory / name))

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(candidate)
    return ordered


class RuntimeAccel:
    def __init__(self, prefer_native: bool = True, library_path: str = "") -> None:
        self.native_enabled = False
        self.library_path = ""
        self.load_error = ""
        self._lib = None
        self._dll_dirs: list[object] = []
        self._float_ptr = ctypes.POINTER(ctypes.c_float)
        if prefer_native:
            self._try_load(library_path)

    def _try_load(self, library_path: str) -> None:
        last_error = ""
        for candidate in _candidate_library_paths(library_path):
            if not os.path.exists(candidate):
                continue
            try:
                if os.name == "nt" and hasattr(os, "add_dll_directory"):
                    candidate_dir = os.path.dirname(candidate)
                    self._dll_dirs.append(os.add_dll_directory(candidate_dir))
                    gcc_path = shutil.which("gcc")
                    if gcc_path:
                        gcc_dir = os.path.dirname(gcc_path)
                        if gcc_dir and os.path.normcase(gcc_dir) != os.path.normcase(candidate_dir):
                            self._dll_dirs.append(os.add_dll_directory(gcc_dir))
                lib = ctypes.CDLL(candidate)
                self._configure(lib)
                self._lib = lib
                self.library_path = candidate
                self.native_enabled = True
                self.load_error = ""
                return
            except OSError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
        self.load_error = last_error or "native library not found"

    def _configure(self, lib) -> None:
        lib.sdxl_scale_model_input_f32.argtypes = [
            self._float_ptr,
            self._float_ptr,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.sdxl_scale_model_input_f32.restype = ctypes.c_int

        lib.sdxl_euler_step_f32.argtypes = [
            self._float_ptr,
            self._float_ptr,
            self._float_ptr,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.sdxl_euler_step_f32.restype = ctypes.c_int

        lib.sdxl_cfg_euler_step_f32.argtypes = [
            self._float_ptr,
            self._float_ptr,
            self._float_ptr,
            self._float_ptr,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.sdxl_cfg_euler_step_f32.restype = ctypes.c_int

        lib.sdxl_nchw_to_nhwc_scaled_f32.argtypes = [
            self._float_ptr,
            self._float_ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        lib.sdxl_nchw_to_nhwc_scaled_f32.restype = ctypes.c_int

    @staticmethod
    def _as_float32_c_array(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if not out.flags.c_contiguous:
            out = np.ascontiguousarray(out, dtype=np.float32)
        return out

    @staticmethod
    def _ensure_output_like(out: np.ndarray | None, template: np.ndarray) -> np.ndarray:
        if out is None:
            return np.empty_like(template, dtype=np.float32, order="C")
        candidate = np.asarray(out, dtype=np.float32)
        if (
            candidate.shape != template.shape
            or not candidate.flags.c_contiguous
            or not candidate.flags.writeable
        ):
            return np.empty_like(template, dtype=np.float32, order="C")
        return candidate

    @staticmethod
    def _ensure_output_shape(out: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        if out is None:
            return np.empty(shape, dtype=np.float32)
        candidate = np.asarray(out, dtype=np.float32)
        if (
            tuple(candidate.shape) != tuple(shape)
            or not candidate.flags.c_contiguous
            or not candidate.flags.writeable
        ):
            return np.empty(shape, dtype=np.float32)
        return candidate

    def describe(self) -> str:
        if self.native_enabled:
            return f"native:{Path(self.library_path).name}"
        if self.load_error:
            return f"numpy-fallback ({self.load_error})"
        return "numpy-fallback"

    def _call(self, func_name: str, *args) -> None:
        if not self.native_enabled or self._lib is None:
            raise RuntimeError("native runtime accelerator is not loaded")
        func = getattr(self._lib, func_name)
        rc = func(*args)
        if rc != 0:
            raise RuntimeError(f"{func_name} failed with code {rc}")

    def scale_model_input(self, sample: np.ndarray, sigma: float, out: np.ndarray | None = None) -> np.ndarray:
        sample_arr = self._as_float32_c_array(sample)
        out_arr = self._ensure_output_like(out, sample_arr)
        if self.native_enabled:
            self._call(
                "sdxl_scale_model_input_f32",
                sample_arr.ctypes.data_as(self._float_ptr),
                out_arr.ctypes.data_as(self._float_ptr),
                ctypes.c_size_t(sample_arr.size),
                ctypes.c_float(float(sigma)),
            )
            return out_arr

        inv = np.float32(1.0 / math.sqrt(float(sigma) * float(sigma) + 1.0))
        np.multiply(sample_arr, inv, out=out_arr)
        return out_arr

    def euler_step(
        self,
        model_output: np.ndarray,
        sample: np.ndarray,
        sigma: float,
        sigma_next: float,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        sample_arr = self._as_float32_c_array(sample)
        model_arr = self._as_float32_c_array(model_output)
        if sample_arr.shape != model_arr.shape:
            raise ValueError(f"shape mismatch: sample={sample_arr.shape}, model_output={model_arr.shape}")
        out_arr = self._ensure_output_like(out, sample_arr)
        if self.native_enabled:
            self._call(
                "sdxl_euler_step_f32",
                sample_arr.ctypes.data_as(self._float_ptr),
                model_arr.ctypes.data_as(self._float_ptr),
                out_arr.ctypes.data_as(self._float_ptr),
                ctypes.c_size_t(sample_arr.size),
                ctypes.c_float(float(sigma)),
                ctypes.c_float(float(sigma_next)),
            )
            return out_arr

        delta = np.float32(float(sigma_next) - float(sigma))
        np.multiply(model_arr, delta, out=out_arr)
        np.add(out_arr, sample_arr, out=out_arr)
        return out_arr

    def cfg_euler_step(
        self,
        cond: np.ndarray,
        uncond: np.ndarray,
        sample: np.ndarray,
        cfg_scale: float,
        sigma: float,
        sigma_next: float,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        cond_arr = self._as_float32_c_array(cond)
        uncond_arr = self._as_float32_c_array(uncond)
        sample_arr = self._as_float32_c_array(sample)
        if cond_arr.shape != uncond_arr.shape or cond_arr.shape != sample_arr.shape:
            raise ValueError(
                "shape mismatch for cfg_euler_step: "
                f"cond={cond_arr.shape}, uncond={uncond_arr.shape}, sample={sample_arr.shape}"
            )
        out_arr = self._ensure_output_like(out, sample_arr)
        if self.native_enabled:
            self._call(
                "sdxl_cfg_euler_step_f32",
                cond_arr.ctypes.data_as(self._float_ptr),
                uncond_arr.ctypes.data_as(self._float_ptr),
                sample_arr.ctypes.data_as(self._float_ptr),
                out_arr.ctypes.data_as(self._float_ptr),
                ctypes.c_size_t(sample_arr.size),
                ctypes.c_float(float(cfg_scale)),
                ctypes.c_float(float(sigma)),
                ctypes.c_float(float(sigma_next)),
            )
            return out_arr

        delta = np.float32(float(sigma_next) - float(sigma))
        np.subtract(cond_arr, uncond_arr, out=out_arr)
        np.multiply(out_arr, np.float32(cfg_scale), out=out_arr)
        np.add(out_arr, uncond_arr, out=out_arr)
        np.multiply(out_arr, delta, out=out_arr)
        np.add(out_arr, sample_arr, out=out_arr)
        return out_arr

    def nchw_to_nhwc_scaled(
        self,
        src: np.ndarray,
        scale: float,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        src_arr = self._as_float32_c_array(src)
        if src_arr.ndim != 4:
            raise ValueError(f"nchw_to_nhwc_scaled expects rank-4 input, got {src_arr.shape}")
        n, c, h, w = src_arr.shape
        out_shape = (n, h, w, c)
        out_arr = self._ensure_output_shape(out, out_shape)
        if self.native_enabled:
            self._call(
                "sdxl_nchw_to_nhwc_scaled_f32",
                src_arr.ctypes.data_as(self._float_ptr),
                out_arr.ctypes.data_as(self._float_ptr),
                ctypes.c_int(n),
                ctypes.c_int(c),
                ctypes.c_int(h),
                ctypes.c_int(w),
                ctypes.c_float(float(scale)),
            )
            return out_arr

        out_arr[...] = np.moveaxis(src_arr, 1, -1)
        if scale != 1.0:
            np.multiply(out_arr, np.float32(scale), out=out_arr)
        return out_arr


class RuntimeTensorArena:
    def __init__(self, accel: RuntimeAccel) -> None:
        self.accel = accel
        self._model_input: np.ndarray | None = None
        self._step_buffers: list[np.ndarray | None] = [None, None]
        self._step_toggle = 0
        self._vae_input: np.ndarray | None = None
        self._timestep = np.empty((1,), dtype=np.float32)

    @staticmethod
    def _ensure_like(buf: np.ndarray | None, template: np.ndarray) -> np.ndarray:
        template_arr = np.asarray(template, dtype=np.float32)
        if (
            buf is None
            or buf.shape != template_arr.shape
            or buf.dtype != np.float32
            or not buf.flags.c_contiguous
            or not buf.flags.writeable
        ):
            return np.empty_like(template_arr, dtype=np.float32, order="C")
        return buf

    @staticmethod
    def _ensure_shape(buf: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        if (
            buf is None
            or tuple(buf.shape) != tuple(shape)
            or buf.dtype != np.float32
            or not buf.flags.c_contiguous
            or not buf.flags.writeable
        ):
            return np.empty(shape, dtype=np.float32)
        return buf

    def describe(self) -> str:
        return self.accel.describe()

    def scale_model_input(self, latents: np.ndarray, sigma: float) -> np.ndarray:
        latents_arr = np.asarray(latents, dtype=np.float32)
        self._model_input = self._ensure_like(self._model_input, latents_arr)
        return self.accel.scale_model_input(latents_arr, sigma, out=self._model_input)

    def _next_step_buffer(self, sample: np.ndarray) -> np.ndarray:
        sample_arr = np.asarray(sample, dtype=np.float32)
        idx = self._step_toggle
        self._step_toggle ^= 1
        buf = self._ensure_like(self._step_buffers[idx], sample_arr)
        self._step_buffers[idx] = buf
        return buf

    def step(self, model_output: np.ndarray, sample: np.ndarray, sigma: float, sigma_next: float) -> np.ndarray:
        out = self._next_step_buffer(sample)
        return self.accel.euler_step(model_output, sample, sigma, sigma_next, out=out)

    def step_cfg(
        self,
        cond: np.ndarray,
        uncond: np.ndarray,
        sample: np.ndarray,
        cfg_scale: float,
        sigma: float,
        sigma_next: float,
    ) -> np.ndarray:
        out = self._next_step_buffer(sample)
        return self.accel.cfg_euler_step(cond, uncond, sample, cfg_scale, sigma, sigma_next, out=out)

    def timestep_tensor(self, timestep: int | float) -> np.ndarray:
        self._timestep[0] = np.float32(timestep)
        return self._timestep

    def vae_input(self, latents: np.ndarray, scaling_factor: float) -> np.ndarray:
        latents_arr = np.asarray(latents, dtype=np.float32)
        out_shape = (latents_arr.shape[0], latents_arr.shape[2], latents_arr.shape[3], latents_arr.shape[1])
        self._vae_input = self._ensure_shape(self._vae_input, out_shape)
        return self.accel.nchw_to_nhwc_scaled(latents_arr, 1.0 / float(scaling_factor), out=self._vae_input)


def get_runtime_accel(prefer_native: bool = True, library_path: str = "") -> RuntimeAccel:
    key = (bool(prefer_native), os.path.abspath(library_path) if library_path else "")
    accel = _ACCEL_CACHE.get(key)
    if accel is None:
        accel = RuntimeAccel(prefer_native=prefer_native, library_path=library_path)
        _ACCEL_CACHE[key] = accel
    return accel


__all__ = [
    "RuntimeAccel",
    "RuntimeTensorArena",
    "get_runtime_accel",
]
