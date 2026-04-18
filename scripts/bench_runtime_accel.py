#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(Path(os.path.abspath(__file__))).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phone_runtime_accel import RuntimeAccel  # noqa: E402


def benchmark(label: str, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    ms_per_iter = (elapsed * 1000.0) / iters
    print(f"{label:<24} {ms_per_iter:>10.4f} ms/iter")
    return ms_per_iter


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic benchmarks for the optional SDXL runtime accelerator")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--library", default="", help="Explicit path to sdxl runtime accelerator library")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((1, 4, 128, 128), dtype=np.float32)
    cond = rng.standard_normal((1, 4, 128, 128), dtype=np.float32)
    uncond = rng.standard_normal((1, 4, 128, 128), dtype=np.float32)
    sigma = 14.6146
    sigma_next = 9.2113
    cfg_scale = 3.5
    scaling_factor = 0.13025

    native = RuntimeAccel(prefer_native=True, library_path=args.library)
    numpy_only = RuntimeAccel(prefer_native=False)

    print("Runtime accelerator benchmark")
    print(f"Native backend: {native.describe()}")
    if args.require_native and not native.native_enabled:
        raise SystemExit("Native accelerator was required but not loaded")

    scale_np = numpy_only.scale_model_input(latents, sigma)
    step_np = numpy_only.euler_step(cond, latents, sigma, sigma_next)
    cfg_step_np = numpy_only.cfg_euler_step(cond, uncond, latents, cfg_scale, sigma, sigma_next)
    nhwc_np = numpy_only.nchw_to_nhwc_scaled(latents, 1.0 / scaling_factor)

    if native.native_enabled:
        scale_native = native.scale_model_input(latents, sigma)
        step_native = native.euler_step(cond, latents, sigma, sigma_next)
        cfg_step_native = native.cfg_euler_step(cond, uncond, latents, cfg_scale, sigma, sigma_next)
        nhwc_native = native.nchw_to_nhwc_scaled(latents, 1.0 / scaling_factor)
        print(f"scale_model_input diff : {max_abs_diff(scale_np, scale_native):.8f}")
        print(f"euler_step diff        : {max_abs_diff(step_np, step_native):.8f}")
        print(f"cfg_euler_step diff    : {max_abs_diff(cfg_step_np, cfg_step_native):.8f}")
        print(f"nchw_to_nhwc diff      : {max_abs_diff(nhwc_np, nhwc_native):.8f}")
    else:
        print("Native backend is unavailable; timings below show NumPy fallback only.")

    print("\nNumPy fallback:")
    scale_out = np.empty_like(latents)
    step_out = np.empty_like(latents)
    cfg_step_out = np.empty_like(latents)
    nhwc_out = np.empty((1, 128, 128, 4), dtype=np.float32)
    numpy_scale = benchmark("scale_model_input", lambda: numpy_only.scale_model_input(latents, sigma, out=scale_out), args.warmup, args.iters)
    numpy_step = benchmark("euler_step", lambda: numpy_only.euler_step(cond, latents, sigma, sigma_next, out=step_out), args.warmup, args.iters)
    numpy_cfg = benchmark("cfg_euler_step", lambda: numpy_only.cfg_euler_step(cond, uncond, latents, cfg_scale, sigma, sigma_next, out=cfg_step_out), args.warmup, args.iters)
    numpy_nhwc = benchmark("nchw_to_nhwc_scaled", lambda: numpy_only.nchw_to_nhwc_scaled(latents, 1.0 / scaling_factor, out=nhwc_out), args.warmup, args.iters)

    if native.native_enabled:
        print("\nNative backend:")
        native_scale_out = np.empty_like(latents)
        native_step_out = np.empty_like(latents)
        native_cfg_step_out = np.empty_like(latents)
        native_nhwc_out = np.empty((1, 128, 128, 4), dtype=np.float32)
        native_scale = benchmark("scale_model_input", lambda: native.scale_model_input(latents, sigma, out=native_scale_out), args.warmup, args.iters)
        native_step = benchmark("euler_step", lambda: native.euler_step(cond, latents, sigma, sigma_next, out=native_step_out), args.warmup, args.iters)
        native_cfg = benchmark("cfg_euler_step", lambda: native.cfg_euler_step(cond, uncond, latents, cfg_scale, sigma, sigma_next, out=native_cfg_step_out), args.warmup, args.iters)
        native_nhwc = benchmark("nchw_to_nhwc_scaled", lambda: native.nchw_to_nhwc_scaled(latents, 1.0 / scaling_factor, out=native_nhwc_out), args.warmup, args.iters)

        print("\nEstimated speedups:")
        print(f"scale_model_input  : {numpy_scale / native_scale:8.3f}x")
        print(f"euler_step         : {numpy_step / native_step:8.3f}x")
        print(f"cfg_euler_step     : {numpy_cfg / native_cfg:8.3f}x")
        print(f"nchw_to_nhwc_scaled: {numpy_nhwc / native_nhwc:8.3f}x")

    print("\nNote: this benchmark measures scheduler/layout hot paths only, not QNN inference time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
