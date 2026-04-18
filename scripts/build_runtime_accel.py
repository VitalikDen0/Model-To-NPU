#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(os.path.abspath(__file__)).parents[1]
SRC = REPO_ROOT / "NPU" / "runtime_accel" / "sdxl_runtime_accel.c"
OUT_ROOT = REPO_ROOT / "NPU" / "build" / "runtime_accel"
DEFAULT_NDK_CANDIDATES = [
    Path(os.environ.get("ANDROID_NDK_ROOT", "")),
    Path(os.environ.get("ANDROID_NDK_HOME", "")),
    Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358"),
    Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\27.2.12479018"),
]


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def detect_host_compiler(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in (os.environ.get("CC", "").strip(), "gcc", "clang"):
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError("No host C compiler found (tried CC, gcc, clang)")


def detect_ndk_root(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(path for path in DEFAULT_NDK_CANDIDATES if str(path))
    for candidate in candidates:
        if candidate and candidate.exists() and (candidate / "toolchains" / "llvm").exists():
            return candidate
    raise RuntimeError("Android NDK not found. Pass --ndk-root or set ANDROID_NDK_ROOT")


def build_host(compiler: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        out_path = out_dir / "sdxl_runtime_accel.dll"
    elif sys.platform == "darwin":
        out_path = out_dir / "libsdxl_runtime_accel.dylib"
    else:
        out_path = out_dir / "libsdxl_runtime_accel.so"

    cmd = [compiler, "-O3", "-ffast-math", "-std=c11", "-shared"]
    if os.name != "nt":
        cmd.append("-fPIC")
    else:
        cmd.append("-static-libgcc")
    cmd.extend(["-o", str(out_path), str(SRC), "-lm"])
    run(cmd)
    return out_path


def detect_ndk_clang(ndk_root: Path, api_level: int) -> Path:
    bin_dir = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin"
    candidates = [
        bin_dir / f"aarch64-linux-android{api_level}-clang.cmd",
        bin_dir / f"aarch64-linux-android{api_level}-clang",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Android clang wrapper for API {api_level} not found under {bin_dir}")


def build_android_arm64(ndk_root: Path, api_level: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler = detect_ndk_clang(ndk_root, api_level)
    out_path = out_dir / "libsdxl_runtime_accel.so"
    cmd = [
        str(compiler),
        "-O3",
        "-ffast-math",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-o",
        str(out_path),
        str(SRC),
        "-lm",
    ]
    run(cmd)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the optional SDXL runtime accelerator shared library")
    parser.add_argument("--target", choices=["host", "android-arm64", "all"], default="host")
    parser.add_argument("--compiler", default=None, help="Host compiler override (gcc/clang)")
    parser.add_argument("--ndk-root", default=None, help="Android NDK root for android-arm64 builds")
    parser.add_argument("--api-level", type=int, default=28, help="Android API level for android-arm64 build")
    args = parser.parse_args()

    if not SRC.exists():
        raise RuntimeError(f"Source file not found: {SRC}")

    built: list[Path] = []
    if args.target in ("host", "all"):
        compiler = detect_host_compiler(args.compiler)
        built.append(build_host(compiler, OUT_ROOT / "windows-x64" if os.name == "nt" else OUT_ROOT / "linux-x86_64"))

    if args.target in ("android-arm64", "all"):
        ndk_root = detect_ndk_root(args.ndk_root)
        built.append(build_android_arm64(ndk_root, args.api_level, OUT_ROOT / "android-arm64"))

    print("[ok] built artifacts:")
    for artifact in built:
        print(f"  - {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
