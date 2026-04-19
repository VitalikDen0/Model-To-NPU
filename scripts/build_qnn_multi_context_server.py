#!/usr/bin/env python3
"""Build the standalone multi-context QNN server for Android (arm64-v8a).

Uses NDK clang directly - no SampleApp dependency.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SERVER_SRC = SCRIPT_DIR.parent / "NPU" / "qnn_multi_context_server.c"
OUT_DIR = SCRIPT_DIR.parent / "NPU" / "out" / "arm64-v8a"

DEFAULT_NDK_ROOT = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")
DEFAULT_QAIRT_ROOT = Path(r"D:\platform-tools\sdxl_npu\qairt_2.44\qairt\2.44.0.260225")

ADB = Path(r"D:\platform-tools\adb.exe")
PHONE_BIN_DIR = "/data/local/tmp/sdxl_qnn/bin"


def find_clang(ndk_root: Path) -> Path:
    """Find the NDK aarch64 clang compiler."""
    toolchain = ndk_root / "toolchains" / "llvm" / "prebuilt"
    # Windows: windows-x86_64, Linux: linux-x86_64
    for host in ["windows-x86_64", "linux-x86_64", "darwin-x86_64"]:
        d = toolchain / host
        if d.exists():
            # Find highest API level clang
            for api in [35, 34, 33, 31, 30, 29, 28, 26, 24, 21]:
                clang = d / "bin" / f"aarch64-linux-android{api}-clang"
                if sys.platform == "win32":
                    clang = clang.with_suffix(".cmd")
                if clang.exists():
                    return clang
            # Fallback: plain clang with --target
            clang = d / "bin" / "clang"
            if sys.platform == "win32":
                clang = clang.with_suffix(".exe")
            if clang.exists():
                return clang
    raise FileNotFoundError(f"Cannot find NDK clang under {ndk_root}")


def build(qairt_root: Path, ndk_root: Path, out_dir: Path) -> Path:
    clang = find_clang(ndk_root)
    qnn_include = qairt_root / "include" / "QNN"

    out_dir.mkdir(parents=True, exist_ok=True)
    output_binary = out_dir / "qnn-multi-context-server"

    cmd = [
        str(clang),
        "-O2",
        "-Wall", "-Wextra", "-Wno-unused-parameter",
        f"-I{qnn_include}",
        f"-I{qnn_include / 'HTP'}",
        f"-I{qnn_include / 'System'}",
        str(SERVER_SRC),
        "-o", str(output_binary),
        "-ldl",
        "-lm",
        "-static-libgcc",
        "-pie",
    ]

    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not output_binary.exists():
        raise FileNotFoundError(f"Build finished but binary not found: {output_binary}")

    print(f"[ok] Built: {output_binary} ({output_binary.stat().st_size} bytes)")
    return output_binary


def deploy(binary: Path) -> None:
    """Push binary to phone via ADB."""
    if not ADB.exists():
        print(f"[skip] ADB not found at {ADB}, skipping deploy")
        return

    print(f"[deploy] Pushing to {PHONE_BIN_DIR}/")
    subprocess.run([
        str(ADB), "push", str(binary),
        f"{PHONE_BIN_DIR}/{binary.name}"
    ], check=True)
    subprocess.run([
        str(ADB), "shell", "chmod", "755",
        f"{PHONE_BIN_DIR}/{binary.name}"
    ], check=True)
    print(f"[ok] Deployed: {PHONE_BIN_DIR}/{binary.name}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build multi-context QNN server for Android")
    ap.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT)
    ap.add_argument("--ndk-root", type=Path, default=DEFAULT_NDK_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--deploy", action="store_true", help="Push binary to phone via ADB")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not SERVER_SRC.exists():
        print(f"ERROR: Source not found: {SERVER_SRC}", file=sys.stderr)
        return 1

    qnn_include = args.qairt_root / "include" / "QNN"
    if not qnn_include.exists():
        print(f"ERROR: QNN headers not found: {qnn_include}", file=sys.stderr)
        return 1

    binary = build(args.qairt_root, args.ndk_root, args.out_dir)

    if args.deploy:
        deploy(binary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
