#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

DEFAULT_MANIFEST = Path(r"D:\platform-tools\wan21_13b_work\qnn\wan_t2v_1p3b_832x480_17f_seq128_qnn_manifest.json")
DEFAULT_SDK_ROOT = Path(r"D:\platform-tools\sdxl_npu\qairt_2.44\qairt\2.44.0.260225")
DEFAULT_NDK_ROOT = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")
DEFAULT_PHONE_BASE = "/data/local/tmp/wan21_t2v_qnn"
DEFAULT_PHONE_GENERATE = Path(__file__).resolve().parents[1] / "phone_generate.py"

RUNTIME_LIBS = [
    ("libQnnHtp.so", Path("lib") / "aarch64-android" / "libQnnHtp.so"),
    ("libQnnHtpNetRunExtensions.so", Path("lib") / "aarch64-android" / "libQnnHtpNetRunExtensions.so"),
    ("libQnnHtpPrepare.so", Path("lib") / "aarch64-android" / "libQnnHtpPrepare.so"),
    ("libQnnHtpProfilingReader.so", Path("lib") / "aarch64-android" / "libQnnHtpProfilingReader.so"),
    ("libQnnHtpV79Stub.so", Path("lib") / "aarch64-android" / "libQnnHtpV79Stub.so"),
    ("libQnnSystem.so", Path("lib") / "aarch64-android" / "libQnnSystem.so"),
    ("libQnnHtpV79Skel.so", Path("lib") / "hexagon-v79" / "unsigned" / "libQnnHtpV79Skel.so"),
]

RUNTIME_BINS = [
    ("qnn-net-run", Path("bin") / "aarch64-android" / "qnn-net-run"),
    ("qnn-context-binary-generator", Path("bin") / "aarch64-android" / "qnn-context-binary-generator"),
]


def run(cmd: list[str], *, label: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(str(exc.stdout).strip())
        if exc.stderr:
            print(str(exc.stderr).strip(), file=sys.stderr)
        raise
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit={result.returncode}").strip()
        raise RuntimeError(f"{label} failed: {detail}")
    return result


def find_adb(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    which = shutil.which("adb")
    if which:
        candidates.append(Path(which))
    home = Path.home()
    candidates.extend([
        home / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        home / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    ])
    for candidate in candidates:
        if candidate.exists():
            try:
                result = subprocess.run([str(candidate), "version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except Exception:
                pass
    raise SystemExit("adb not found; pass --adb explicitly")


def pick_serial(adb: Path, requested: str | None) -> str:
    out = run([str(adb), "devices", "-l"], label="adb devices", timeout=30).stdout
    ready: list[str] = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            ready.append(parts[0])
    if requested:
        if requested not in ready:
            raise SystemExit(f"Requested serial {requested!r} not found; ready={ready}")
        return requested
    if not ready:
        raise SystemExit("No ready adb device found")
    if len(ready) > 1:
        raise SystemExit(f"Multiple adb devices found: {', '.join(ready)}; pass --serial")
    return ready[0]


def adb_push(adb: Path, serial: str, local_path: Path, remote_path: str) -> None:
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    run([str(adb), "-s", serial, "push", str(local_path), remote_path], label=f"adb push {local_path.name}", timeout=3600)


def adb_shell(adb: Path, serial: str, command: str, *, root: bool = False, timeout: int = 3600) -> str:
    if root:
        wrapped = f"su -c {shlex.quote(command)}"
        cmd = [str(adb), "-s", serial, "shell", wrapped]
    else:
        cmd = [str(adb), "-s", serial, "shell", command]
    return run(cmd, label="adb shell", timeout=timeout).stdout.strip()


def ndk_libcxx(ndk_root: Path) -> Path:
    path = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "sysroot" / "usr" / "lib" / "aarch64-linux-android" / "libc++_shared.so"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deploy WAN 2.1 basic-debug runtime to phone")
    ap.add_argument("--adb")
    ap.add_argument("--serial")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    ap.add_argument("--ndk-root", type=Path, default=DEFAULT_NDK_ROOT)
    ap.add_argument("--phone-base", default=DEFAULT_PHONE_BASE)
    ap.add_argument("--phone-generate", type=Path, default=DEFAULT_PHONE_GENERATE)
    ap.add_argument("--context-wait-seconds", type=int, default=7200)
    ap.add_argument("--context-poll-seconds", type=float, default=5.0)
    return ap.parse_args()


def wait_for_context(
    adb: Path,
    serial: str,
    *,
    expected_path: str,
    timeout_seconds: int,
    poll_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    process_name = "qnn-context-binary-generator"
    while time.time() < deadline:
        status = adb_shell(
            adb,
            serial,
            (
                f"if [ -f {expected_path} ]; then "
                f"echo READY; ls -lh {expected_path}; "
                f"elif pidof {process_name} >/dev/null 2>&1; then "
                f"echo RUNNING; pidof {process_name}; "
                "else "
                f"echo MISSING; ls -lh {os.path.dirname(expected_path)} 2>/dev/null || true; "
                "fi"
            ),
            root=True,
            timeout=120,
        )
        if status.startswith("READY"):
            print(status)
            return
        if status.startswith("RUNNING"):
            print(status)
            time.sleep(max(0.2, poll_seconds))
            continue
        raise RuntimeError(f"Context generation stopped before output appeared: {status}")
    raise RuntimeError(f"Timed out waiting for context file: {expected_path}")


def context_status(adb: Path, serial: str, *, expected_path: str) -> str:
    status = adb_shell(
        adb,
        serial,
        (
            f"if [ -f {expected_path} ]; then "
            "echo READY; "
            f"elif pidof qnn-context-binary-generator >/dev/null 2>&1; then echo RUNNING; "
            "else echo MISSING; fi"
        ),
        root=True,
        timeout=120,
    )
    return status.splitlines()[0].strip() if status else "MISSING"


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    transformer = manifest.get("components", {}).get("transformer")
    if not isinstance(transformer, dict):
        raise SystemExit("manifest.components.transformer is missing")

    model_lib = Path(str(transformer["android_lib"]))
    if not model_lib.exists():
        raise SystemExit(f"Transformer Android lib not found: {model_lib}")

    context_arg = str(transformer["context_binary_file_arg"])
    context_out = str(transformer["context_binary_output"])

    adb = find_adb(args.adb)
    serial = pick_serial(adb, args.serial)
    base = args.phone_base.rstrip("/")

    adb_shell(adb, serial, f"mkdir -p {base}/bin {base}/lib {base}/model {base}/context {base}/inputs {base}/outputs {base}/work {base}/phone_gen", root=False)

    for name, rel in RUNTIME_LIBS:
        adb_push(adb, serial, args.sdk_root / rel, f"{base}/lib/{name}")
    adb_push(adb, serial, ndk_libcxx(args.ndk_root), f"{base}/lib/libc++_shared.so")

    for name, rel in RUNTIME_BINS:
        adb_push(adb, serial, args.sdk_root / rel, f"{base}/bin/{name}")
    adb_shell(adb, serial, f"chmod 755 {base}/bin/qnn-net-run {base}/bin/qnn-context-binary-generator", root=False)

    adb_push(adb, serial, model_lib, f"{base}/model/{model_lib.name}")
    adb_push(adb, serial, args.manifest, f"{base}/{args.manifest.name}")
    if args.phone_generate.exists():
        adb_push(adb, serial, args.phone_generate, f"{base}/phone_gen/generate.py")

    expected_context_path = f"{base}/context/{context_out}"
    current_status = context_status(adb, serial, expected_path=expected_context_path)
    if current_status == "READY":
        print(f"[info] context already exists: {expected_context_path}")
    elif current_status == "RUNNING":
        print("[info] reusing already-running qnn-context-binary-generator")
    else:
        shell = (
            f"export LD_LIBRARY_PATH={base}/lib:$LD_LIBRARY_PATH; "
            f"export ADSP_LIBRARY_PATH='{base}/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'; "
            f"rm -f {expected_context_path}; "
            f"{base}/bin/qnn-context-binary-generator --model {base}/model/{model_lib.name} "
            f"--backend {base}/lib/libQnnHtp.so --output_dir {base}/context --binary_file {context_arg}"
        )
        try:
            adb_shell(adb, serial, shell, root=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("[info] qnn-context-binary-generator exceeded 300s startup window; switching to polling mode")
            post_status = context_status(adb, serial, expected_path=expected_context_path)
            if post_status == "MISSING":
                raise RuntimeError("Context generation timed out and no running generator was detected")
            print(f"[info] post-timeout context status: {post_status}")
    wait_for_context(
        adb,
        serial,
        expected_path=expected_context_path,
        timeout_seconds=args.context_wait_seconds,
        poll_seconds=args.context_poll_seconds,
    )
    print(f"\n[ok] WAN basic-debug runtime deployed to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
