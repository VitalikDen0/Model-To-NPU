#!/usr/bin/env python3
"""
Deploy SDXL NPU pipeline to phone via ADB.

Pushes context binaries, QNN runtime libs, phone_generate.py,
tokenizer files, the optional TAESD ONNX preview decoder,
and optional TAESD QNN preview assets when they are available locally.

Usage:
  python scripts/deploy_to_phone.py --contexts-dir /path/to/contexts
  python scripts/deploy_to_phone.py --adb /path/to/adb.exe --serial SERIAL
"""
import argparse
import os
import subprocess
import sys
import time


DEFAULT_PHONE_BASE = "/sdcard/Download/sdxl_qnn"

# Expected context binary files
CONTEXT_FILES = [
    "clip_l.serialized.bin.bin",
    "clip_g.serialized.bin.bin",
    "vae_decoder.serialized.bin.bin",
    # Split UNet (FP16)
    "unet_encoder_fp16.serialized.bin.bin",
    "unet_decoder_fp16.serialized.bin.bin",
]

# QNN runtime libraries needed on phone
QNN_LIBS = [
    "libQnnHtp.so",
    "libQnnHtpNetRunExtensions.so",
    "libQnnHtpV79Stub.so",
    "libQnnHtpV79Skel.so",
    "libQnnSystem.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpProfilingReader.so",
]

OPTIONAL_QNN_LIBS = [
    "libQnnGpu.so",
    "libQnnGpuNetRunExtensions.so",
]

OPTIONAL_QNN_BINS = [
    "qnn-context-binary-generator",
    "qnn-gpu-target-server",
]


def phone_dirs(phone_base):
    return [
        f"{phone_base}/context",
        f"{phone_base}/phone_gen/tokenizer",
        f"{phone_base}/phone_gen/work",
        f"{phone_base}/lib",
        f"{phone_base}/bin",
        f"{phone_base}/model",
        f"{phone_base}/outputs",
    ]


def adb_cmd(adb_path, serial, *args):
    """Run an ADB command and return (returncode, stdout, stderr)."""
    cmd = [adb_path]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr


def adb_push(adb_path, serial, local, remote):
    """Push a file to phone, with progress info."""
    size_mb = os.path.getsize(local) / (1024 * 1024)
    print(f"  Push {os.path.basename(local)} ({size_mb:.1f} MB) → {remote}")
    rc, out, err = adb_cmd(adb_path, serial, "push", local, remote)
    if rc != 0:
        print(f"    ERROR: {err.strip()}")
        return False
    return True


def find_adb():
    """Find ADB in common locations."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "adb.exe"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "adb"),
        "adb",
    ]
    # Also check ANDROID_HOME
    android_home = os.environ.get("ANDROID_HOME", "")
    if android_home:
        candidates.insert(0, os.path.join(android_home, "platform-tools", "adb"))
        candidates.insert(0, os.path.join(android_home, "platform-tools", "adb.exe"))

    for c in candidates:
        try:
            r = subprocess.run([c, "version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def find_optional_taesd_onnx(repo_root: str) -> str | None:
    candidates = [
        os.path.join(repo_root, "examples", "rooted-phone-sample", "phone_gen", "taesd_decoder.onnx"),
        os.path.join("D:/platform-tools/sdxl_npu/taesd_decoder", "taesd_decoder.onnx"),
        os.path.join("D:/platform-tools/NPU/taesd_decoder", "taesd_decoder.onnx"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_optional_taesd_context(repo_root: str) -> str | None:
    candidates = [
        os.path.join(repo_root, "examples", "rooted-phone-sample", "context", "taesd_decoder.serialized.bin.bin"),
        os.path.join("D:/platform-tools/sdxl_npu", "context", "taesd_decoder.serialized.bin.bin"),
        os.path.join("D:/platform-tools/sdxl_npu", "taesd_decoder", "context", "taesd_decoder.serialized.bin.bin"),
        os.path.join("D:/platform-tools/NPU", "context", "taesd_decoder.serialized.bin.bin"),
        os.path.join("D:/platform-tools/NPU", "taesd_decoder", "context", "taesd_decoder.serialized.bin.bin"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_optional_taesd_model(repo_root: str) -> str | None:
    candidates = [
        os.path.join("D:/platform-tools/sdxl_npu", "taesd_decoder", "android_lib", "libTAESDDecoder.so"),
        os.path.join("D:/platform-tools/NPU", "taesd_decoder", "android_lib", "libTAESDDecoder.so"),
        os.path.join(repo_root, "local_tools", "taesd_decoder", "android_lib", "libTAESDDecoder.so"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_optional_taesd_gpu_runner() -> str | None:
    candidates = [
        r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130\bin\aarch64-android\qnn-net-run",
        r"D:\platform-tools\sdxl_npu\qairt_2.31\qairt\2.31.0.250130\bin\aarch64-android\qnn-net-run",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_optional_context_runner(repo_root: str, qnn_bin_dir: str | None) -> str | None:
    candidates = []
    if qnn_bin_dir:
        candidates.append(os.path.join(qnn_bin_dir, "qnn-context-runner"))
    candidates.append(os.path.join(repo_root, "NPU", "out", "arm64-v8a", "qnn-context-runner"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy SDXL NPU to phone")
    parser.add_argument("--adb", type=str, default=None,
                        help="Path to adb binary")
    parser.add_argument("--serial", type=str, default=None,
                        help="ADB device serial")
    parser.add_argument("--contexts-dir", type=str, required=True,
                        help="Directory containing context binary files")
    parser.add_argument("--phone-base", type=str, default=DEFAULT_PHONE_BASE,
                        help="Phone-side deploy directory (default: /sdcard/Download/sdxl_qnn)")
    parser.add_argument("--qnn-lib-dir", type=str, default=None,
                        help="Directory containing QNN .so runtime libs")
    parser.add_argument("--qnn-bin-dir", type=str, default=None,
                        help="Directory containing qnn-net-run binary")
    parser.add_argument("--skip-contexts", action="store_true",
                        help="Skip pushing context binaries")
    parser.add_argument("--skip-libs", action="store_true",
                        help="Skip pushing QNN libs")
    args = parser.parse_args()

    # Find ADB
    adb = args.adb or find_adb()
    if not adb:
        print("ERROR: ADB not found. Install Android platform-tools or use --adb")
        sys.exit(1)

    # Check device
    rc, out, _ = adb_cmd(adb, args.serial, "devices")
    if rc != 0:
        print("ERROR: ADB not working")
        sys.exit(1)
    devices = [l.split()[0] for l in out.strip().split("\n")[1:] if "\tdevice" in l]
    if not devices:
        print("ERROR: No device connected")
        sys.exit(1)
    serial = args.serial or devices[0]
    phone_base = args.phone_base.rstrip("/")
    print(f"Device: {serial}")

    # Create directories on phone
    print("\n[1/5] Creating directories...")
    for d in phone_dirs(phone_base):
        adb_cmd(adb, serial, "shell", f"mkdir -p {d}")

    # Push context binaries
    if not args.skip_contexts:
        print("\n[2/5] Pushing context binaries...")
        ctx_dir = args.contexts_dir
        if not os.path.isdir(ctx_dir):
            print(f"ERROR: Contexts directory not found: {ctx_dir}")
            sys.exit(1)
        for f in CONTEXT_FILES:
            local = os.path.join(ctx_dir, f)
            if os.path.exists(local):
                adb_push(adb, serial, local, f"{phone_base}/context/{f}")
            else:
                print(f"  SKIP {f} (not found)")
    else:
        print("\n[2/5] Skipping context binaries")

    # Push QNN runtime libs
    if not args.skip_libs and args.qnn_lib_dir:
        print("\n[3/5] Pushing QNN runtime libs...")
        for lib in QNN_LIBS:
            local = os.path.join(args.qnn_lib_dir, lib)
            if os.path.exists(local):
                adb_push(adb, serial, local, f"{phone_base}/lib/{lib}")
            else:
                print(f"  SKIP {lib} (not found)")

        for lib in OPTIONAL_QNN_LIBS:
            local = os.path.join(args.qnn_lib_dir, lib)
            if os.path.exists(local):
                adb_push(adb, serial, local, f"{phone_base}/lib/{lib}")
            else:
                print(f"  SKIP {lib} (optional GPU runtime not found)")

        # Push qnn-net-run binary
        if args.qnn_bin_dir:
            qnr = os.path.join(args.qnn_bin_dir, "qnn-net-run")
            if os.path.exists(qnr):
                adb_push(adb, serial, qnr, f"{phone_base}/bin/qnn-net-run")
                adb_cmd(adb, serial, "shell", f"chmod 755 {phone_base}/bin/qnn-net-run")
            for bin_name in OPTIONAL_QNN_BINS:
                local = os.path.join(args.qnn_bin_dir, bin_name)
                if os.path.exists(local):
                    adb_push(adb, serial, local, f"{phone_base}/bin/{bin_name}")
                    adb_cmd(adb, serial, "shell", f"chmod 755 {phone_base}/bin/{bin_name}")
                else:
                    print(f"  SKIP {bin_name} (optional binary not found)")
    else:
        print("\n[3/5] Skipping QNN libs (use --qnn-lib-dir)")

    # Push phone_generate.py, tokenizer, and optional TAESD preview model
    print("\n[4/5] Pushing phone_generate.py + tokenizer...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gen_py = os.path.join(repo_root, "phone_generate.py")
    if os.path.exists(gen_py):
        adb_push(adb, serial, gen_py, f"{phone_base}/phone_gen/generate.py")
    else:
        print(f"  ERROR: {gen_py} not found")

    tok_dir = os.path.join(repo_root, "tokenizer")
    for f in ["vocab.json", "merges.txt"]:
        local = os.path.join(tok_dir, f)
        if os.path.exists(local):
            adb_push(adb, serial, local, f"{phone_base}/phone_gen/tokenizer/{f}")
        else:
            print(f"  ERROR: {local} not found")

    taesd_onnx = find_optional_taesd_onnx(repo_root)
    if taesd_onnx:
        adb_push(adb, serial, taesd_onnx, f"{phone_base}/phone_gen/taesd_decoder.onnx")
    else:
        print("  SKIP taesd_decoder.onnx (optional preview model not found locally)")

    taesd_context = find_optional_taesd_context(repo_root)
    if taesd_context:
        adb_push(adb, serial, taesd_context, f"{phone_base}/context/taesd_decoder.serialized.bin.bin")
    else:
        print("  SKIP taesd_decoder.serialized.bin.bin (optional QNN preview context not found locally)")

    taesd_model = find_optional_taesd_model(repo_root)
    if taesd_model:
        adb_push(adb, serial, taesd_model, f"{phone_base}/model/libTAESDDecoder.so")
    else:
        print("  SKIP libTAESDDecoder.so (optional QNN preview model not found locally)")

    taesd_gpu_runner = find_optional_taesd_gpu_runner()
    if taesd_gpu_runner:
        adb_push(adb, serial, taesd_gpu_runner, f"{phone_base}/bin/qnn-net-run-gpu")
        adb_cmd(adb, serial, "shell", f"chmod 755 {phone_base}/bin/qnn-net-run-gpu")
    else:
        print("  SKIP qnn-net-run-gpu (optional TAESD GPU preview runner not found locally)")

    extra_files = [
        (os.path.join(repo_root, "SDXL", "run_ctxgen_lightning.sh"), f"{phone_base}/run_ctxgen_lightning.sh", True),
        (os.path.join(repo_root, "SDXL", "htp_backend_extensions_lightning.json"), f"{phone_base}/htp_backend_extensions_lightning.json", False),
        (os.path.join(repo_root, "SDXL", "htp_backend_ext_config_lightning.json"), f"{phone_base}/htp_backend_ext_config_lightning.json", False),
    ]
    context_runner = find_optional_context_runner(repo_root, args.qnn_bin_dir)
    if context_runner:
        extra_files.append((context_runner, f"{phone_base}/bin/qnn-context-runner", True))
    else:
        print("  SKIP qnn-context-runner (optional persistent runner not found locally)")
    for local, remote, make_executable in extra_files:
        if os.path.exists(local):
            adb_push(adb, serial, local, remote)
            if make_executable:
                adb_cmd(adb, serial, "shell", f"chmod 755 {remote}")
        else:
            print(f"  SKIP {os.path.basename(local)} (not found)")

    # Verify deployment
    print("\n[5/5] Verifying...")
    rc, out, _ = adb_cmd(adb, serial, "shell", f"ls -la {phone_base}/context/ 2>/dev/null")
    if rc == 0:
        print("  Context binaries:")
        for line in out.strip().split("\n"):
            if ".serialized" in line or "total" in line:
                print(f"    {line.strip()}")

    rc, out, _ = adb_cmd(adb, serial, "shell",
                         f"ls -la {phone_base}/phone_gen/generate.py "
                         f"{phone_base}/phone_gen/tokenizer/ 2>/dev/null")
    if rc == 0:
        print("  Phone generator:")
        for line in out.strip().split("\n"):
            print(f"    {line.strip()}")

    # Quick sanity check
    rc, out, _ = adb_cmd(adb, serial, "shell",
                         f"du -sh {phone_base}/ 2>/dev/null")
    if rc == 0:
        print(f"\n  Total on phone: {out.strip()}")

    print("\nDeployment complete!")
    print(f"Models are in:  {phone_base}/context/")
    print(f"Generator in:   {phone_base}/phone_gen/generate.py")
    print(f"\nTo test: adb shell 'export PATH=/data/data/com.termux/files/usr/bin:$PATH && "
          f"export SDXL_QNN_BASE={phone_base} && "
          f"python3 {phone_base}/phone_gen/generate.py \"hello world\"'")


if __name__ == "__main__":
    main()
