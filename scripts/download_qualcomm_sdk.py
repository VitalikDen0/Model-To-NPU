#!/usr/bin/env python3
"""
Download and install Qualcomm AI Engine Direct SDK (QAIRT).

This script downloads QAIRT SDK via pip (qai-hub-models includes qairt).
For manual download, visit: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk

Requirements: Python 3.10.x
"""
import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    v = sys.version_info
    if v.major != 3 or v.minor != 10:
        print(f"WARNING: Python {v.major}.{v.minor} detected. QAIRT requires Python 3.10.x")
        print("Some QAIRT tools may not work with other Python versions.")
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != 'y':
            sys.exit(1)


def install_qairt():
    print("=" * 60)
    print("Installing Qualcomm AI Engine Direct SDK (QAIRT)")
    print("=" * 60)

    check_python_version()

    packages = [
        "qai-hub",
        "qai-hub-models",
    ]

    for pkg in packages:
        print(f"\n[*] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # Check if qairt is available
    print("\n[*] Checking QAIRT installation...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import qai_hub; print('qai_hub version:', qai_hub.__version__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("WARNING: qai_hub import failed:", result.stderr.strip())
    except Exception as e:
        print(f"WARNING: Could not verify installation: {e}")

    # Try to find local QAIRT tools
    print("\n[*] Checking for local QAIRT converter tools...")
    qairt_paths = [
        Path(os.environ.get("QNN_SDK_ROOT", "")) / "bin",
        Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "qairt",
        Path(sys.prefix) / "Lib" / "site-packages" / "qairt",
    ]

    for p in qairt_paths:
        if p.exists():
            print(f"  Found: {p}")

    print("\n" + "=" * 60)
    print("QAIRT SDK installation complete!")
    print()
    print("For the ONNX-to-QNN converter, you also need:")
    print("  pip install qairt")
    print()
    print("Set QNN_SDK_ROOT environment variable to the SDK root directory.")
    print("Example: set QNN_SDK_ROOT=C:\\Qualcomm\\AIStack\\qairt\\2.31.0.250130")
    print("=" * 60)


if __name__ == "__main__":
    install_qairt()
