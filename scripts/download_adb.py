#!/usr/bin/env python3
"""
Download Android platform-tools (ADB) for Windows.

Downloads the official Google platform-tools package and extracts it.
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve


PLATFORM_TOOLS_URL = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
DEFAULT_INSTALL_DIR = Path.home() / "platform-tools"


def download_progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size if total_size > 0 else 0
    print(f"\r  Downloading: {pct}%", end="", flush=True)


def install_adb(install_dir=None):
    dest = Path(install_dir) if install_dir else DEFAULT_INSTALL_DIR

    print("=" * 60)
    print("Downloading Android Platform Tools (ADB)")
    print("=" * 60)

    # Check if already installed
    adb_path = dest / "adb.exe"
    if adb_path.exists():
        print(f"ADB already found at: {adb_path}")
        resp = input("Re-download? [y/N] ")
        if resp.lower() != 'y':
            return str(adb_path)

    # Download
    zip_path = dest.parent / "platform-tools-latest.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Downloading from Google...")
    print(f"    URL: {PLATFORM_TOOLS_URL}")
    urlretrieve(PLATFORM_TOOLS_URL, str(zip_path), download_progress)
    print()

    # Extract
    print(f"[*] Extracting to {dest}...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(dest.parent))

    # The ZIP contains a platform-tools/ subfolder
    extracted = dest.parent / "platform-tools"
    if extracted != dest and extracted.exists():
        if dest.exists():
            shutil.rmtree(str(dest))
        extracted.rename(dest)

    # Cleanup ZIP
    zip_path.unlink(missing_ok=True)

    adb = dest / "adb.exe"
    if adb.exists():
        print(f"\n[OK] ADB installed: {adb}")
        print(f"\nAdd to PATH:")
        print(f'  set PATH={dest};%PATH%')
        print(f"\nTest connection:")
        print(f"  {adb} devices")
    else:
        print(f"\n[ERROR] adb.exe not found at expected location: {adb}")

    return str(adb)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="Installation directory")
    args = ap.parse_args()
    install_adb(args.dir)
