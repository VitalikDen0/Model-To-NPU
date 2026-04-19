#!/usr/bin/env python3
"""
Extract Termux .deb packages into a standalone prefix tree.

Produces a directory structure identical to what Termux's `dpkg -i` would
create, but without needing dpkg or a working Termux environment.

A .deb is an `ar` archive containing:
  - debian-binary   (version string)
  - control.tar.*   (package metadata)
  - data.tar.*      (actual files: usr/bin, usr/lib, ...)

We only need the data.tar.* part.  All paths inside are relative to the
Termux prefix (data/data/com.termux/files/usr/) — we strip that prefix
and extract directly into ``<output>/prefix/``.

Usage:
    python scripts/build_termux_prefix.py                         # defaults
    python scripts/build_termux_prefix.py --debs-dir path/to/debs --output path/to/out
"""
import argparse
import io
import os
import stat
import struct
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBS_DIR = ROOT / "local_tools" / "termux_repo" / "debs"
DEFAULT_OUTPUT = ROOT / "build" / "termux_prefix_standalone"

TERMUX_PREFIX_STRIP = "data/data/com.termux/files/usr"


def _read_ar_members(fp):
    """Yield (name, size, data_bytes) for each member of an ar(1) archive."""
    magic = fp.read(8)
    if magic != b"!<arch>\n":
        raise ValueError("Not a valid ar archive")

    while True:
        header = fp.read(60)
        if len(header) < 60:
            break
        name = header[0:16].decode("ascii").strip().rstrip("/")
        size = int(header[48:58].decode("ascii").strip())
        data = fp.read(size)
        if size % 2:
            fp.read(1)  # padding byte
        yield name, size, data


def _extract_data_tar(deb_data: bytes, dest: Path, strip_prefix: str):
    """Extract the data.tar.* from a deb's raw bytes into dest."""
    for name, _size, data in _read_ar_members(io.BytesIO(deb_data)):
        if name.startswith("data.tar"):
            break
    else:
        raise ValueError("No data.tar.* found in .deb")

    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for member in tf.getmembers():
            # Normalise path: strip leading ./ and the termux prefix
            rel = member.name.lstrip("./")
            if rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix):].lstrip("/")
            elif rel.startswith("usr/"):
                # Some packages use plain usr/ paths
                pass
            elif rel in (".", ""):
                continue

            if not rel:
                continue

            target = dest / rel

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                os.symlink(member.linkname, target)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src:
                    target.write_bytes(src.read())
                # Preserve executable bit
                if member.mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_prefix(debs_dir: Path, output: Path):
    """Extract all .deb files from debs_dir into output/prefix/."""
    prefix_dir = output / "prefix"
    prefix_dir.mkdir(parents=True, exist_ok=True)

    deb_files = sorted(debs_dir.glob("*.deb"))
    if not deb_files:
        print(f"ERROR: no .deb files found in {debs_dir}")
        return

    print(f"[build_termux_prefix] Extracting {len(deb_files)} packages -> {prefix_dir}")
    total_size = 0
    skipped = []
    for i, deb_path in enumerate(deb_files, 1):
        pkg_name = deb_path.stem
        deb_data = deb_path.read_bytes()
        total_size += len(deb_data)
        try:
            _extract_data_tar(deb_data, prefix_dir, TERMUX_PREFIX_STRIP)
            print(f"  [{i}/{len(deb_files)}] {pkg_name} ({len(deb_data) / 1e6:.2f} MB)")
        except Exception as e:
            print(f"  [{i}/{len(deb_files)}] {pkg_name} - SKIPPED (corrupt/truncated: {e})")
            skipped.append(pkg_name)

    # Verify critical files
    python_bin = prefix_dir / "bin" / "python3"
    python_lib = list((prefix_dir / "lib").glob("libpython3*.so*")) if (prefix_dir / "lib").exists() else []

    print(f"\n[build_termux_prefix] Done! ({total_size / 1e6:.1f} MB compressed)")
    if skipped:
        print(f"  WARNING: {len(skipped)} packages skipped (re-download them):")
        for s in skipped:
            print(f"    - {s}")
    print(f"  Prefix: {prefix_dir}")

    if python_bin.exists():
        print(f"  python3: OK ({python_bin})")
    else:
        print(f"  WARNING: python3 not found at {python_bin}")
        # Check if it's under bin/python3.13 etc
        for p in sorted((prefix_dir / "bin").glob("python3*")):
            print(f"    found: {p}")

    if python_lib:
        print(f"  libpython: OK ({python_lib[0].name})")
    else:
        print("  WARNING: libpython3*.so not found")

    # Quick import test (won't work on Windows but documents what to check)
    print(f"\n  To test on phone:")
    print(f"    export PREFIX={prefix_dir}")
    print(f"    export LD_LIBRARY_PATH=$PREFIX/lib")
    print(f"    $PREFIX/bin/python3 -c 'import numpy; import PIL; print(\"OK\")'")

    return prefix_dir


def main():
    ap = argparse.ArgumentParser(description="Extract Termux .deb packages into standalone prefix")
    ap.add_argument("--debs-dir", type=str, default=str(DEFAULT_DEBS_DIR),
                    help="Directory containing .deb files")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Output directory (prefix will be at <output>/prefix/)")
    args = ap.parse_args()

    build_prefix(Path(args.debs_dir), Path(args.output))


if __name__ == "__main__":
    main()
