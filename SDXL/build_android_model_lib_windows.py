from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


CONVERTER_JNI_FILES = [
    "QnnModel.cpp",
    "QnnModel.hpp",
    "QnnTypeMacros.hpp",
    "QnnWrapperUtils.cpp",
    "QnnWrapperUtils.hpp",
    "QnnModelPal.hpp",
    "Application.mk",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Android QNN model lib on Windows by prebuilding raw weights into .o files.")
    ap.add_argument("--sdk-root", required=True)
    ap.add_argument("--model-cpp", required=True)
    ap.add_argument("--model-bin", required=True)
    ap.add_argument("--ndk-root", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--lib-name", default="qnn_model")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel objcopy jobs. Default: CPU count.")
    return ap.parse_args()


def copy_converter_sources(sdk_root: Path, jni_dir: Path) -> None:
    share_jni = sdk_root / "share" / "QNN" / "converter" / "jni"
    for name in CONVERTER_JNI_FILES:
        shutil.copy2(share_jni / name, jni_dir / name)
    shutil.copy2(share_jni / "linux" / "QnnModelPal.cpp", jni_dir / "QnnModelPal.cpp")


def extract_bin(model_bin: Path, raw_dir: Path) -> list[Path]:
    with tarfile.open(model_bin) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".raw")]
        tf.extractall(raw_dir, members=members)
    return sorted(raw_dir.rglob("*.raw"))


def build_objcopy_command(raw_rel: Path, out_rel: Path, llvm_objcopy: Path) -> list[str]:
    return [
        str(llvm_objcopy),
        "-I",
        "binary",
        "-O",
        "elf64-littleaarch64",
        "-B",
        "aarch64",
        str(raw_rel).replace("\\", "/"),
        str(out_rel).replace("\\", "/"),
    ]


def run_objcopy(build_dir: Path, llvm_objcopy: Path, raw_path: Path, out_path: Path) -> None:
    raw_rel = raw_path.relative_to(build_dir)
    out_rel = out_path.relative_to(build_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_objcopy_command(raw_rel, out_rel, llvm_objcopy)
    subprocess.run(cmd, cwd=build_dir, check=True)


def _rmtree_onerror(func, path, exc_info) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    func(path)


def reset_build_dir(build_dir: Path) -> None:
    if not build_dir.exists():
        return
    try:
        shutil.rmtree(build_dir, onerror=_rmtree_onerror)
        return
    except Exception as exc:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stale_dir = build_dir.with_name(f"{build_dir.name}_stale_{stamp}")
        try:
            build_dir.rename(stale_dir)
            print(f"[warn] existing build dir was locked; moved to {stale_dir}")
            return
        except Exception:
            raise RuntimeError(f"Failed to reset build dir {build_dir}: {exc}") from exc


def write_rsp_file(path: Path, items: list[Path]) -> None:
    normalized = [str(item).replace("\\", "/") for item in items]
    path.write_text("\n".join(f'"{item}"' for item in normalized) + "\n", encoding="utf-8")


def compile_cpp(compiler: Path, include_dir: Path, src_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(compiler),
        "-c",
        "-std=c++11",
        "-O3",
        "-fPIC",
        "-fvisibility=hidden",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-fno-rtti",
        "-Wno-write-strings",
        "-Wno-c99-designator",
        '-DQNN_API=__attribute__((visibility(\"default\")))',
        "-I",
        str(src_path.parent),
        "-I",
        str(include_dir),
        "-o",
        str(out_path),
        str(src_path),
    ]
    subprocess.run(cmd, check=True)


def link_shared_library(linker: Path, output_path: Path, rsp_path: Path, ndk_lib_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(linker),
        "-shared",
        "-nostdlib",
        "-o",
        str(output_path),
        f"@{rsp_path}",
        "-L",
        str(ndk_lib_dir),
        "-lc++_shared",
        "-Wl,-z,max-page-size=16384",
        "-Wl,--allow-shlib-undefined",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    sdk_root = Path(args.sdk_root).resolve()
    model_cpp = Path(args.model_cpp).resolve()
    model_bin = Path(args.model_bin).resolve()
    ndk_root = Path(args.ndk_root).resolve()
    build_dir = Path(args.build_dir).resolve()
    jni_dir = build_dir / "jni"
    raw_dir = build_dir / "obj" / "binary"
    llvm_objcopy = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "llvm-objcopy.exe"
    clangxx = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "aarch64-linux-android21-clang++.cmd"
    ndk_lib_dir = ndk_root / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "sysroot" / "usr" / "lib" / "aarch64-linux-android"
    lib_filename = args.lib_name if args.lib_name.endswith(".so") else f"lib{args.lib_name}.so"
    lib_stem = Path(lib_filename).stem
    model_cpp_copy_name = model_cpp.name if model_cpp.suffix else f"{model_cpp.name}.cpp"
    if lib_stem.startswith("lib"):
        library_module = lib_stem[3:]
    else:
        library_module = lib_stem
    prebuilt_dir = build_dir / "obj" / "local" / "arm64-v8a" / "objs" / library_module / "prebuilt_objs"
    compiled_dir = build_dir / "obj" / "local" / "arm64-v8a" / "objs" / library_module / "manual_cpp"
    link_rsp = build_dir / "obj" / "local" / "arm64-v8a" / "objs" / library_module / "link_inputs.rsp"
    output = build_dir / "libs" / "arm64-v8a" / lib_filename

    reset_build_dir(build_dir)
    jni_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prebuilt_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)

    copy_converter_sources(sdk_root, jni_dir)
    shutil.copy2(model_cpp, jni_dir / model_cpp_copy_name)
    shutil.copy2(model_bin, jni_dir / model_bin.name)

    raw_files = extract_bin(model_bin, raw_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files extracted from {model_bin}")

    print(f"[info] extracted {len(raw_files)} raw files")
    jobs = args.jobs if args.jobs > 0 else None
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for raw_path in raw_files:
            out_path = prebuilt_dir / (raw_path.stem + ".o")
            fut = pool.submit(run_objcopy, build_dir, llvm_objcopy, raw_path, out_path)
            futures[fut] = raw_path.name
        completed = 0
        total = len(futures)
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed % 250 == 0 or completed == total:
                print(f"[info] objcopy {completed}/{total}")

    include_dir = sdk_root / "include" / "QNN"
    cpp_sources = [
        jni_dir / "QnnModel.cpp",
        jni_dir / "QnnModelPal.cpp",
        jni_dir / "QnnWrapperUtils.cpp",
        jni_dir / model_cpp_copy_name,
    ]
    compiled_objs: list[Path] = []
    for src_path in cpp_sources:
        out_path = compiled_dir / f"{src_path.stem}.o"
        print(f"[info] compile {src_path.name}")
        compile_cpp(clangxx, include_dir, src_path, out_path)
        compiled_objs.append(out_path)

    prebuilt_objs = sorted(prebuilt_dir.glob("*.o"))
    if not prebuilt_objs:
        raise RuntimeError(f"No prebuilt object files found in {prebuilt_dir}")

    write_rsp_file(link_rsp, compiled_objs + prebuilt_objs)
    print(f"[info] link {output.name} with {len(compiled_objs)} compiled objs + {len(prebuilt_objs)} prebuilt objs")
    link_shared_library(clangxx, output, link_rsp, ndk_lib_dir)

    if not output.is_file():
        raise RuntimeError(f"Expected output library missing: {output}")
    print(f"[ok] built {output}")


if __name__ == "__main__":
    main()
