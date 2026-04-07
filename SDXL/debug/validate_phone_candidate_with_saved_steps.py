#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ADB = Path("D:/platform-tools/adb.exe")
DEFAULT_PHONE_CONTROL_BASE = "/data/local/tmp/sdxl_qnn"
DEFAULT_PHONE_CANDIDATE_BASE = "/data/local/tmp/sdxl_qnn_244"
DEFAULT_INPUT_ROOT = f"{DEFAULT_PHONE_CONTROL_BASE}/runtime_work_lightning"
DEFAULT_CONTEXT = (
    f"{DEFAULT_PHONE_CANDIDATE_BASE}"
    "/context/unet_lightning8step_int8_qairt244_anime4s.serialized.bin.bin"
)
DEFAULT_CONFIG = f"{DEFAULT_PHONE_CANDIDATE_BASE}/htp_backend_extensions_lightning.json"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_PHONE_CANDIDATE_BASE}/candidate_saved_steps"
DEFAULT_HOST_OUT = Path("D:/platform-tools/GitHub/NPU/outputs/saved_step_candidate_validation")
PROFILE_FILE_RE = re.compile(r"^Input Log File Location:\s*(.+)$")
PROFILE_GRAPH_RE = re.compile(r"^Graph \d+ \((.+?)\):$")
PROFILE_NAMED_US_RE = re.compile(r"^\s*(.+?):\s*(\d+) us$")
PROFILE_NAMED_CYCLES_RE = re.compile(r"^\s*(.+?):\s*(\d+) cycles$")
PROFILE_NAMED_COUNT_RE = re.compile(r"^\s*(.+?):\s*(\d+) count$")


@dataclass
class StepResult:
    step: int
    return_code: int
    elapsed_ms: float
    thermals: dict[str, float]
    output_dir: str
    control_output_exists: bool
    candidate_output_exists: bool
    control_stats: dict[str, Any] | None = None
    candidate_stats: dict[str, Any] | None = None
    diff_vs_control: dict[str, Any] | None = None
    profile_metrics: dict[str, Any] | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a phone-side candidate UNet context against already-saved working phone step inputs, "
            "then optionally compare candidate noise_pred outputs against the control outputs from the "
            "working runtime."
        )
    )
    ap.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    ap.add_argument("--serial", default="e01ad23a")
    ap.add_argument("--control-base", default=DEFAULT_PHONE_CONTROL_BASE)
    ap.add_argument("--candidate-base", default=DEFAULT_PHONE_CANDIDATE_BASE)
    ap.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--context", default=DEFAULT_CONTEXT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--steps", default="1,2", help="Comma-separated step indices and ranges, e.g. 0,1,2 or 1-7")
    ap.add_argument("--perf-profile", default="sustained_high_performance")
    ap.add_argument("--log-level", default="warn")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="How many saved steps to evaluate per qnn-net-run process. 0 means batch all requested steps together.",
    )
    ap.add_argument("--use-mmap", action="store_true", default=True)
    ap.add_argument("--no-use-mmap", dest="use_mmap", action="store_false")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--compare-control", action="store_true", default=True)
    ap.add_argument("--no-compare-control", dest="compare_control", action="store_false")
    ap.add_argument("--host-out", type=Path, default=DEFAULT_HOST_OUT)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument(
        "--profiling-level",
        choices=["off", "basic", "detailed"],
        default="off",
        help="Optional qnn-net-run profiling level for each batch.",
    )
    ap.add_argument(
        "--profile-viewer",
        default="",
        help="Optional on-device qnn-profile-viewer path. Auto-detected when profiling is enabled.",
    )
    return ap.parse_args()


def parse_step_spec(spec: str) -> list[int]:
    steps: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            steps.update(range(start, end + 1))
        else:
            steps.add(int(part))
    return sorted(steps)


def run_adb(adb: Path, serial: str, args: list[str], *, capture: bool = True, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [str(adb), "-s", serial] + args
    cp = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"ADB failed ({cp.returncode}): {' '.join(cmd)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    return cp


def run_su(adb: Path, serial: str, command: str, *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_adb(
        adb,
        serial,
        ["shell", f"su --mount-master -c {shlex.quote(command)}"],
        timeout=timeout,
        check=check,
    )


def normalize_temp(raw: str) -> float | None:
    try:
        value = float(raw.strip())
    except Exception:
        return None
    if abs(value) >= 1000:
        value /= 1000.0
    if value <= 0:
        return None
    return value


def temp_group(zone_type: str) -> str | None:
    zt = zone_type.lower()
    if zt.startswith("cpu-") or zt.startswith("cpuss-"):
        return "CPU"
    if zt.startswith("gpuss-") or zt.startswith("gpu") or "kgsl" in zt:
        return "GPU"
    if zt.startswith("nsphvx-") or zt.startswith("nsphmx-") or zt.startswith("nsp"):
        return "NPU"
    return None


def phone_thermals(adb: Path, serial: str) -> dict[str, float]:
    cmd = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "if [ -f \"$z/type\" ] && [ -f \"$z/temp\" ]; then "
        "type=$(cat \"$z/type\" 2>/dev/null); temp=$(cat \"$z/temp\" 2>/dev/null); "
        "echo \"$type|$temp\"; fi; done"
    )
    cp = run_su(adb, serial, cmd, timeout=120)
    out: dict[str, float] = {}
    for line in cp.stdout.splitlines():
        if "|" not in line:
            continue
        zone_type, raw = line.split("|", 1)
        group = temp_group(zone_type)
        temp_c = normalize_temp(raw)
        if group is None or temp_c is None:
            continue
        if group not in out or temp_c > out[group]:
            out[group] = temp_c
    return out


def tensor_stats(x: np.ndarray) -> dict[str, Any]:
    xf = x.astype(np.float64, copy=False)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(np.isfinite(xf).sum()),
        "min": float(np.nanmin(xf)),
        "max": float(np.nanmax(xf)),
        "mean": float(np.nanmean(xf)),
        "std": float(np.nanstd(xf)),
    }


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    mask = np.isfinite(av) & np.isfinite(bv)
    if not np.any(mask):
        return {"finite_overlap": 0}
    av = av[mask]
    bv = bv[mask]
    diff = bv - av
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    cosine = float(np.dot(av, bv) / denom) if denom else float("nan")
    return {
        "finite_overlap": int(mask.sum()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine,
    }


def read_noise_pred(raw_path: Path) -> np.ndarray:
    arr = np.fromfile(raw_path, dtype=np.float32)
    expected = 1 * 128 * 128 * 4
    if arr.size != expected:
        raise RuntimeError(f"Unexpected output size for {raw_path}: got {arr.size}, expected {expected}")
    return np.transpose(arr.reshape(1, 128, 128, 4), (0, 3, 1, 2)).astype(np.float32, copy=False)


def build_run_command(
    *,
    candidate_base: str,
    context: str,
    config: str,
    input_list: str,
    output_dir: str,
    perf_profile: str,
    profiling_level: str,
    log_level: str,
    use_mmap: bool,
) -> str:
    mmap_flag = " --use_mmap" if use_mmap else ""
    config_flag = f" --config_file {config}" if config else ""
    profiling_flag = f" --profiling_level {profiling_level}" if profiling_level and profiling_level != "off" else ""
    return (
        f"cd {candidate_base} && "
        f"export LD_LIBRARY_PATH={candidate_base}/lib:{candidate_base}/bin:{candidate_base}/model && "
        f"export ADSP_LIBRARY_PATH='{candidate_base}/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp' && "
        f"rm -rf {output_dir} && mkdir -p {output_dir} && "
        f"{candidate_base}/bin/qnn-net-run "
        f"--retrieve_context {context} "
        f"--backend {candidate_base}/lib/libQnnHtp.so "
        f"--input_list {input_list} "
        f"--output_dir {output_dir} "
        f"--perf_profile {perf_profile}{config_flag}{mmap_flag}{profiling_flag} --log_level {log_level}"
    )


def chunked_steps(steps: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        return [steps]
    return [steps[i : i + batch_size] for i in range(0, len(steps), batch_size)]


def auto_profile_viewer(args: argparse.Namespace) -> str:
    if args.profile_viewer:
        return args.profile_viewer
    for candidate in (
        f"{args.candidate_base}/bin/qnn-profile-viewer",
        f"{args.control_base}/bin/qnn-profile-viewer",
    ):
        cp = run_su(args.adb, args.serial, f"test -f {candidate} && echo OK || echo MISSING", timeout=60, check=False)
        if cp.stdout.strip() == "OK":
            return candidate
    return ""


def parse_profile_viewer(text: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "sections": {},
        "graphs": {},
    }
    current_section: str | None = None
    current_graph: str | None = None
    current_subsection: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        file_match = PROFILE_FILE_RE.match(line)
        if file_match:
            profile["input_log"] = file_match.group(1).strip()
            continue
        if line.endswith("Stats:"):
            current_section = line.strip().rstrip(":")
            current_graph = None
            current_subsection = None
            continue
        stripped = line.strip()
        if stripped.endswith(":") and stripped in {
            "Init Stats:",
            "Compose Graphs Stats:",
            "Finalize Stats:",
            "De-Init Stats:",
            "Execute Stats (Overall):",
            "Execute Stats (Average):",
            "Execute Stats (Min):",
            "Execute Stats (Max):",
            "Total Inference Time:",
        }:
            current_subsection = stripped.rstrip(":")
            continue
        graph_match = PROFILE_GRAPH_RE.match(stripped)
        if graph_match:
            current_graph = graph_match.group(1)
            profile["graphs"].setdefault(current_graph, {})
            continue
        match_us = PROFILE_NAMED_US_RE.match(line)
        if match_us:
            key = match_us.group(1).strip()
            value = int(match_us.group(2))
            if current_graph:
                bucket = profile["graphs"].setdefault(current_graph, {})
                if current_subsection:
                    bucket.setdefault(current_subsection, {})[key] = value
                else:
                    bucket[key] = value
            elif current_section:
                profile["sections"].setdefault(current_section, {})[key] = value
            continue
        match_cycles = PROFILE_NAMED_CYCLES_RE.match(line)
        if match_cycles:
            key = match_cycles.group(1).strip()
            value = int(match_cycles.group(2))
            if current_graph:
                bucket = profile["graphs"].setdefault(current_graph, {})
                if current_subsection:
                    bucket.setdefault(current_subsection, {})[key] = value
                else:
                    bucket[key] = value
            elif current_section:
                profile["sections"].setdefault(current_section, {})[key] = value
            continue
        match_count = PROFILE_NAMED_COUNT_RE.match(line)
        if match_count:
            key = match_count.group(1).strip()
            value = int(match_count.group(2))
            if current_graph:
                bucket = profile["graphs"].setdefault(current_graph, {})
                if current_subsection:
                    bucket.setdefault(current_subsection, {})[key] = value
                else:
                    bucket[key] = value
            elif current_section:
                profile["sections"].setdefault(current_section, {})[key] = value
            continue
        if stripped.startswith("NetRun IPS"):
            try:
                value = float(stripped.split(":", 1)[1].split()[0])
            except Exception:
                continue
            profile["sections"].setdefault("Execute Stats (Overall)", {})["NetRun IPS"] = value
    return profile


def normalize_profile_key(key: str) -> str:
    normalized = key.strip()
    if normalized.startswith("Backend (") and normalized.endswith(")"):
        normalized = normalized[len("Backend (") : -1].strip()
    return normalized


def extract_profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    sections = profile.get("sections", {})
    graphs = profile.get("graphs", {})
    avg_graph = next(iter(graphs.values()), {}).get("Total Inference Time", {})
    init_section = {normalize_profile_key(k): v for k, v in sections.get("Init Stats", {}).items()}
    deinit_section = {normalize_profile_key(k): v for k, v in sections.get("De-Init Stats", {}).items()}
    overall_section = {normalize_profile_key(k): v for k, v in sections.get("Execute Stats (Overall)", {}).items()}
    avg_graph = {normalize_profile_key(k): v for k, v in avg_graph.items()}
    return {
        "init_netrun_us": init_section.get("NetRun"),
        "deinit_netrun_us": deinit_section.get("NetRun"),
        "load_binary_qnn_us": init_section.get("QNN (load binary) time"),
        "load_binary_rpc_us": init_section.get("RPC (load binary) time"),
        "ips": overall_section.get("NetRun IPS"),
        "exec_netrun_us": avg_graph.get("NetRun"),
        "exec_rpc_us": avg_graph.get("RPC (execute) time"),
        "exec_qnn_accel_us": avg_graph.get("QNN accelerator (execute) time"),
        "exec_accel_cycles": avg_graph.get("Accelerator (execute) time (cycles)"),
        "hvx_threads": avg_graph.get("Number of HVX threads used"),
        "yield_count": avg_graph.get("Num times yield occured"),
    }


def read_profile_metrics(
    args: argparse.Namespace,
    *,
    profile_viewer: str,
    batch_output_dir: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if args.profiling_level == "off" or not profile_viewer:
        return None, None
    profile_log = f"{batch_output_dir}/qnn-profiling-data_0.log"
    exists = run_su(args.adb, args.serial, f"test -f {profile_log} && echo OK || echo MISSING", timeout=60, check=False)
    if exists.stdout.strip() != "OK":
        return None, None
    cp = run_su(args.adb, args.serial, f"{profile_viewer} --input_log {profile_log}", timeout=300, check=False)
    viewer_text = (cp.stdout or "").strip()
    if cp.returncode != 0 or not viewer_text:
        return None, viewer_text or (cp.stderr or "").strip() or None
    parsed = parse_profile_viewer(viewer_text)
    return extract_profile_metrics(parsed), viewer_text


def main() -> None:
    args = parse_args()
    steps = parse_step_spec(args.steps)
    host_out = args.host_out / time.strftime("saved_step_candidate_validation_%Y%m%d_%H%M%S")
    host_out.mkdir(parents=True, exist_ok=True)
    profile_viewer = auto_profile_viewer(args) if args.profiling_level != "off" else ""

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": steps,
        "adb": str(args.adb),
        "serial": args.serial,
        "control_base": args.control_base,
        "candidate_base": args.candidate_base,
        "input_root": args.input_root,
        "context": args.context,
        "config": args.config,
        "output_root": args.output_root,
        "perf_profile": args.perf_profile,
        "profiling_level": args.profiling_level,
        "profile_viewer": profile_viewer,
        "batch_size": args.batch_size,
        "use_mmap": args.use_mmap,
        "results": [],
    }

    start_thermals = phone_thermals(args.adb, args.serial)
    summary["thermals_start"] = start_thermals
    max_thermals = dict(start_thermals)

    for batch_index, batch in enumerate(chunked_steps(steps, args.batch_size), start=1):
        batch_input_list = f"{args.candidate_base}/candidate_saved_steps_batch_{batch_index:03d}.txt"
        batch_output_dir = f"{args.output_root}/batch_{batch_index:03d}"
        build_batch_cmd = "; ".join(
            [f"cat {args.input_root}/step_{step:03d}/input_list.txt {'>' if i == 0 else '>>'} {batch_input_list}"
             for i, step in enumerate(batch)]
        ) + f"; chmod 666 {batch_input_list}"
        run_su(args.adb, args.serial, build_batch_cmd, timeout=120)

        need_run = True
        if args.skip_existing:
            existing_flags = []
            for result_idx, step in enumerate(batch):
                candidate_raw = f"{batch_output_dir}/Result_{result_idx}/noise_pred.raw"
                existing_flags.append(
                    run_su(args.adb, args.serial, f"test -f {candidate_raw}", timeout=60, check=False).returncode == 0
                )
            need_run = not all(existing_flags)

        if need_run:
            cmd = build_run_command(
                candidate_base=args.candidate_base,
                context=args.context,
                config=args.config,
                input_list=batch_input_list,
                output_dir=batch_output_dir,
                perf_profile=args.perf_profile,
                profiling_level=args.profiling_level,
                log_level=args.log_level,
                use_mmap=args.use_mmap,
            )
            t0 = time.perf_counter()
            cp = run_su(args.adb, args.serial, cmd, timeout=args.timeout_sec, check=False)
            batch_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        else:
            cp = subprocess.CompletedProcess([], 0, stdout="[skip] existing candidate batch output", stderr="")
            batch_elapsed_ms = 0.0

        batch_profile_metrics, batch_profile_viewer_text = read_profile_metrics(
            args,
            profile_viewer=profile_viewer,
            batch_output_dir=batch_output_dir,
        )

        thermals = phone_thermals(args.adb, args.serial)
        for k, v in thermals.items():
            if k not in max_thermals or v > max_thermals[k]:
                max_thermals[k] = v

        per_step_elapsed_ms = batch_elapsed_ms / len(batch) if batch else 0.0
        for result_idx, step in enumerate(batch):
            candidate_raw = f"{batch_output_dir}/Result_{result_idx}/noise_pred.raw"
            control_raw = f"{args.input_root}/step_{step:03d}/output/Result_0/noise_pred.raw"
            host_step_dir = host_out / f"step_{step:03d}"
            host_step_dir.mkdir(parents=True, exist_ok=True)

            candidate_exists = run_su(args.adb, args.serial, f"test -f {candidate_raw}", timeout=60, check=False).returncode == 0
            control_exists = run_su(args.adb, args.serial, f"test -f {control_raw}", timeout=60, check=False).returncode == 0

            result = StepResult(
                step=step,
                return_code=cp.returncode,
                elapsed_ms=per_step_elapsed_ms,
                thermals=thermals,
                output_dir=f"{batch_output_dir}/Result_{result_idx}",
                control_output_exists=control_exists,
                candidate_output_exists=candidate_exists,
                profile_metrics=batch_profile_metrics,
                stdout_tail="\n".join((cp.stdout or "").splitlines()[-40:]) if cp.stdout else None,
                stderr_tail="\n".join((cp.stderr or "").splitlines()[-40:]) if cp.stderr else None,
            )

            if args.compare_control and candidate_exists and control_exists:
                cand_local = host_step_dir / "candidate_noise_pred.raw"
                ctrl_local = host_step_dir / "control_noise_pred.raw"
                run_adb(args.adb, args.serial, ["pull", candidate_raw, str(cand_local)], timeout=300)
                run_adb(args.adb, args.serial, ["pull", control_raw, str(ctrl_local)], timeout=300)
                cand = read_noise_pred(cand_local)
                ctrl = read_noise_pred(ctrl_local)
                result.candidate_stats = tensor_stats(cand)
                result.control_stats = tensor_stats(ctrl)
                result.diff_vs_control = diff_metrics(ctrl, cand)

            summary["results"].append(asdict(result))

            print(f"[step {step:03d}] rc={result.return_code} elapsed={result.elapsed_ms/1000.0:.1f}s thermals={result.thermals}")
            if result.profile_metrics:
                pm = result.profile_metrics
                exec_us = pm.get("exec_netrun_us")
                cycles = pm.get("exec_accel_cycles")
                hvx = pm.get("hvx_threads")
                if exec_us or cycles:
                    exec_str = f" exec={exec_us/1_000_000.0:.3f}s" if exec_us else ""
                    cycles_str = f" cycles={cycles}" if cycles else ""
                    hvx_str = f" hvx={hvx}" if hvx else ""
                    print(f"  profile:{exec_str}{cycles_str}{hvx_str}")
            if result.diff_vs_control:
                d = result.diff_vs_control
                print(
                    f"  diff: cosine={d.get('cosine', float('nan')):.6f} "
                    f"rmse={d.get('rmse', float('nan')):.6f} mae={d.get('mae', float('nan')):.6f}"
                )
        if batch_profile_viewer_text:
            (host_out / f"batch_{batch_index:03d}_profile_viewer.txt").write_text(
                batch_profile_viewer_text,
                encoding="utf-8",
            )

    summary["thermals_max"] = max_thermals
    ok = [r for r in summary["results"] if r["return_code"] == 0]
    summary["ok_steps"] = [r["step"] for r in ok]
    summary["failed_steps"] = [r["step"] for r in summary["results"] if r["return_code"] != 0]
    summary["avg_elapsed_ms"] = float(np.mean([r["elapsed_ms"] for r in ok])) if ok else math.nan

    out_json = host_out / "report.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] report: {out_json}")


if __name__ == "__main__":
    main()
