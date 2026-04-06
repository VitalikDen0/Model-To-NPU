#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sdxl_speed_probe import adb_shell, detect_phone_base, phone_info, run_command

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_tools" / "outputs"
DEFAULT_ADB = Path(r"D:\platform-tools\adb.exe")
DEFAULT_PHONE_BASE = "/sdcard/Download/sdxl_qnn"


@dataclass(frozen=True)
class ProbeCase:
    name: str
    context_rel: str
    input_list_rel: str
    use_native_output: bool = False
    use_mmap: bool = False
    num_inferences: int = 1
    keep_num_outputs: int = 1


CASES = (
    ProbeCase("encoder_single", "context/unet_encoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_enc.txt"),
    ProbeCase("decoder_single", "context/unet_decoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_dec.txt"),
    ProbeCase("encoder_batch_cfg", "context/unet_encoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_enc_batch.txt", num_inferences=2, keep_num_outputs=2),
    ProbeCase("decoder_batch_cfg", "context/unet_decoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_dec_batch.txt", num_inferences=2, keep_num_outputs=2),
    ProbeCase("encoder_single_mmap", "context/unet_encoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_enc.txt", use_mmap=True),
    ProbeCase("decoder_single_mmap", "context/unet_decoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_dec.txt", use_mmap=True),
    ProbeCase("encoder_single_repeat4", "context/unet_encoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_enc.txt", num_inferences=4, keep_num_outputs=1),
    ProbeCase("decoder_single_repeat4", "context/unet_decoder_fp16.serialized.bin.bin", "phone_gen/work_speed_probe/unet/cond/il_dec.txt", num_inferences=4, keep_num_outputs=1),
)

PROFILE_INIT_RE = re.compile(r"^\s*NetRun:\s*(\d+) us$")
PROFILE_BACKEND_RE = re.compile(r"^\s*Backend \((.+?)\):\s*(\d+) us$")
PROFILE_GRAPH_RE = re.compile(r"^Graph \d+ \((.+?)\):$")
PROFILE_FILE_RE = re.compile(r"^Input Log File Location:\s*(.+)$")


def remote_script(lines: list[str]) -> str:
    body = "\n".join(lines)
    return f"su --mount-master -c {shlex.quote(body)}"


def run_remote(adb_path: Path, lines: list[str], timeout: int = 0) -> tuple[str, float, int]:
    cmd = [str(adb_path), "shell", remote_script(lines)]
    completed, elapsed = run_command(cmd, timeout=timeout)
    stdout = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return stdout.strip(), elapsed * 1000.0, completed.returncode


def parse_profile_viewer(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sections": {},
        "graphs": {},
    }
    current_section: str | None = None
    current_graph: str | None = None
    current_subsection: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        file_match = PROFILE_FILE_RE.match(line)
        if file_match:
            data["input_log"] = file_match.group(1).strip()
            continue
        if line.endswith("Stats:"):
            current_section = line.strip().rstrip(":")
            current_graph = None
            current_subsection = None
            continue
        if line.strip().endswith(":") and line.strip() in {
            "Init Stats:", "Compose Graphs Stats:", "Finalize Stats:", "De-Init Stats:",
            "Execute Stats (Overall):", "Execute Stats (Average):", "Execute Stats (Min):", "Execute Stats (Max):",
            "Total Inference Time:",
        }:
            current_subsection = line.strip().rstrip(":")
            continue
        graph_match = PROFILE_GRAPH_RE.match(line.strip())
        if graph_match:
            current_graph = graph_match.group(1)
            data["graphs"].setdefault(current_graph, {})
            continue
        init_match = PROFILE_INIT_RE.match(line)
        if init_match and current_section and not current_graph:
            data["sections"].setdefault(current_section, {})["NetRun_us"] = int(init_match.group(1))
            continue
        backend_match = PROFILE_BACKEND_RE.match(line)
        if backend_match:
            key = backend_match.group(1).strip()
            value = int(backend_match.group(2))
            if current_graph:
                graph_bucket = data["graphs"].setdefault(current_graph, {})
                if current_subsection:
                    graph_bucket.setdefault(current_subsection, {})[key] = value
                else:
                    graph_bucket[key] = value
            elif current_section:
                data["sections"].setdefault(current_section, {})[key] = value
            continue
        if line.strip().startswith("NetRun IPS"):
            try:
                value = float(line.split(":", 1)[1].split()[0])
                data["sections"].setdefault("Execute Stats (Overall)", {})["NetRun_IPS"] = value
            except Exception:
                pass
            continue
        if line.strip().startswith("NetRun:") and current_graph:
            try:
                value = int(line.split(":", 1)[1].strip().split()[0])
                graph_bucket = data["graphs"].setdefault(current_graph, {})
                if current_subsection:
                    graph_bucket.setdefault(current_subsection, {})["NetRun_us"] = value
            except Exception:
                pass
    return data


def extract_key_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    sections = profile.get("sections", {})
    graphs = profile.get("graphs", {})
    avg_graph = next(iter(graphs.values()), {}).get("Total Inference Time", {})
    result = {
        "init_netrun_us": sections.get("Init Stats", {}).get("NetRun_us"),
        "deinit_netrun_us": sections.get("De-Init Stats", {}).get("NetRun_us"),
        "ips": sections.get("Execute Stats (Overall)", {}).get("NetRun_IPS"),
        "exec_netrun_us": avg_graph.get("NetRun_us"),
        "exec_rpc_us": avg_graph.get("RPC (execute) time"),
        "exec_accel_us": avg_graph.get("Accelerator (execute) time"),
        "exec_accel_excluding_wait_us": avg_graph.get("Accelerator (execute excluding wait) time"),
        "exec_qnn_us": avg_graph.get("QNN (execute) time"),
        "load_binary_qnn_us": sections.get("Init Stats", {}).get("QNN (load binary) time"),
        "load_binary_rpc_us": sections.get("Init Stats", {}).get("RPC (load binary) time"),
    }
    return result


def output_file_count_info(adb_path: Path, output_dir: str) -> dict[str, Any]:
    lines = [
        f"find {shlex.quote(output_dir)} -type f | wc -l",
        f"find {shlex.quote(output_dir)} -type f -exec wc -c {{}} + | tail -n 1",
    ]
    out, _, code = run_remote(adb_path, lines, timeout=30)
    if code != 0:
        return {"raw": out}
    parts = [x.strip() for x in out.splitlines() if x.strip()]
    result: dict[str, Any] = {"raw": out}
    if parts:
        try:
            result["file_count"] = int(parts[0].split()[0])
        except Exception:
            pass
    if len(parts) > 1:
        try:
            result["total_bytes"] = int(parts[1].split()[0])
        except Exception:
            pass
    return result


def run_case(adb_path: Path, phone_base: str, case: ProbeCase, timeout: int) -> dict[str, Any]:
    output_dir = f"{phone_base}/phone_gen/debug_overhead/{case.name}"
    context = f"{phone_base}/{case.context_rel}"
    input_list = f"{phone_base}/{case.input_list_rel}"
    profile_log = f"{output_dir}/qnn-profiling-data_0.log"

    base_cmd = [
        f"{phone_base}/bin/qnn-net-run",
        "--retrieve_context", context,
        "--backend", f"{phone_base}/lib/libQnnHtp.so",
        "--input_list", input_list,
        "--output_dir", output_dir,
        "--perf_profile", "burst",
        "--log_level", "verbose",
        "--profiling_level", "basic",
        "--num_inferences", str(case.num_inferences),
        "--keep_num_outputs", str(case.keep_num_outputs),
    ]
    if case.use_mmap:
        base_cmd.append("--use_mmap")
    if case.use_native_output:
        base_cmd.append("--use_native_output_files")

    lines = [
        f"export LD_LIBRARY_PATH={phone_base}/lib:{phone_base}/bin:{phone_base}/model",
        f"export ADSP_LIBRARY_PATH={phone_base}/lib\\;/vendor/lib64/rfs/dsp\\;/vendor/lib/rfsa/adsp\\;/vendor/dsp",
        f"rm -rf {shlex.quote(output_dir)}",
        " ".join(shlex.quote(x) for x in base_cmd),
    ]
    stdout, wall_ms, code = run_remote(adb_path, lines, timeout=timeout)
    if code != 0:
        return {
            "name": case.name,
            "error": stdout,
            "wall_ms": wall_ms,
        }

    viewer_lines = [
        f"{phone_base}/bin/qnn-profile-viewer --input_log {shlex.quote(profile_log)}"
    ]
    viewer_text, viewer_ms, viewer_code = run_remote(adb_path, viewer_lines, timeout=timeout)
    result: dict[str, Any] = {
        "name": case.name,
        "wall_ms": wall_ms,
        "viewer_ms": viewer_ms,
        "stdout_tail": "\n".join(stdout.splitlines()[-40:]),
        "output_dir": output_dir,
        "context": context,
        "input_list": input_list,
        "use_mmap": case.use_mmap,
        "num_inferences": case.num_inferences,
        "keep_num_outputs": case.keep_num_outputs,
        "output_files": output_file_count_info(adb_path, output_dir),
    }
    if viewer_code == 0:
        result["profile_viewer"] = viewer_text
        parsed = parse_profile_viewer(viewer_text)
        result["profile_parsed"] = parsed
        result["profile_metrics"] = extract_key_metrics(parsed)
        metrics = result["profile_metrics"]
        init_us = metrics.get("init_netrun_us") or 0
        exec_us = metrics.get("exec_netrun_us") or 0
        deinit_us = metrics.get("deinit_netrun_us") or 0
        accounted_us = init_us + exec_us * case.num_inferences + deinit_us
        metrics["profile_accounted_total_us"] = accounted_us
        metrics["profile_accounted_per_inference_us"] = accounted_us / case.num_inferences if case.num_inferences > 0 else None
        result["residual_overhead_ms"] = ((wall_ms * 1000.0 - accounted_us) / 1000.0) if accounted_us else None
        if case.num_inferences > 1:
            result["wall_per_inference_ms"] = wall_ms / case.num_inferences
            result["profile_accounted_per_inference_ms"] = accounted_us / 1000.0 / case.num_inferences if accounted_us else None
    else:
        result["profile_viewer_error"] = viewer_text
    return result


def build_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# SDXL UNet overhead probe — {report['timestamp']}")
    lines.append("")
    phone = report.get("phone_info", {})
    if phone:
        lines.append(f"- Phone: **{phone.get('ro.product.manufacturer','')} {phone.get('ro.product.model','')}** / {phone.get('ro.soc.model','')} ({phone.get('ro.board.platform','')})")
    lines.append(f"- Phone base: `{report['phone_base']}`")
    lines.append(f"- Work dir: `{report['work_dir']}`")
    lines.append("")
    lines.append("## Probe results")
    lines.append("")
    lines.append("| Case | Wall | Init | Execute | Deinit | Residual | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for case in report.get("cases", []):
        if case.get("error"):
            lines.append(f"| {case['name']} | error | — | — | — | — | failed |")
            continue
        m = case.get("profile_metrics", {})
        note_parts = []
        if case.get("use_mmap"):
            note_parts.append("mmap")
        if case.get("num_inferences", 1) > 1:
            note_parts.append(f"repeat×{case['num_inferences']}")
        if case["name"].endswith("batch_cfg"):
            note_parts.append("batched CFG")
        lines.append(
            "| {name} | {wall:.1f}ms | {init} | {exe} | {deinit} | {res} | {notes} |".format(
                name=case["name"],
                wall=case.get("wall_ms", 0.0),
                init=f"{m.get('init_netrun_us', 0)/1000:.1f}ms" if m.get("init_netrun_us") else "—",
                exe=f"{m.get('exec_netrun_us', 0)/1000:.1f}ms" if m.get("exec_netrun_us") else "—",
                deinit=f"{m.get('deinit_netrun_us', 0)/1000:.1f}ms" if m.get("deinit_netrun_us") else "—",
                res=f"{case.get('residual_overhead_ms', 0):.1f}ms" if case.get("residual_overhead_ms") is not None else "—",
                notes=", ".join(note_parts) if note_parts else "baseline",
            )
        )
    lines.append("")

    case_map = {case["name"]: case for case in report.get("cases", []) if not case.get("error")}
    enc = case_map.get("encoder_single", {})
    dec = case_map.get("decoder_single", {})
    enc_rep = case_map.get("encoder_single_repeat4", {})
    dec_rep = case_map.get("decoder_single_repeat4", {})
    enc_mmap = case_map.get("encoder_single_mmap", {})
    dec_mmap = case_map.get("decoder_single_mmap", {})
    batch_enc = case_map.get("encoder_batch_cfg", {})
    batch_dec = case_map.get("decoder_batch_cfg", {})

    def pct_delta(a: float | None, b: float | None) -> str:
        if not a or not b:
            return "—"
        return f"{((a - b) / a) * 100:.1f}%"

    def ratio(a: float | None, b: float | None) -> str:
        if not a or not b:
            return "—"
        return f"{a / b:.2f}×"

    lines.append("## What stands out")
    lines.append("")
    if enc and dec:
        lines.append(f"- Single no-CFG UNet half-step costs: encoder **{enc.get('wall_ms', 0):.1f}ms**, decoder **{dec.get('wall_ms', 0):.1f}ms**.")
    if enc_rep and enc:
        lines.append(f"- Encoder repeat×4 average vs single: **{enc_rep.get('wall_per_inference_ms', 0):.1f}ms** vs **{enc.get('wall_ms', 0):.1f}ms** ({pct_delta(enc.get('wall_ms'), enc_rep.get('wall_per_inference_ms'))} saved per inference).")
    if dec_rep and dec:
        lines.append(f"- Decoder repeat×4 average vs single: **{dec_rep.get('wall_per_inference_ms', 0):.1f}ms** vs **{dec.get('wall_ms', 0):.1f}ms** ({pct_delta(dec.get('wall_ms'), dec_rep.get('wall_per_inference_ms'))} saved per inference).")
    if enc_mmap and enc:
        lines.append(f"- Encoder mmap effect: **{enc_mmap.get('wall_ms', 0):.1f}ms** vs **{enc.get('wall_ms', 0):.1f}ms** (gain {pct_delta(enc.get('wall_ms'), enc_mmap.get('wall_ms'))}).")
    if dec_mmap and dec:
        lines.append(f"- Decoder mmap effect: **{dec_mmap.get('wall_ms', 0):.1f}ms** vs **{dec.get('wall_ms', 0):.1f}ms** (gain {pct_delta(dec.get('wall_ms'), dec_mmap.get('wall_ms'))}).")
    if batch_enc and enc:
        lines.append(f"- Encoder batched CFG total vs single: **{batch_enc.get('wall_ms', 0):.1f}ms** vs **{enc.get('wall_ms', 0):.1f}ms** ({ratio(batch_enc.get('wall_ms'), enc.get('wall_ms'))} of single-call cost, but for two inferences).")
    if batch_dec and dec:
        lines.append(f"- Decoder batched CFG total vs single: **{batch_dec.get('wall_ms', 0):.1f}ms** vs **{dec.get('wall_ms', 0):.1f}ms** ({ratio(batch_dec.get('wall_ms'), dec.get('wall_ms'))} of single-call cost, but for two inferences).")
    lines.append("")
    lines.append("Interpretation: if repeat×4 average collapses much lower than single-call wall time, the killer is context/process lifecycle, not pure NPU math. If mmap helps a lot, binary/context file I/O is one of the few cheap wins you should take immediately.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deep UNet overhead probe on phone QNN runtime")
    ap.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    ap.add_argument("--phone-base", default=DEFAULT_PHONE_BASE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--timeout-sec", type=int, default=240)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    phone_base = detect_phone_base(args.adb, args.phone_base)
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phone_base": phone_base,
        "work_dir": f"{phone_base}/phone_gen/work_speed_probe/unet",
        "phone_info": phone_info(args.adb),
        "cases": [],
    }
    for case in CASES:
        print(f"[probe] {case.name} ...", flush=True)
        report["cases"].append(run_case(args.adb, phone_base, case, args.timeout_sec))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sdxl_unet_overhead_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_summary(report), encoding="utf-8")
    print(f"[ok] JSON: {json_path}")
    print(f"[ok] Markdown: {md_path}")
    print(build_summary(report))


if __name__ == "__main__":
    main()
