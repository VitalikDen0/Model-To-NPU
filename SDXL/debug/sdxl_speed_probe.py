#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SDXL_ROOT = Path(os.environ.get("SDXL_NPU_ROOT", r"D:\platform-tools\sdxl_npu"))
DEFAULT_DIFFUSERS_DIR = DEFAULT_SDXL_ROOT / "diffusers_pipeline"
DEFAULT_MERGED_UNET_DIR = DEFAULT_SDXL_ROOT / "unet_lightning8step_merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_tools" / "outputs"
DEFAULT_ADB = Path(os.environ.get("ADB_PATH", r"D:\platform-tools\adb.exe"))
DEFAULT_PHONE_BASE = os.environ.get("SDXL_QNN_BASE", "/sdcard/Download/sdxl_qnn")
DEFAULT_TERMUX_PYTHON = "/data/data/com.termux/files/usr/bin/python3"
DEFAULT_NEGATIVE = (
    "(worst quality, low quality, normal quality:1.4), "
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, "
    "wrong anatomy, extra limb, missing limb, floating limbs, "
    "(mutated hands and fingers:1.4), disconnected limbs, mutation, "
    "mutated, ugly, disgusting, blurry, amputation, text, watermark, "
    "signature, censor, censored, bar"
)


@dataclass(frozen=True)
class Variant:
    name: str
    cfg: float
    progressive_cfg: bool


VARIANTS = (
    Variant("cfg1", 1.0, False),
    Variant("cfg2", 2.0, False),
    Variant("cfg2_prog", 2.0, True),
)


def resolve_variants(spec: str) -> tuple[Variant, ...]:
    if not spec.strip():
        return VARIANTS
    wanted = [name.strip() for name in spec.split(",") if name.strip()]
    available = {variant.name: variant for variant in VARIANTS}
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(
            "Unknown variant(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(available)
        )
    return tuple(available[name] for name in wanted)


def run_command(cmd: list[str], *, cwd: Path | None = None, timeout: int = 0, env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
    )
    elapsed = time.perf_counter() - started
    return completed, elapsed


def nvidia_smi_info() -> dict[str, Any]:
    try:
        completed, _ = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        if completed.returncode != 0:
            return {"error": completed.stderr.strip() or completed.stdout.strip()}
        rows = []
        for line in completed.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                rows.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_mib": float(parts[2]),
                        "power_limit_w": float(parts[3]),
                    }
                )
        return {"gpus": rows}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def adb_shell(adb_path: Path, shell_command: str, *, timeout: int = 0, check: bool = True) -> tuple[str, float]:
    completed, elapsed = run_command([str(adb_path), "shell", shell_command], timeout=timeout)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if check and completed.returncode != 0:
        raise RuntimeError(f"adb shell failed ({completed.returncode}): {stderr or stdout}")
    return (stdout + ("\n" + stderr if stderr else "")).strip(), elapsed


def phone_info(adb_path: Path) -> dict[str, Any]:
    props: dict[str, str] = {}
    for prop in [
        "ro.product.manufacturer",
        "ro.product.model",
        "ro.board.platform",
        "ro.soc.model",
        "ro.build.version.release",
    ]:
        try:
            out, _ = adb_shell(adb_path, f"getprop {prop}", timeout=10, check=False)
            props[prop] = out.strip()
        except Exception as exc:  # noqa: BLE001
            props[prop] = f"ERR: {exc}"
    return props


def detect_phone_base(adb_path: Path, preferred_base: str) -> str:
    candidates = [
        preferred_base,
        "/sdcard/Download/sdxl_qnn",
        "/data/local/tmp/sdxl_qnn",
    ]
    required = [
        "phone_gen/generate.py",
        "phone_gen/tokenizer/vocab.json",
        "context/clip_l.serialized.bin.bin",
        "bin/qnn-net-run",
    ]
    for base in dict.fromkeys(candidates):
        checks = " && ".join([f"test -e {shlex.quote(base + '/' + rel)}" for rel in required])
        out, _ = adb_shell(adb_path, f"su --mount-master -c {shlex.quote(checks + ' && echo OK || echo MISS')}", timeout=15, check=False)
        if "OK" in out:
            return base
    raise RuntimeError(
        "Could not find a complete phone runtime base. Checked: " + ", ".join(dict.fromkeys(candidates))
    )


def detect_phone_script(adb_path: Path) -> str:
    candidates = [
        "/sdcard/Download/sdxl_qnn/phone_gen/generate.py",
        "/data/local/tmp/sdxl_qnn/phone_gen/generate.py",
    ]
    for script_path in candidates:
        out, _ = adb_shell(adb_path, f"su --mount-master -c {shlex.quote(f'test -e {shlex.quote(script_path)} && echo OK || echo MISS')}", timeout=15, check=False)
        if "OK" in out:
            return script_path
    raise RuntimeError("Could not find phone generate.py in known locations")


class PcLightningRuntime:
    def __init__(self, diffusers_dir: Path, merged_unet_dir: Path, device: str = "cuda") -> None:
        self.diffusers_dir = diffusers_dir
        self.merged_unet_dir = merged_unet_dir
        self.device_name = device
        self.load_ms: float | None = None
        self._load_runtime()

    def _load_runtime(self) -> None:
        started = time.perf_counter()
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the PC benchmark")

        self.torch = torch
        self.device = torch.device(self.device_name)
        self.tok_l = CLIPTokenizer.from_pretrained(str(self.diffusers_dir / "tokenizer"))
        self.tok_g = CLIPTokenizer.from_pretrained(str(self.diffusers_dir / "tokenizer_2"))
        self.te_l = CLIPTextModel.from_pretrained(
            str(self.diffusers_dir / "text_encoder"), torch_dtype=torch.float16
        ).to(self.device).eval()
        self.te_g = CLIPTextModelWithProjection.from_pretrained(
            str(self.diffusers_dir / "text_encoder_2"), torch_dtype=torch.float16
        ).to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            str(self.merged_unet_dir), torch_dtype=torch.float16
        ).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(
            str(self.diffusers_dir / "vae"), torch_dtype=torch.float16
        ).to(self.device).eval()
        sched_cfg = json.loads((self.diffusers_dir / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
        self.scheduler = EulerDiscreteScheduler.from_config(sched_cfg, timestep_spacing="trailing")
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _encode_prompt(self, prompt: str, negative_prompt: str | None) -> tuple[Any, Any, Any | None, Any | None, float]:
        torch = self.torch
        started = time.perf_counter()
        ids_l = self.tok_l(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
        ids_g = self.tok_g(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out_l = self.te_l(ids_l, output_hidden_states=True)
            hs_l = out_l.hidden_states[-2]
            out_g = self.te_g(ids_g, output_hidden_states=True)
            hs_g = out_g.hidden_states[-2]
            pooled = out_g.text_embeds

        prompt_embeds = torch.cat([hs_l, hs_g], dim=-1)
        neg_embeds = None
        neg_pooled = None
        if negative_prompt is not None:
            nids_l = self.tok_l(negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
            nids_g = self.tok_g(negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                nout_l = self.te_l(nids_l, output_hidden_states=True)
                nhs_l = nout_l.hidden_states[-2]
                nout_g = self.te_g(nids_g, output_hidden_states=True)
                nhs_g = nout_g.hidden_states[-2]
                neg_pooled = nout_g.text_embeds
            neg_embeds = torch.cat([nhs_l, nhs_g], dim=-1)
        self._sync()
        return prompt_embeds, pooled, neg_embeds, neg_pooled, (time.perf_counter() - started) * 1000.0

    def benchmark(self, *, prompt: str, negative_prompt: str | None, seed: int, steps: int, width: int, height: int, cfg_scale: float, progressive_cfg: bool) -> dict[str, Any]:
        torch = self.torch
        with torch.no_grad():
            prompt_embeds, pooled, neg_embeds, neg_pooled, clip_ms = self._encode_prompt(prompt, negative_prompt if cfg_scale > 1.0 else None)
            latent_h, latent_w = height // 8, width // 8
            gen = torch.Generator(device=self.device).manual_seed(seed)
            latents = torch.randn((1, 4, latent_h, latent_w), generator=gen, device=self.device, dtype=torch.float16)
            self.scheduler.set_timesteps(steps, device=self.device)
            latents = latents * self.scheduler.init_noise_sigma
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float16, device=self.device)
            added_cond = {"text_embeds": pooled.half(), "time_ids": time_ids}
            neg_added_cond = None
            if cfg_scale > 1.0 and neg_pooled is not None:
                neg_added_cond = {"text_embeds": neg_pooled.half(), "time_ids": time_ids}

            cfg_cutoff = math.ceil(steps / 2) if (cfg_scale > 1.0 and progressive_cfg) else steps
            step_times_ms: list[float] = []
            denoise_started = time.perf_counter()
            for idx, t in enumerate(self.scheduler.timesteps):
                step_started = time.perf_counter()
                lat_in = self.scheduler.scale_model_input(latents, t)
                step_uses_cfg = cfg_scale > 1.0 and idx < cfg_cutoff and neg_embeds is not None and neg_added_cond is not None
                if step_uses_cfg:
                    assert neg_embeds is not None and neg_added_cond is not None
                    lat_in_2 = torch.cat([lat_in, lat_in])
                    pe_2 = torch.cat([neg_embeds.half(), prompt_embeds.half()])
                    ac_2 = {
                        "text_embeds": torch.cat([neg_added_cond["text_embeds"], added_cond["text_embeds"]]),
                        "time_ids": torch.cat([neg_added_cond["time_ids"], added_cond["time_ids"]]),
                    }
                    noise_pred = self.unet(lat_in_2, t, encoder_hidden_states=pe_2, added_cond_kwargs=ac_2).sample
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = self.unet(lat_in, t, encoder_hidden_states=prompt_embeds.half(), added_cond_kwargs=added_cond).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                self._sync()
                step_times_ms.append((time.perf_counter() - step_started) * 1000.0)
            denoise_ms = (time.perf_counter() - denoise_started) * 1000.0

            vae_started = time.perf_counter()
            vae_sf = self.vae.config.scaling_factor
            _ = self.vae.decode(latents / vae_sf).sample
            self._sync()
            vae_ms = (time.perf_counter() - vae_started) * 1000.0

        total_ms = clip_ms + denoise_ms + vae_ms
        return {
            "clip_ms": clip_ms,
            "unet_total_ms": denoise_ms,
            "unet_avg_ms": float(np.mean(step_times_ms)),
            "unet_step_times_ms": [round(x, 3) for x in step_times_ms],
            "vae_ms": vae_ms,
            "total_ms_excluding_load": total_ms,
            "load_ms": self.load_ms,
            "total_ms_including_load": total_ms + (self.load_ms or 0.0),
            "cfg_cutoff": cfg_cutoff,
        }


def parse_phone_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    final_match = re.search(r"CLIP:\s*(\d+)ms\s*\|\s*UNet:\s*(\d+)ms\s*\|\s*VAE:\s*(\d+)ms", stdout)
    total_match = re.search(r"Total:\s*([0-9]+(?:\.[0-9]+)?)s", stdout)
    unet_match = re.search(r"UNet total:\s*(\d+)ms\s*\((\d+)ms/step\)", stdout)
    saved_match = re.search(r"Saved:\s*(.+)", stdout)
    step_times = [float(x) for x in re.findall(r"\[UNet\s+\d+/\d+\](?:\s+CFG)?(?:\s+\[[^\]]+\])?\s+([0-9]+(?:\.[0-9]+)?)ms", stdout)]
    clip_cond = re.search(r"\[CLIP cond\]\s*L=(\d+)ms\s*G=(\d+)ms", stdout)
    clip_uncond = re.search(r"\[CLIP uncond\]\s*L=(\d+)ms\s*G=(\d+)ms", stdout)

    if final_match:
        metrics["clip_ms"] = float(final_match.group(1))
        metrics["unet_total_ms"] = float(final_match.group(2))
        metrics["vae_ms"] = float(final_match.group(3))
    if total_match:
        metrics["total_ms"] = float(total_match.group(1)) * 1000.0
    if unet_match:
        metrics["unet_total_ms"] = float(unet_match.group(1))
        metrics["unet_avg_ms"] = float(unet_match.group(2))
    if saved_match:
        metrics["saved_path"] = saved_match.group(1).strip()
    if step_times:
        metrics["unet_step_times_ms"] = step_times
        metrics.setdefault("unet_avg_ms", float(np.mean(step_times)))
    if clip_cond:
        metrics["clip_cond_ms"] = float(clip_cond.group(1)) + float(clip_cond.group(2))
    if clip_uncond:
        metrics["clip_uncond_ms"] = float(clip_uncond.group(1)) + float(clip_uncond.group(2))
    return metrics


def run_phone_variant(*, adb_path: Path, phone_base: str, phone_script: str, termux_python: str, prompt: str, negative_prompt: str | None, seed: int, steps: int, width: int, height: int, cfg_scale: float, progressive_cfg: bool, tag: str, timeout: int) -> dict[str, Any]:
    cmd = [
        shlex.quote(termux_python),
        shlex.quote(phone_script),
        shlex.quote(prompt),
        "--seed", str(seed),
        "--steps", str(steps),
        "--cfg", str(cfg_scale),
        "--name", shlex.quote(tag),
    ]
    if negative_prompt and cfg_scale > 1.0:
        cmd.extend(["--neg", shlex.quote(negative_prompt)])
    if progressive_cfg and cfg_scale > 1.0:
        cmd.append("--prog-cfg")

    inner = (
        f"export SDXL_QNN_BASE={shlex.quote(phone_base)}; "
        f"export SDXL_QNN_OUTPUT_DIR={shlex.quote(phone_base + '/outputs')}; "
        f"export SDXL_QNN_WORK_DIR={shlex.quote(phone_base + '/phone_gen/work_speed_probe')}; "
        "export SDXL_QNN_USE_MMAP=1; "
        "export SDXL_QNN_LOG_LEVEL=warn; "
        + " ".join(cmd)
    )
    shell_command = f"su --mount-master -c {shlex.quote(inner)}"
    completed, elapsed = run_command([str(adb_path), "shell", shell_command], timeout=timeout)
    stdout = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise RuntimeError(f"phone benchmark failed ({completed.returncode}):\n{stdout[-4000:]}")
    return {
        "wall_ms": elapsed * 1000.0,
        "metrics": parse_phone_metrics(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-80:]),
        "command": shell_command,
        "size": [width, height],
    }


def build_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# SDXL speed probe — {report['timestamp']}")
    lines.append("")
    host_gpu = report.get("host_gpu", {})
    if host_gpu.get("gpus"):
        first = host_gpu["gpus"][0]
        lines.append(f"- Host GPU: **{first['name']}** ({int(first['memory_mib'])} MiB, driver {first['driver_version']})")
    phone = report.get("phone_info", {})
    if phone:
        lines.append(f"- Phone: **{phone.get('ro.product.manufacturer','')} {phone.get('ro.product.model','')}** / {phone.get('ro.board.platform','')}")
    lines.append(f"- Prompt: `{report['prompt']}`")
    lines.append(f"- Config: {report['width']}×{report['height']}, {report['steps']} steps, seed {report['seed']}")
    lines.append("- Phone runtime mode: **mmap ON**")
    lines.append("")
    lines.append("## Key numbers")
    lines.append("")
    lines.append("| Variant | Phone total | Phone UNet/step | PC total* | PC UNet/step | Phone/PC total | Phone/PC UNet step |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in [v.name for v in VARIANTS]:
        phone_entry = report.get("phone", {}).get(name)
        pc_entry = report.get("pc", {}).get(name)
        if not phone_entry and not pc_entry:
            continue
        phone_total = phone_entry["metrics"].get("total_ms") if phone_entry else None
        phone_unet = phone_entry["metrics"].get("unet_avg_ms") if phone_entry else None
        pc_total = pc_entry.get("total_ms_excluding_load") if pc_entry else None
        pc_unet = pc_entry.get("unet_avg_ms") if pc_entry else None
        total_ratio = (phone_total / pc_total) if (phone_total and pc_total) else None
        unet_ratio = (phone_unet / pc_unet) if (phone_unet and pc_unet) else None
        lines.append(
            "| {name} | {pt} | {pu} | {gt} | {gu} | {tr} | {ur} |".format(
                name=name,
                pt=f"{phone_total/1000:.2f}s" if phone_total else "—",
                pu=f"{phone_unet/1000:.3f}s" if phone_unet else "—",
                gt=f"{pc_total/1000:.2f}s" if pc_total else "—",
                gu=f"{pc_unet/1000:.3f}s" if pc_unet else "—",
                tr=f"{total_ratio:.2f}×" if total_ratio else "—",
                ur=f"{unet_ratio:.2f}×" if unet_ratio else "—",
            )
        )
    lines.append("")
    lines.append(r"\* PC total excludes initial model load; that load is reported separately in JSON.")
    lines.append("")

    phone_cfg1 = report.get("phone", {}).get("cfg1", {}).get("metrics", {})
    phone_cfg2 = report.get("phone", {}).get("cfg2", {}).get("metrics", {})
    phone_prog = report.get("phone", {}).get("cfg2_prog", {}).get("metrics", {})
    pc_cfg1 = report.get("pc", {}).get("cfg1", {})
    pc_cfg2 = report.get("pc", {}).get("cfg2", {})
    pc_prog = report.get("pc", {}).get("cfg2_prog", {})

    def _ratio(a: float | None, b: float | None) -> str:
        if not a or not b:
            return "—"
        return f"{a / b:.2f}×"

    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- Phone CFG penalty vs phone cfg1 (UNet/step): **{_ratio(phone_cfg2.get('unet_avg_ms'), phone_cfg1.get('unet_avg_ms'))}**")
    lines.append(f"- PC CFG penalty vs PC cfg1 (UNet/step): **{_ratio(pc_cfg2.get('unet_avg_ms'), pc_cfg1.get('unet_avg_ms'))}**")
    lines.append(f"- Phone progressive-CFG gain vs full CFG (UNet/step): **{_ratio(phone_cfg2.get('unet_avg_ms'), phone_prog.get('unet_avg_ms'))}**")
    lines.append(f"- PC progressive-CFG gain vs full CFG (UNet/step): **{_ratio(pc_cfg2.get('unet_avg_ms'), pc_prog.get('unet_avg_ms'))}**")
    lines.append("")
    lines.append("If the phone CFG penalty is much larger than the PC CFG penalty, the extra loss is probably not just raw NPU compute — it smells like runtime overhead (split passes, qnn-net-run launches, I/O, or context churn).")
    return "\n".join(lines)


def save_report(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sdxl_speed_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_summary(report), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local SDXL Lightning speed probe for RTX + phone NPU")
    ap.add_argument("--prompt", default="1girl, upper body, looking at viewer, masterpiece, best quality")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diffusers-dir", type=Path, default=DEFAULT_DIFFUSERS_DIR)
    ap.add_argument("--merged-unet-dir", type=Path, default=DEFAULT_MERGED_UNET_DIR)
    ap.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    ap.add_argument("--phone-base", default=DEFAULT_PHONE_BASE)
    ap.add_argument("--termux-python", default=DEFAULT_TERMUX_PYTHON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--variants", default="cfg1,cfg2,cfg2_prog",
                    help="Comma-separated list: cfg1,cfg2,cfg2_prog")
    ap.add_argument("--skip-pc", action="store_true")
    ap.add_argument("--skip-phone", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=1800)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    variants = resolve_variants(args.variants)
    resolved_phone_base = detect_phone_base(args.adb, args.phone_base) if not args.skip_phone else args.phone_base
    resolved_phone_script = detect_phone_script(args.adb) if not args.skip_phone else ""
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "phone_base": resolved_phone_base,
        "phone_script": resolved_phone_script,
        "host_gpu": nvidia_smi_info(),
        "phone_info": phone_info(args.adb) if not args.skip_phone else {},
        "pc": {},
        "phone": {},
        "notes": [
            "PC benchmark uses the same Lightning-merged SDXL weights as the repo speed path.",
            "PC total excludes initial model load; phone wall-clock includes actual runtime overhead via qnn-net-run.",
            "Phone variants are executed with SDXL_QNN_USE_MMAP=1.",
            "This is an empirical throughput probe, not a theoretical TOPS/TFLOPS calculator.",
        ],
    }

    pc_runtime: PcLightningRuntime | None = None
    if not args.skip_pc:
        pc_runtime = PcLightningRuntime(args.diffusers_dir, args.merged_unet_dir, args.device)
        for variant in variants:
            print(f"[pc] {variant.name} ...", flush=True)
            report["pc"][variant.name] = pc_runtime.benchmark(
                prompt=args.prompt,
                negative_prompt=args.negative,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
                cfg_scale=variant.cfg,
                progressive_cfg=variant.progressive_cfg,
            )

    if not args.skip_phone:
        for variant in variants:
            print(f"[phone] {variant.name} ...", flush=True)
            report["phone"][variant.name] = run_phone_variant(
                adb_path=args.adb,
                phone_base=resolved_phone_base,
                phone_script=resolved_phone_script,
                termux_python=args.termux_python,
                prompt=args.prompt,
                negative_prompt=args.negative,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
                cfg_scale=variant.cfg,
                progressive_cfg=variant.progressive_cfg,
                tag=f"speed_{variant.name}_s{args.seed}",
                timeout=args.timeout_sec,
            )

    json_path, md_path = save_report(args.out_dir, report)
    print(f"[ok] JSON: {json_path}")
    print(f"[ok] Markdown: {md_path}")
    print(build_summary(report))


if __name__ == "__main__":
    main()
