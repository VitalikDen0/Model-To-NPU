# Model-to-NPU Pipeline for Snapdragon

> **SDXL on Snapdragon 8 Elite NPU — ~30 s total** (UNet ~19 s, VAE ~1.9 s, CLIP ~9 ms cached)
> at 1024×1024, 8 steps, CFG=3.5, progressive guidance.

> [!TIP]
> End-to-end SDXL flow is available and practically validated (`checkpoint -> final phone-generated PNG`).
> Work on **SD3**, **Flux**, **Wan** and other model families has started — they will be released as the methods are developed and validated.

<p align="center">
  <a href="README_EN.md"><img src="https://img.shields.io/badge/docs-English-0A66C2?style=for-the-badge" alt="English docs"></a>
  <a href="README_RU.md"><img src="https://img.shields.io/badge/docs-Русский-1F883D?style=for-the-badge" alt="Russian docs"></a>
  <a href="APK/README.md"><img src="https://img.shields.io/badge/Android-APK-3DDC84?style=for-the-badge&logo=android&logoColor=white" alt="Android APK"></a>
</p>

<p align="center">
  <a href="README_EN.md#requirements-for-the-current-sdxl-pipeline"><img src="https://img.shields.io/badge/Phone%20Python-3.13%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Phone Python 3.13+"></a>
  <a href="README_EN.md#requirements-for-the-current-sdxl-pipeline"><img src="https://img.shields.io/badge/Host%20Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Host Python 3.10"></a>
  <a href="README_EN.md#performance"><img src="https://img.shields.io/badge/Snapdragon-8%20Elite-5C2D91?style=for-the-badge" alt="Snapdragon 8 Elite"></a>
  <a href="README_EN.md#architecture"><img src="https://img.shields.io/badge/QNN%20%2F%20QAIRT-Hexagon%20NPU-CB2E6D?style=for-the-badge" alt="QNN / QAIRT"></a>
</p>

Repository for **model-to-NPU pipelines** targeting Qualcomm Snapdragon devices.

**Current implemented family:** `SDXL/`
**Exploratory Wan workspace:** `WAN 2.1 1.3B/` (research / selection / 480p-first planning)
**Future models:** SD3, Flux, Wan, and others — as the optimization methods are developed

**Current public beta result:** SDXL generation on a Snapdragon phone NPU with a persistent multi-context QNN server, CLIP-L + CLIP-G (cached), split UNet (encoder→decoder via in-memory RUN_CHAIN), VAE, Termux runtime, and an Android APK.

## Performance snapshot (v0.4.1)

Measured on OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM), `seed=44`, `steps=8`, `CFG=3.5`, `--prog-cfg`, Live Preview OFF:

| Stage | Time | Notes |
| ----- | ---- | ----- |
| CLIP-L + CLIP-G | ~9 ms | cached result (first run ~2.8 s) |
| UNet (8 steps) | ~19.3 s | ~2411 ms/step via persistent server |
| VAE decoder | ~1.9 s | FP16 |
| **Total (warm)** | **~30.4 s** | |

**Speedup history:** 273.6 s → 104.4 s → 75.6 s → **30.4 s** (current)

## Quick links

- [English documentation](README_EN.md)
- [Русская документация](README_RU.md)
- [License and usage terms](LICENSE)
- [Mandatory attribution notice](NOTICE)
- [Android app notes](APK/README.md)
- [WAN 2.1 1.3B exploration workspace](WAN%202.1%201.3B/README.md)
- [Historical performance archive (EN)](HISTORY_EN.md)
- [Historical performance archive (RU)](HISTORY_RU.md)
- [Live phone-side layout example](examples/phone-sdxl-qnn-layout.md)
- [SDXL script map (EN)](SDXL/SCRIPTS_OVERVIEW.md)
- [SDXL script map (RU)](SDXL/SCRIPTS_OVERVIEW_RU.md)
- [SDXL runbook used files+commands](SDXL/RUNBOOK_USED_FILES_AND_COMMANDS.md)
- [Current lessons learned](SDXL/LESSONS_LEARNED.md)
- [UNet quantization review (EN)](SDXL/UNET_QUANTIZATION_REVIEW.md)
- [UNet quantization review (RU)](SDXL/UNET_QUANTIZATION_REVIEW_RU.md)
- [UNet overhead review (EN)](SDXL/UNET_OVERHEAD_REVIEW.md)
- [UNet overhead review (RU)](SDXL/UNET_OVERHEAD_REVIEW_RU.md)

## License model

This repository is now distributed under the
**PolyForm Noncommercial License 1.0.0**.

This is **source-available**, not OSI open source, because commercial use is
prohibited.

In short:

- use, study, modify, and fork are allowed for **non-commercial** purposes;
- redistributions must include the PolyForm terms (or their canonical URL) and
  the `Required Notice:` lines from [`NOTICE`](NOTICE);
- third-party components keep their own licenses and must be respected
  separately.

## Changelog

- **0.4.3** — **shared prewarm reuse + self-contained runtime hotfix**: `qnn-multi-context-server` now supports shared FIFO IPC plus deterministic context IDs, so the app-open prewarm can be reused by later foreground generate runs instead of warming a private throwaway child process. APK prewarm and foreground generation now share the same app-cache work directory and opt into `SDXL_QNN_SHARED_SERVER=1`, which is the practical path toward one logical multi-resolution QNN runtime built from multiple fixed-shape contexts. Device-side validation also hardened bundled runtime delivery: APK now forces payload refresh when the packaged runtime version changes, and shared-server startup waits for FIFO IPC readiness after `READY`, fixing the race where the first `LOAD` could fail before request/response FIFOs became visible. The refreshed public `v0.4.3` asset also releases shared prewarm after 30 seconds of foreground/background inactivity (including after generation completes), stages TAESD preview assets more aggressively inside the bundled payload so preview stops depending on stale shared-storage leftovers, prefers the APK-bundled QNN runtime over stale `/data/local/tmp` leftovers when explicit bundled paths are exported, safely restages backend-extension configs with relative paths, packages the missing core QNN runtime pieces (`qnn-net-run` plus the required HTP/System libs), tightens preview/final bitmap cleanup, and keeps `832x480` as a WAN-only size instead of a generic SDXL preset.
- **0.4.2** — **APK prewarm & UX**: models start loading at app open (QNN server + all contexts), auto-killed 30 s after minimize, restarted on return; all generation settings (prompt, seed, steps, CFG, resolution, checkboxes) saved and restored between sessions; app renamed to **NPU Gen**; resolution picker moved below ½ CFG; TAESD preview stride fix deployed (stride-based preview for Lightning speed).
- **0.4.1** — **APK runtime payload bugfix**: the Android app now bundles and uses the current `phone_generate.py` plus optional `qnn-multi-context-server`, `qnn-context-runner`, and `libsdxl_runtime_accel.so` from APK assets, so stale `/sdcard` scripts no longer break new `--width` / `--height` arguments with `argparse` exit code `2`. Bundled offline runtime is auto-extracted before Python probing, Settings → Verify now shows bundled runtime payload status, and `scripts/deploy_to_phone.py` now pushes `qnn-multi-context-server` and recursively deploys resolution-scoped context directories.
- **0.4.0** — **variable resolution support** (512×512 to 1536×1536 and arbitrary multiples of 8): phone_generate.py, export, and APK all accept `--width`/`--height`; per-resolution QNN context directories (`context/{W}x{H}/`); APK resolution picker UI; `build_termux_prefix.py` for self-contained Termux prefix extraction; RuntimeBootstrap sets executable permissions on bundled binaries.

For the complete historical changelog including all intermediate versions and pre-reset experiments, see [HISTORY_EN.md](HISTORY_EN.md).

## What this repo already demonstrates

- SDXL can be pushed all the way to a **real Snapdragon phone NPU** at ~30 s/image;
- the runtime is not just a benchmark stub — it includes prompt processing, scheduler logic, PNG output, and an APK UI;
- a custom **persistent QNN server** written in C eliminates process spawn overhead and enables in-memory encoder→decoder piping;
- the repository openly separates the public beta runtime path, the build/export path, and the experimental lab scripts.

## Gallery

<!-- markdownlint-disable MD033 -->
<table align="center">
  <tr>
    <td width="50%"><img src="https://github.com/user-attachments/assets/915ef71e-d72b-4fa0-823d-b316289f2041" alt="SDXL on phone sample 1" width="100%"></td>
    <td width="50%"><img src="https://github.com/user-attachments/assets/4bc1ac51-a98e-4931-a3e9-247327e0bbe5" alt="SDXL on phone sample 2" width="100%"></td>
  </tr>
  <tr>
    <td width="50%"><img src="https://github.com/user-attachments/assets/1c87282c-ccc2-4dc1-b003-0693dd0fa3d4" alt="SDXL on phone sample 3" width="100%"></td>
    <td width="50%"><img src="https://github.com/user-attachments/assets/8f5e3d0d-ebe6-4cea-98f7-2b13b51a9ede" alt="SDXL on phone sample 4" width="100%"></td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

All gallery samples and the currently documented phone-side examples are **1024×1024** outputs from the current Lightning-merged SDXL path.

## Proof that it actually runs on-device

<!-- markdownlint-disable MD033 -->
<table align="center">
  <tr>
    <td width="33%" align="center">
      <b>Earlier public screenshot — 273.6s total</b><br>
      <img src="https://github.com/user-attachments/assets/15c785f0-b7a3-4dac-8535-e14055bf3453" alt="Earlier phone-side proof screenshot at 273.6 seconds" width="100%">
    </td>
    <td width="33%" align="center">
      <b>v0.2.0 public marker — 100.8s total</b><br>
      <img src="https://github.com/user-attachments/assets/70988ed8-bf42-4235-8a70-19bf35db6574" alt="Phone-side proof screenshot for v0.2.0 at 100.8 seconds" width="100%">
    </td>
    <td width="33%" align="center">
      <b>v0.2.3 screenshot (Live Preview ON) — 78.0s total</b><br>
      <img src="https://github.com/user-attachments/assets/e36a584f-bb39-427a-805d-ea44e9a8b3a0" alt="Phone-side proof screenshot for v0.2.3 at 78.0 seconds" width="100%">
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

Speedup since the first public screenshot: **273.6 s → 30.4 s** (~**9× faster**, ~**89% reduction**).

## Current repository layout

- `SDXL/` — SDXL conversion, calibration, verification, QNN, and runtime experiments;
- `NPU/` — persistent multi-context QNN server (C source + build scripts);
- `WAN 2.1 1.3B/` — early Wan 2.1 T2V 1.3B research, candidate selection, download helpers, and phone probing;
- `APK/` — Android app for on-device generation;
- `scripts/` — deploy and helper scripts (including `build_qnn_multi_context_server.py`);
- `tokenizer/` — shared tokenizer files;
- `phone_generate.py` — standalone phone-side generator used by the public beta runtime path.

If more model families are added later, each of them should get its own top-level folder alongside `SDXL/`.
