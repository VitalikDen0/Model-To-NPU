# Model-to-NPU Pipeline for Snapdragon

> Snapdragon NPU experiments that grew into a real phone-side SDXL pipeline.
> [!WARNING]
> **WAN end-to-end beta is NOT VERIFIED (`НЕ ПРОВЕРЕН`) and may not work at all.**
> Hot-swap `WxH` buckets and HotSwap LoRA are still test-stage features and can break.
> Stabilization/polish target after `v0.5.0`: about **2 weeks**.

**Docs:** [English](README_EN.md) · [Русский](README_RU.md) · [Android APK](APK/README.md)

This repository is about **running large diffusion models on Qualcomm Snapdragon devices**, not just exporting graphs and calling it a day.

Right now the most complete path is:

- **SDXL on Snapdragon 8 Elite NPU**;
- real phone-side generation with **CLIP + split UNet + VAE**;
- a custom **persistent QNN server** in C;
- a standalone phone runtime via `phone_generate.py`;
- an Android app in `APK/`.

## What is working today

- **SDXL end-to-end exists**: `checkpoint -> build/export -> deploy -> phone PNG`;
- **v0.5.1 APK includes Hot Preload, Hot-Swap LoRA UI, and Wan 2.1 UI fixes**;
- **`v0.5.0` APK split SDXL/WAN into separate tabs** with independent per-tab generation settings;
- **WAN remains beta**: end-to-end path and hot resolution switching are available for testing, but not validated for production;
- **AI Hub helpers already exist** for heavyweight compile flows, especially useful for WAN and large UNet pieces.

## Current direction

- **Publicly usable now:** SDXL
- **Active engineering focus:** WAN and FLUX
- **Training / method labs:** SD1.5 and SD3.5

SDXL is temporarily frozen as the main product branch while the repo shifts toward broader model-family support.

## Quick links

- [English documentation](README_EN.md)
- [Русская документация](README_RU.md)
- [Android app notes](APK/README.md)
- [WAN 2.1 workspace](WAN%202.1%201.3B/README.md)
- [Archive index (EN)](ARCHIVE_EN.md)
- [Архивный индекс (RU)](ARCHIVE_RU.md)
- [License](LICENSE)
- [Notice / attribution](NOTICE)

## Performance in one paragraph

The validated warm SDXL path on OnePlus 13 / Snapdragon 8 Elite is still in the **~30 s total** class at `1024x1024`, `8` steps, with cached CLIP, split UNet, and VAE on-device. In `v0.4.8-beta3`, the custom QNN server got a stronger HTP perf configuration and the major decoder regression dropped from roughly **`~820 ms`** per decoder pass to about **`~725–776 ms`**. There is still a residual tail of around **`~50 ms`** versus the historical ideal marker, and it is documented honestly instead of being swept under the rug.

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
    <td width="50%" align="center">
      <b>Earlier public screenshot — 273.6s total</b><br>
      <img src="https://github.com/user-attachments/assets/15c785f0-b7a3-4dac-8535-e14055bf3453" alt="Earlier phone-side proof screenshot at 273.6 seconds" width="100%">
    </td>
    <td width="50%" align="center">
      <b>v0.2.0 public marker — 100.8s total</b><br>
      <img src="https://github.com/user-attachments/assets/70988ed8-bf42-4235-8a70-19bf35db6574" alt="Phone-side proof screenshot for v0.2.0 at 100.8 seconds" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>v0.2.3 screenshot (Live Preview ON) — 78.0s total</b><br>
      <img src="https://github.com/user-attachments/assets/e36a584f-bb39-427a-805d-ea44e9a8b3a0" alt="Phone-side proof screenshot for v0.2.3 at 78.0 seconds" width="100%">
    </td>
    <td width="50%" align="center">
      <b>Current cold-start APK proof — 34.6s total</b><br>
      <img src="https://github.com/user-attachments/assets/04b6e61a-79d6-4ce5-a7d6-158461ca97e6" alt="Current phone-side proof screenshot at 34.6 seconds (cold start)" width="100%"><br>
      <sub>Measured accelerator-visible time inside this run: ~16.25 s.</sub>
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

Public screenshot lineage so far: **273.6 s → 100.8 s → 78.0 s → 34.6 s**, with the fourth slot now showing the current **34.6 s cold-start APK** proof image.

Inside that latest run, the accelerator-visible stages add up to **~16.25 s** total: `CLIP 0.134 s` + `UNet 14.248 s` + `VAE 1.872 s`. The screenshot-visible **34.6 s total** therefore still includes cold start / runtime bring-up / UI orchestration overhead rather than pure accelerator work.

The best validated historical warm-path marker remains **30.4 s total**. The new proof slot is intentionally described as a **cold-start APK** run, not as a replacement for that warm-path number.

Observed fast-path thermals in the current short-run proof cycle sat around **85–95°C** without visible throttling, so a few back-to-back generations remained practically safe/usable in the tested burst window.

## Important files, explained like a human

- `phone_generate.py` — the **main phone runtime entrypoint**. SDXL really runs through this file; WAN support here is currently a runtime/probe path, not final generation.
- `phone_runtime_accel.py` — optional **native math/layout helper** for scheduler and tensor operations, with a safe NumPy fallback if the native library is unavailable.
- `NPU/qnn_multi_context_server.c` — the **persistent QNN server** that keeps contexts alive and runs split UNet faster than repeated `qnn-net-run` spawns.
- `SDXL/run_end_to_end.ps1` — the most practical **host-side wrapper** for `checkpoint -> build -> deploy`.
- `scripts/build_all.py` — reproducible early build stages for SDXL: checkpoint conversion, Lightning merge, ONNX export.
- `scripts/deploy_to_phone.py` — pushes runtime files, QNN libs/bins, contexts, and optional TAESD pieces to the phone.
- `WAN 2.1 1.3B/export_and_compile_wan_aihub.py` — AI Hub helper for WAN package prep, compile jobs, status, and downloads.
- `WAN 2.1 1.3B/wan_tool.py` — WAN helper CLI for model selection, downloads, and phone checks.

## Repository map

- `SDXL/` — SDXL build/export/verification scripts and experiments
- `WAN 2.1 1.3B/` — WAN research workspace, AI Hub helpers, runtime probes
- `NPU/` — custom native runtime pieces, including the QNN multi-context server
- `APK/` — Android app
- `scripts/` — deployment and utility scripts
- `tokenizer/` — shared tokenizer assets

## Recent changes

- **0.5.1** — Added Hot Preload button for SDXL (keep-alive server integration), Hot-Swap LoRA UI (slot dropdown + `build_lora_slot.py` script), Wan 2.1 UI fixes (Frames/FPS sliders, default steps=30), and fixes for dynamic resolution WxH text field snapping bugs.
- **0.5.0** — APK split into SDXL/WAN tabs with separate saved settings per tab; WAN host flow got hot manifest bucket selection by requested `WxH`; added `WAN 2.1 1.3B/run_end_to_end.ps1` beta wrapper.
- **0.4.8-beta3** — stronger HTP perf mode in `qnn-multi-context-server`; major decoder regression mostly fixed; residual tail documented as known issue.
- **0.4.8-beta2** — APK runtime hotfix plus a dedicated **Copy error** action.
- **0.4.8-beta** — bundled Python runtime, dual root/no-root paths, TAESD preview intentionally disabled in APK because it hurt the fast path.
- **0.4.7** — exact CFG forwarding including `1.0`, better TAESD failure reporting.
- **0.4.6** — more deterministic packaged runtime refresh and safer public APK behavior.

## About the archive

The repo accumulated a lot of historical notes, one-off reviews, and working documents. They are still useful, but they no longer belong on the front page.

- Use [ARCHIVE_EN.md](ARCHIVE_EN.md) for English archive links.
- Use [ARCHIVE_RU.md](ARCHIVE_RU.md) for Russian archive links.

## License

This repo is distributed under **PolyForm Noncommercial License 1.0.0**.

That means, in plain language:

- non-commercial use/study/modification/forks are allowed;
- redistributions must keep the required notice text;
- third-party components keep their own licenses.
