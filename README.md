# Model-to-NPU Pipeline for Snapdragon

**⚠️ CRITICAL PERFORMANCE REALITY (current SDXL runtime):** true UNet compute itself is currently about **11.5 s**.
Most of the larger observed wall-clock time is heavy **runtime-algorithm / driver overhead** (process lifecycle, context reload/init/deinit, orchestration, and surrounding I/O).
Reducing this overhead is the current top optimization priority.

> [!TIP]
> End-to-end SDXL flow is available and practically validated (`checkpoint -> final phone-generated PNG`).
> Some advanced build/conversion branches remain openly marked as beta or experimental.

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
**Current public beta result:** SDXL generation on a Snapdragon phone NPU with CLIP-L, CLIP-G, split UNet, VAE, Termux runtime, and an Android APK.

## Latest validation snapshot

- **2026-04-06:** practical SDXL full loop was re-validated from checkpoint build to final phone-generated PNG.
- Checkpoint used: `waiIllustriousSDXL_v160.safetensors`.
- Final validated output: `NPU/outputs/wai160_phone_native_cfg35_20260406.png`.
- Latest precise runtime timing (`seed=777`, `steps=8`, `CFG=3.5`, `--prog-cfg`, Live Preview OFF): `CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`, **`62.0 s total`**.
- Updated full walkthroughs are documented in:
  - [`README_EN.md`](README_EN.md)
  - [`README_RU.md`](README_RU.md)
  - [`SDXL/RUNBOOK_USED_FILES_AND_COMMANDS.md`](SDXL/RUNBOOK_USED_FILES_AND_COMMANDS.md)

## Quick links

- [English documentation](README_EN.md)
- [Русская документация](README_RU.md)
- [Android app notes](APK/README.md)
- [Live phone-side layout example](examples/phone-sdxl-qnn-layout.md)
- [SDXL script map (EN)](SDXL/SCRIPTS_OVERVIEW.md)
- [SDXL script map (RU)](SDXL/SCRIPTS_OVERVIEW_RU.md)
- [SDXL runbook used files+commands](SDXL/RUNBOOK_USED_FILES_AND_COMMANDS.md)
- [Current lessons learned](SDXL/LESSONS_LEARNED.md)
- [UNet quantization review (EN)](SDXL/UNET_QUANTIZATION_REVIEW.md)
- [UNet quantization review (RU)](SDXL/UNET_QUANTIZATION_REVIEW_RU.md)
- [UNet overhead review (EN)](SDXL/UNET_OVERHEAD_REVIEW.md)
- [UNet overhead review (RU)](SDXL/UNET_OVERHEAD_REVIEW_RU.md)

## Changelog

- **0.2.3** — APK/runtime/docs were refreshed around the new fast path: the split-UNet reuse pass now makes the early guided steps decay instead of hovering near a flat ~12 s plateau; the README-visible APK marker is **78.0 s total** (Live Preview ON), while the latest precise runtime-only run reached **62.0 s total** (`CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`) with Live Preview OFF on the same `v0.2.3` path; TAESD live preview now runs through rebuilt QNN GPU assets at roughly **1.0 s** per step instead of the older 5.5–6.0 s CPU path.
- **0.2.2** — APK/runtime snapshot refreshed for the current validation cycle: TAESD preview wiring was repaired for the QNN path, APK preview timing parsing now handles `QNN GPU` preview lines again, and the deploy/docs/sample notes were synchronized around the current phone runtime layout while early CFG-step tuning remains under active investigation.
- **0.2.1** — APK now routes transient runtime files (`WORK_DIR`, generated PNGs, and live preview frames) through app-private cache directories instead of shared storage, while keeping the deployed model tree in the public phone path.
- **0.2.0** — phone runtime and APK now show live **CPU / GPU / NPU** temperatures, default to QNN `sustained_high_performance`, auto-enable HTP backend extensions when `libQnnHtpNetRunExtensions.so` is deployed, and the current full `8`-step progressive-CFG best path reached about **79.7–80.6s total** on OnePlus 13.
- **0.1.3** — phone runtime and APK launch path now enable QNN `mmap` by default, repo-visible SDXL speed/overhead probes were added, and the current control run reached **104.4s total** at `1024×1024`, `8` steps, `CFG=1.0` on OnePlus 13.
- **0.1.2-beta** — APK now exposes a **½-CFG** toggle that applies guidance only to the first `ceil(steps / 2)` denoising steps; added a dedicated UNet quantization review with per-block risk notes and safer experiment boundaries.
- **0.1.2** — APK gained optional **Live Preview (TAESD)** using `phone_gen/taesd_decoder.onnx` + `onnxruntime` on the phone.
- **Runtime / preview fixes** — TAESD preview path was repaired end-to-end, including export, phone runtime wiring, and deployment docs.
- **APK log parsing fixes** — timing parsing and progress extraction were cleaned up so CLIP / UNet / VAE status reporting no longer drifts on mixed log lines.
- **Docs / sample sync** — rooted phone sample and the live phone layout docs were synchronized with the current shared-path runtime structure.

## What this repo already demonstrates

- SDXL can be pushed all the way to a **real Snapdragon phone NPU**;
- the runtime is not just a benchmark stub — it includes prompt processing, scheduler logic, PNG output, and an APK UI;
- the repository now openly separates:
  - the **public beta runtime path**,
  - the **build/export path**,
  - and the **experimental lab scripts** used to debug QAIRT/QNN edge cases.

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

Compared with the earlier public on-device screenshots and the latest precise `v0.2.3` runtime run:

- **273.6s → 78.0s total** (README-visible APK screenshot marker, Live Preview ON);
- **273.6s → 62.0s total** (latest precise runtime run, Live Preview OFF);
- **100.8s → 62.0s total**;
- **78.0s → 62.0s total** on the same `v0.2.3` generation path when running without live-preview overhead.

That corresponds to:

- **211.6s faster** vs `273.6s` (~**77.3%**);
- **38.8s faster** vs `100.8s` (~**38.5%**);
- **16.0s faster** vs the `78.0s` screenshot marker (~**20.5%**).

In short: **78.0s** remains a valid README-visible APK marker with Live Preview ON, while **62.0s** is the latest precise runtime timing for the same `v0.2.3` path with Live Preview OFF.

## Runtime updates vs APK version

Performance-sensitive changes in this project often land in `phone_generate.py` (deployed to the phone as `phone_gen/generate.py`), not only in the APK UI layer.

That means:

- the APK version may stay the same while runtime speed still improves;
- updating only `phone_generate.py` can make the same APK build run faster;
- APK rebuild/reinstall is not always required when only runtime generation logic changes.

## Why CFG is much slower here

For this project, **CFG above 1.0 is not a tiny tweak**.

- CFG means the model needs both **conditional** and **unconditional** predictions;
- this repository uses a **split UNet** (`encoder` + `decoder`) on the phone;
- so one logical UNet pass is already two QNN executions;
- naive CFG therefore turns each denoising step into **four** phone-side UNet subprocess calls:
  - uncond encoder,
  - uncond decoder,
  - cond encoder,
  - cond decoder.

The current phone runtime now batches cond+uncond work more efficiently, so it no longer pays the worst possible subprocess overhead every step. But the NPU still has to do roughly **double the real denoising work**, so generation time is still close to **2× slower** than the no-CFG path.

In short:

- **CFG = 1.0 or lower** → cheapest path;
- **CFG > 1.0** → sharper/more guided output, but much more UNet work;
- on this stack, that trade-off is very visible in wall-clock time.

## Current repository layout

- `SDXL/` — SDXL conversion, calibration, verification, QNN, and runtime experiments;
- `APK/` — Android app for on-device generation;
- `scripts/` — deploy and helper scripts;
- `tokenizer/` — shared tokenizer files;
- `phone_generate.py` — standalone phone-side generator used by the public beta runtime path.

If more model families are added later, each of them should get its own top-level folder alongside `SDXL/`.
