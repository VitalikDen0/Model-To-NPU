# Model-to-NPU Pipeline for Snapdragon

> [!WARNING]
> This repository is currently undergoing repeated end-to-end re-validation.
> The current layout and commands reflect the latest known working setup, but a clean full pass from export to final phone generation is still being re-checked.

Repository for **model-to-NPU pipelines** targeting Qualcomm Snapdragon devices.

**Current implemented pipeline:** `SDXL/`

## Choose your language

- [English documentation](README_EN.md)
- [Русская документация](README_RU.md)

## Quick links

- [`README_EN.md`](README_EN.md) — full English guide
- [`README_RU.md`](README_RU.md) — полное руководство на русском
- [`APK/README.md`](APK/README.md) — Android application notes
- [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md) — live phone-side SDXL deployment example gathered over ADB

## Current repository layout

- `SDXL/` — current SDXL-specific conversion / build scripts
- `APK/` — Android app for on-device generation
- `scripts/` — deploy and helper scripts
- `tokenizer/` — shared tokenizer files
- `phone_generate.py` — standalone phone-side generator

If more model families are added later, each of them should get its own top-level folder alongside `SDXL/`.
