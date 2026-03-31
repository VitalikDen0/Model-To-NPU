# Model-to-NPU Pipeline for Snapdragon

**Languages:** [English](README_EN.md) | [Русский](README_RU.md)

> [!WARNING]
> This repository is currently undergoing repeated end-to-end re-validation.
> The layout and commands below reflect the latest known working setup, but a clean full-pass check from PC export to final phone generation is still in progress.

<p align="center">
  <b>Repository for model-to-NPU pipelines on Qualcomm Snapdragon devices</b><br>
  Current implemented pipeline: <b>SDXL on Qualcomm Hexagon NPU</b>.
</p>

---

## What is this?

This repository is intended to grow into a home for multiple **model-specific pipelines** targeting Qualcomm Snapdragon NPUs.

- each model family gets its own folder;
- the **current implemented folder** is `SDXL/`;
- the **current Android app** lives in `APK/`;
- shared deployment helpers and assets live in `scripts/`, `tokenizer/`, and top-level helpers.

Right now the implemented and documented pipeline is **Stable Diffusion XL** running **natively on the phone NPU** (Hexagon HTP). The working SDXL path uses CLIP-L, CLIP-G, Split UNet (encoder + decoder), and VAE on the device.

**Current tested model combination:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310) + [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) (ByteDance)

> **Performance note:** the public beta timings, APK screenshots, and example outputs in this repository assume the **Lightning LoRA is already baked into the UNet**. That merge is not just a convenience step — it is the practical speed path used here. Running the plain base SDXL branch without the Lightning merge is dramatically slower on-device and should not be compared against the numbers shown below.
> **Resolution note:** the currently documented exports, context binaries, previews, and example images are built specifically for **1024×1024**.
> **Scope note:** the repo structure is broader than SDXL, but the actually validated pipeline in this repo is still SDXL-first.

## Current status

- **Repository direction:** multi-model Snapdragon NPU pipelines
- **Currently implemented family:** `SDXL/`
- **Current phone app target:** SDXL
- **Status of scripts:** being re-tested for full end-to-end reproducibility
- **Status of docs:** updated to the current known layout

For a live example of what a deployed phone-side SDXL directory currently looks like, see [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).
There is also a small rooted artifact bundle under [`examples/rooted-phone-sample/`](examples/rooted-phone-sample/) for reference and educational exploration.

For accumulated technical pitfalls and implementation notes, see [`SDXL/LESSONS_LEARNED.md`](SDXL/LESSONS_LEARNED.md) and the Russian counterpart [`SDXL/LESSONS_LEARNED_RU.md`](SDXL/LESSONS_LEARNED_RU.md).

For a dedicated review of the current SDXL UNet structure, split boundaries, and quantization risk zones, see [`SDXL/UNET_QUANTIZATION_REVIEW.md`](SDXL/UNET_QUANTIZATION_REVIEW.md) and [`SDXL/UNET_QUANTIZATION_REVIEW_RU.md`](SDXL/UNET_QUANTIZATION_REVIEW_RU.md).

For the latest runtime-overhead findings, `mmap` impact, and the post-`0.1.3` control-run numbers, see [`SDXL/UNET_OVERHEAD_REVIEW.md`](SDXL/UNET_OVERHEAD_REVIEW.md) and [`SDXL/UNET_OVERHEAD_REVIEW_RU.md`](SDXL/UNET_OVERHEAD_REVIEW_RU.md).

For a categorized map of every script currently living under `SDXL/`, see [`SDXL/SCRIPTS_OVERVIEW.md`](SDXL/SCRIPTS_OVERVIEW.md) and [`SDXL/SCRIPTS_OVERVIEW_RU.md`](SDXL/SCRIPTS_OVERVIEW_RU.md).

## Requirements for the current SDXL pipeline

### Phone

| Component | Requirement |
| --------- | ----------- |
| **SoC** | Qualcomm Snapdragon 8 Elite (SM8750) or QNN HTP compatible |
| **RAM** | 16 GB (peak ~12 GB, need >= 6 GB free) |
| **Storage** | ~10 GB for models and context binaries in a shared phone path such as `/sdcard/Download/sdxl_qnn` |
| **Root** | Not required for the current default layout |
| **Termux** | Python 3.13+, numpy, Pillow, `termux-setup-storage` |

### PC (for building the current pipeline)

| Component | Version |
| --------- | ------- |
| Python | 3.10.x (must be 3.10, not 3.11+) |
| QAIRT SDK | 2.31+ (Qualcomm AI Engine Direct) |
| Android NDK | r26+ (for `.so` build) |
| PyTorch | 2.x |
| Windows | 10/11 |

## Performance

Measured on OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM):

| Stage | Time | Format |
| ----- | ---- | ------ |
| CLIP-L | ~375 ms | FP16 |
| CLIP-G | ~1500 ms | FP16 |
| UNet encoder (1 step) | ~7.1 s | FP16 |
| UNet decoder (1 step) | ~7.1 s | FP16 |
| UNet (8 steps, no CFG) | ~113 s | FP16 |
| UNet (8 steps, CFG=3.5) | ~236 s | FP16 |
| VAE decoder | ~10 s | FP16 |
| **Total (no CFG)** | **~126 s** | |
| **Total (CFG=3.5)** | **~251 s** | |

Peak RAM: **~12 GB** out of 16 GB  
Resolution: **1024x1024** (fixed)

Recent session-validated `v0.1.3` control run with default `mmap` on OnePlus 13 (`1024×1024`, `8` steps, `CFG=1.0`) reached **104.4 s total** (`CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`), which is about **17.1% faster** than the earlier public no-CFG baseline.

Fresh `v0.2.0` tuned runs with **live thermal logging**, default `sustained_high_performance`, deployed HTP backend extensions, and **progressive CFG** (`8` steps, `CFG=3.5`, `--prog-cfg`) reached **79.7–80.6 s total** on the same OnePlus 13:

- run 1: `CLIP 2.858 s`, `UNet 73.031 s`, `VAE 3.547 s`, **80.6 s total**;
- run 2: `CLIP 2.917 s`, `UNet 72.391 s`, `VAE 3.395 s`, **79.7 s total**.

In those warmed-up full runs, the practical thermal envelope stayed around **CPU ~59–70°C**, **GPU ~50–52°C**, **NPU ~57–72°C**, with short NPU spikes observed up to about **78°C**. An early one-line CPU spike to `88.8°C` appeared before the first run stabilized and looks more like a transient sensor jump than the sustained generation state.

## Quick start

### 1. Environment setup (PC)

```bash
# Install Python 3.10 dependencies
pip install torch diffusers transformers safetensors onnx onnxruntime Pillow numpy

# Download QAIRT SDK
python scripts/download_qualcomm_sdk.py

# Download ADB (if not installed)
python scripts/download_adb.py
```

### 2. Build pipeline

> **Note:** `scripts/build_all.py` currently automates only the earlier repeatable stages and intentionally does **not** blindly run every later QNN/deploy step while those scripts are being re-validated.
> **Script-scope note:** not every file under `SDXL/` is required for the shortest public happy-path. Some of them are verification, calibration, profiling, or QAIRT/QNN workaround helpers.

```bash
# Experimental helper for the early SDXL stages
python scripts/build_all.py --checkpoint path/to/model.safetensors
```

There is also a careful beta wrapper for the currently documented flow:

```powershell
pwsh SDXL/run_end_to_end.ps1 -Checkpoint path/to/model.safetensors -ContextsDir path/to/context_binaries
```

That wrapper intentionally separates the reproducible early build stages from the still-beta runtime/deploy pieces.

Or step by step:

```bash
# 1. Download WAI Illustrious SDXL v1.60 and SDXL-Lightning LoRA

# 2. Convert checkpoint to diffusers format
python SDXL/convert_sdxl_checkpoint_to_diffusers.py

# 3. Merge Lightning LoRA into UNet
python SDXL/bake_lora_into_unet.py

# 4. Export all components to ONNX
python SDXL/export_clip_vae_to_onnx.py
python SDXL/export_sdxl_to_onnx.py

# 5. Convert to QNN
python SDXL/convert_clip_vae_to_qnn.py
python SDXL/convert_lightning_to_qnn.py

# 6. Build Android model libraries (.so)
python SDXL/build_android_model_lib_windows.py
```

Optional live-preview path for the APK / phone runtime:

```bash
# Export the tiny TAESD XL preview decoder
python SDXL/export_taesd_to_onnx.py --validate

# Deploy the single-file ONNX to the phone runtime
adb push D:/platform-tools/sdxl_npu/taesd_decoder/taesd_decoder.onnx /sdcard/Download/sdxl_qnn/phone_gen/

# Optional (only if you want live preview in Termux / APK)
python -m pip install onnxruntime
```

### 3. Deploy to phone

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/context_binaries \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn_sdk/lib/aarch64-android \
  --qnn-bin-dir /path/to/qnn_sdk/bin/aarch64-android
```

When `--qnn-lib-dir` contains `libQnnHtpNetRunExtensions.so`, the deploy script now copies it as well, so the phone runtime and APK can auto-enable the shipped `htp_backend_extensions_lightning.json` path.

### 4. Termux setup (on phone)

```bash
pkg install python python-numpy python-pillow
python -m pip install onnxruntime   # optional, only for TAESD live preview
termux-setup-storage
```

### 5. Generate

#### Standalone (in Termux on phone)

```bash
export PATH=/data/data/com.termux/files/usr/bin:$PATH
export SDXL_QNN_BASE=/sdcard/Download/sdxl_qnn
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "1girl, anime, cherry blossoms"
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "dark castle" --cfg 2.0 --neg "blurry"
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "landscape" --seed 777 --steps 8
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "1girl, upper body, looking at viewer, masterpiece, best quality" --seed 777 --steps 8 --cfg 3.5 --prog-cfg
```

The current runtime defaults to:

- `SDXL_QNN_USE_MMAP=1`
- `SDXL_QNN_PERF_PROFILE=sustained_high_performance`
- live CPU / GPU / NPU logging when `SDXL_SHOW_TEMP=1`

If both `htp_backend_extensions_lightning.json` and `lib/libQnnHtpNetRunExtensions.so` exist in the deployed phone directory, `phone_generate.py` now auto-enables `SDXL_QNN_CONFIG_FILE` even for direct Termux runs.

#### Via APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

The APK provides a full GUI: prompt, negative prompt, CFG, steps, seed, contrast stretching, progress bar, live CPU / GPU / NPU temperatures, and save to gallery.  
APK `v0.2.1` includes the optional **Live Preview (TAESD)** toggle, the **½-CFG** toggle that keeps CFG only on the first `ceil(steps / 2)` denoising steps when guidance is enabled, enables QNN `mmap` + `sustained_high_performance` by default, auto-exports the backend-extension config when the required `.json` + `.so` are present in the deployed path, and now writes transient runtime files through app-private cache directories instead of shared storage.  
The current default shared path is `/sdcard/Download/sdxl_qnn`; use ⚙️ Settings if you want a different layout.

#### Host-side (from PC via ADB)

```bash
python SDXL/generate.py "cat on windowsill, masterpiece" --seed 42
```

## What is actually on the phone right now?

The current default deploy target is `/sdcard/Download/sdxl_qnn`, but the live device snapshot linked below is a historical rooted layout which still helps document the produced files.

- minimal required structure: documented below;
- live observed historical structure: [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).

## Project structure

```text
├── README.md                 ← language landing page
├── README_RU.md              ← Russian documentation
├── README_EN.md              ← you are here
├── LICENSE                   ← Apache 2.0
├── .gitattributes
├── .gitignore
├── phone_generate.py         ← Standalone generator (runs on phone)
├── tokenizer/                ← BPE tokenizer files (CLIP)
│   ├── vocab.json
│   └── merges.txt
├── examples/
│   ├── phone-sdxl-qnn-layout.md    ← Live rooted phone-side layout example
│   ├── phone-sdxl-qnn-layout_RU.md ← Russian translation of the layout example
│   └── rooted-phone-sample/        ← Small rooted artifact bundle (docs, PNGs, configs, scripts)
├── .github/
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
├── scripts/
│   ├── deploy_to_phone.py    ← Deploy to phone via ADB
│   ├── download_qualcomm_sdk.py
│   ├── download_adb.py
│   └── build_all.py          ← Early-stage SDXL helper (later stages under re-test)
├── SDXL/                     ← Current SDXL-specific conversion & build scripts
│   ├── generate.py           ← Host-side generator (from PC)
│   ├── bake_lora_into_unet.py
│   ├── export_clip_vae_to_onnx.py
│   ├── export_sdxl_to_onnx.py
│   ├── convert_clip_vae_to_qnn.py
│   ├── convert_lightning_to_qnn.py
│   ├── export_taesd_to_onnx.py
│   ├── convert_taesd_to_qnn.py
│   ├── build_android_model_lib_windows.py
│   ├── assess_generated_image.py
│   ├── verify_clip_vae_onnx.py
│   ├── verify_e2e_onnx.py
│   ├── LESSONS_LEARNED.md    ← Pitfalls and solutions
│   └── LESSONS_LEARNED_RU.md ← Russian lessons-learned counterpart
└── APK/                      ← Android application
    ├── README.md
    └── app/src/main/
        ├── AndroidManifest.xml
        └── java/com/sdxlnpu/app/
            ├── MainActivity.java
            └── SettingsActivity.java
```

## Architecture

```text
          ┌──────────────────────────────────────────────────────────────┐
          │                      Phone (NPU)                            │
          │                                                             │
Prompt ──▶│ CLIP-L ──┐                                                  │
          │ (FP16)   ├──▶ concat [1,77,2048] ──▶ Split UNet ──▶ VAE ──▶│──▶ PNG
          │ CLIP-G ──┘    + pooled [1,1280]      encoder FP16   FP16   │
          │ (FP16)        + time_ids [1,6]       decoder FP16          │
          │                                      × 8 steps             │
          └──────────────────────────────────────────────────────────────┘
```

**Split UNet:** The full FP16 UNet (~5 GB) exceeds the HTP allocation limit (~3.5 GB), so it is split into encoder (conv_in + down_blocks + mid_block, 2.52 GB) and decoder (up_blocks + conv_out, 2.69 GB). The encoder passes 11 skip-connections + mid + temb to the decoder.

**Scheduler:** EulerDiscrete, trailing spacing (Lightning requirement), pure numpy.

**Tokenizer:** Pure Python BPE (no HuggingFace/transformers), identical to the CLIP tokenizer.

## Minimal phone file structure

```text
/sdcard/Download/sdxl_qnn/
├── context/                               (QNN context binaries)
│   ├── clip_l.serialized.bin.bin          (~223 MB)
│   ├── clip_g.serialized.bin.bin          (~1.3 GB)
│   ├── unet_encoder_fp16.serialized.bin.bin (~2.3 GB)
│   ├── unet_decoder_fp16.serialized.bin.bin (~2.5 GB)
│   └── vae_decoder.serialized.bin.bin     (~151 MB)
├── htp_backend_extensions_lightning.json  (optional HTP backend extensions entrypoint)
├── htp_backend_ext_config_lightning.json  (optional HTP backend tuning config)
├── phone_gen/
│   ├── generate.py                        (standalone generator)
│   ├── taesd_decoder.onnx                 (~5 MB, optional live preview)
│   └── tokenizer/
│       ├── vocab.json                     (CLIP BPE vocabulary)
│       └── merges.txt                     (BPE merge rules)
├── lib/                                   (QNN runtime libraries)
│   └── libQnnHtpNetRunExtensions.so       (optional, auto-used when present)
├── model/                                 (optional/extra model libs used in some flows)
├── bin/
│   └── qnn-net-run                        (QNN inference runner)
└── outputs/                               (generated PNGs)
```

## Limitations

- **Resolution is fixed** at 1024×1024 — others need full re-conversion
- **The documented speed path assumes the Lightning LoRA has been baked into the UNet** — skipping that merge means a much slower baseline SDXL path and the repository timings/examples stop being representative
- **VAE FP16** slightly compresses color range -> percentile contrast stretching is applied
- **TAESD live preview is optional** — it is only used for intermediate previews and now relies on a tiny ONNX decoder (`phone_gen/taesd_decoder.onnx`) plus `onnxruntime`, not on a QNN preview context
- **The best current fast path uses HTP backend extensions** — the repo ships the JSON config, but the runtime only auto-enables it when `libQnnHtpNetRunExtensions.so` is actually deployed under `lib/`
- **CFG > 1.0 is expensive here** — conditional + unconditional predictions are both needed; because the runtime uses a split UNet (`encoder` + `decoder`), naive CFG means four phone-side UNet subprocess calls per step. The current runtime batches part of that work better than before, but wall-clock time is still close to 2× versus the no-CFG path.
- **Termux required** — Python runtime for `phone_generate.py`
- **Android shared-storage access may need manual confirmation** — especially for APK use on Android 11+
- Tested only on **OnePlus 13 (SM8750)**

## Known issues

- First run of each component is slower (loading a context binary into the NPU)
- Low RAM may cause process kill — close other apps
- On Android 11+, the APK may need "all files access" to read `/sdcard/Download/sdxl_qnn`
- If `python3` is not reachable from the app process, adjust the Python command/path in ⚙️ Settings
- If backend extensions are missing, the runtime still works — it simply falls back to the non-config QNN path
- numpy and torch use different RNGs — the same seed produces different but valid images

## License

Apache 2.0 — see [LICENSE](LICENSE)

Dependencies:

- Qualcomm QAIRT SDK — proprietary Qualcomm license
- SDXL-Lightning LoRA (ByteDance) — Apache 2.0
- Stable Diffusion XL — CreativeML Open RAIL-M
