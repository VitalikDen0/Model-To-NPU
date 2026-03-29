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

**Current tested model combination:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/795765/wai-illustrious-sdxl) + [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) (ByteDance)

> **Scope note:** the repo structure is broader than SDXL, but the actually validated pipeline in this repo is still SDXL-first.

## Current status

- **Repository direction:** multi-model Snapdragon NPU pipelines
- **Currently implemented family:** `SDXL/`
- **Current phone app target:** SDXL
- **Status of scripts:** being re-tested for full end-to-end reproducibility
- **Status of docs:** updated to the current known layout

For a live example of what a deployed phone-side SDXL directory currently looks like, see [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).

## Requirements for the current SDXL pipeline

### Phone

| Component | Requirement |
| --------- | ----------- |
| **SoC** | Qualcomm Snapdragon 8 Elite (SM8750) or QNN HTP compatible |
| **RAM** | 16 GB (peak ~12 GB, need >= 6 GB free) |
| **Storage** | ~10 GB for models and context binaries |
| **Root** | Magisk (for `su --mount-master` and QNN runtime access) |
| **Termux** | Python 3.13+, numpy, Pillow |

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

```bash
# Experimental helper for the early SDXL stages
python scripts/build_all.py --checkpoint path/to/model.safetensors
```

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

### 3. Deploy to phone

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/context_binaries \
  --qnn-lib-dir /path/to/qnn_sdk/lib/aarch64-android \
  --qnn-bin-dir /path/to/qnn_sdk/bin/aarch64-android
```

### 4. Termux setup (on phone)

```bash
pkg install python python-numpy python-pillow
```

### 5. Generate

#### Standalone (in Termux on phone)

```bash
su --mount-master
export PATH=/data/data/com.termux/files/usr/bin:$PATH
python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "1girl, anime, cherry blossoms"
python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "dark castle" --cfg 2.0 --neg "blurry"
python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "landscape" --seed 777 --steps 8
```

#### Via APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

The APK provides a full GUI: prompt, negative prompt, CFG, steps, seed, contrast stretching, progress bar, save to gallery.  
Use ⚙️ Settings to specify custom model paths.

#### Host-side (from PC via ADB)

```bash
python SDXL/generate.py "cat on windowsill, masterpiece" --seed 42
```

## What is actually on the phone right now?

The minimal required layout for generation is still the one used by the app and deploy scripts, but the live device snapshot currently contains a richer SDXL deployment with additional model libs, helper scripts, and QNN config JSON files.

- minimal required structure: documented below;
- live observed structure: [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).

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
│   └── phone-sdxl-qnn-layout.md ← Live phone-side deployment example
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
│   ├── build_android_model_lib_windows.py
│   ├── assess_generated_image.py
│   ├── verify_clip_vae_onnx.py
│   ├── verify_e2e_onnx.py
│   └── LESSONS_LEARNED.md    ← Pitfalls and solutions
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
/data/local/tmp/sdxl_qnn/
├── context/                               (QNN context binaries)
│   ├── clip_l.serialized.bin.bin          (~223 MB)
│   ├── clip_g.serialized.bin.bin          (~1.3 GB)
│   ├── unet_encoder_fp16.serialized.bin.bin (~2.3 GB)
│   ├── unet_decoder_fp16.serialized.bin.bin (~2.5 GB)
│   └── vae_decoder.serialized.bin.bin     (~151 MB)
├── phone_gen/
│   ├── generate.py                        (standalone generator)
│   └── tokenizer/
│       ├── vocab.json                     (CLIP BPE vocabulary)
│       └── merges.txt                     (BPE merge rules)
├── lib/                                   (QNN runtime libraries)
├── model/                                 (optional/extra model libs used in some flows)
├── bin/
│   └── qnn-net-run                        (QNN inference runner)
└── outputs/                               (generated PNGs)
```

## Limitations

- **Resolution is fixed** at 1024×1024 — others need full re-conversion
- **VAE FP16** slightly compresses color range -> percentile contrast stretching is applied
- **CFG doubles the time** — UNet runs twice (cond + uncond) per step
- **Root required** — for QNN runtime access
- **Termux required** — Python runtime for `phone_generate.py`
- Tested only on **OnePlus 13 (SM8750)**

## Known issues

- First run of each component is slower (loading a context binary into the NPU)
- Low RAM may cause process kill — close other apps
- `su --mount-master` is mandatory when calling from APK (mount namespace isolation)
- numpy and torch use different RNGs — the same seed produces different but valid images

## License

Apache 2.0 — see [LICENSE](LICENSE)

Dependencies:
- Qualcomm QAIRT SDK — proprietary Qualcomm license
- SDXL-Lightning LoRA (ByteDance) — Apache 2.0
- Stable Diffusion XL — CreativeML Open RAIL-M
