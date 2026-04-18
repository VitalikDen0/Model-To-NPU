# Model-to-NPU Android Application

**Languages:** [Русский](README.md) | [English](README_EN.md)

**Current target:** SDXL

> [!WARNING]
> The Android app and the surrounding scripts are still being re-tested together with the rest of the repository. The app is already usable, but the documentation remains intentionally conservative until the next full validation pass is complete.

## Status: Working / In active validation

This APK is used to generate images directly on the phone through the Qualcomm NPU.  
The currently implemented target is **SDXL Lightning**.  
After the model files are deployed, the workflow is intended to be **fully standalone** — no PC is needed for normal generation.

Current documented APK version: **`0.2.5`**.

Fresh local review on OnePlus 13 for the standard path (`seed=777`, `8` steps, `CFG=3.5`, `--prog-cfg`, Live Preview OFF) reached **75.6 s total** with `burst` + native runtime accel; the rerun with `basic` profiling stayed effectively the same at **75.2 s total**.

Historical note: the older best-known **62.0 s** runtime result belonged to the pre-reset phone state. The run itself was real, but after the later factory reset the exact phone-side context/runtime state, screenshots, and supporting technical artifacts were not preserved, so the repository can no longer honestly reproduce or independently prove that exact chain as a current result.

Important: in this stack, performance is primarily driven by `phone_generate.py` (deployed on phone as `phone_gen/generate.py`), so the same APK version can get faster after updating only that runtime script.

## Architecture

```text
┌─────────────────────────────────┐
│         MainActivity            │
│  ┌───────────────────────────┐  │
│  │ Prompt + Negative Prompt  │  │
│  │ Seed / Steps / CFG        │  │
│  ├───────────────────────────┤  │
│  │     Image Preview         │  │
│  ├───────────────────────────┤  │
│  │  Generate / Stop / Save   │  │
│  └───────────────────────────┘  │
│         ⚙️ SettingsActivity     │
│  ┌───────────────────────────┐  │
│  │ Model path               │  │
│  │ Python path / command    │  │
│  │ Verify / Reset           │  │
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│   Shared Downloads path         │
│   → configurable Python3        │
│   → phone_generate.py           │
│   (BPE tokenizer + scheduler)   │
├─────────────────────────────────┤
│       QNN Runtime (HTP/NPU)     │
│  ┌───────┬───────┬───────────┐  │
│  │CLIP-L │CLIP-G │Split UNet │  │
│  │ FP16  │ FP16  │  FP16     │  │
│  ├───────┴───────┤enc + dec  │  │
│  │    VAE FP16   │           │  │
│  └───────────────┴───────────┘  │
└─────────────────────────────────┘
```

## Requirements

- **Phone**: Snapdragon 8 Elite (SM8750) or another device with a compatible Qualcomm NPU
- **Root**: not required for the current default path layout
- **Termux**: Python 3.13+, numpy, Pillow, `termux-setup-storage`
- **TAESD live preview (optional)**: preferred path is deployed QNN TAESD assets on the GPU; fallback is `onnxruntime` in Termux + `phone_gen/taesd_decoder.onnx`
- **Android 11+**: shared Downloads access may require “all files access”
- **QNN**: context binaries must already be deployed (see `scripts/deploy_to_phone.py`)

## Installation

### 1. Prepare Termux

```bash
pkg install python python-numpy python-pillow
python -m pip install onnxruntime   # optional, only for Live Preview
termux-setup-storage
```

### 2. Deploy model files from PC

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/contexts \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn/lib \
  --qnn-bin-dir /path/to/qnn/bin
```

### 3. Build and install the APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Features

- prompt and negative prompt input;
- steps control (1–20);
- CFG control (1.0–7.0);
- half-CFG toggle — guidance only on the first `ceil(steps / 2)` denoising steps;
- reproducible seed handling;
- percentile-based contrast stretching;
- Live Preview (TAESD) toggle with a preferred QNN GPU path and CPU ONNX fallback;
- save-to-gallery support;
- stop button for ongoing generation;
- progress bar for CLIP → UNet → VAE stages;
- live CPU / GPU / NPU temperature line during generation;
- configurable paths through ⚙️ Settings.

## Model settings

Through ⚙️ in the ActionBar you can:

- change the model directory path (default: `/sdcard/Download/sdxl_qnn`);
- change the Python command/path (default: `python3`);
- verify the required files on the phone;
- reset settings back to detected defaults.

## Performance

| Stage | Time |
| ----- | ---- |
| CLIP-L + CLIP-G | ~2.3 s |
| UNet encoder + decoder × 8 steps (no CFG) | ~113 s |
| UNet encoder + decoder × 8 steps (CFG=3.5) | ~236 s |
| VAE | ~10 s |
| **Total (no CFG)** | **~126 s** |
| **Total (CFG=3.5)** | **~251 s** |

The session-validated `v0.1.3` control run (`1024×1024`, `8` steps, `CFG=1.0`, `mmap` ON) reached **104.4 s total**: `CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`.

The current best session-validated `v0.2.0` path with `sustained_high_performance`, backend extensions, and `--prog-cfg` (`8` steps, `CFG=3.5`) reached **79.7–80.6 s total** on OnePlus 13. During those full runs, live thermals typically stayed around **CPU ~59–70°C**, **GPU ~50–52°C**, **NPU ~57–72°C**, with short NPU spikes seen up to roughly **78°C**.

For the current APK/runtime state, it is more accurate to track three practical markers:

- APK screenshot marker (Live Preview ON): about **78.0 s total**;
- current rebuilt-phone local review (Live Preview OFF, `burst` + native runtime accel): **75.6 s total** with `seed=777`, `steps=8`, `CFG=3.5`, `--prog-cfg`, and stage times `CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`;
- historical pre-reset fast-path note: **62.0 s total** on the old `v0.2.3` path with stage times `CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`.

UNet step progression in the historical `62.0 s` run:

- CFG steps 1..4: **9.765 → 8.230 → 8.386 → 7.936 s**;
- no-guidance steps 5..8: **5.377 → 5.513 → 5.294 → 5.479 s**.

TAESD live preview on rebuilt **QNN GPU** assets is still around **1.0 s** per step (vs older **5.5–6.0 s** CPU preview), and this preview/UI overhead is the main reason APK screenshot-visible totals are higher than no-preview runtime totals.

Recent overhead re-checks also showed that moving the runtime tree back to `/data/local/tmp/sdxl_qnn`, re-running the older pre-`v0.2.4-beta` Python runtime, and trying the custom daemon path again did **not** recover the old `62 s` chain. In the rebuilt environment the daemon path currently hangs, and the original split-context/model artifact state that once produced the faster run was not preserved before the reset.

## Files on the phone

```text
/sdcard/Download/sdxl_qnn/      (or your custom Settings path)
├── context/
│   ├── clip_l.serialized.bin.bin
│   ├── clip_g.serialized.bin.bin
│   ├── unet_encoder_fp16.serialized.bin.bin
│   ├── unet_decoder_fp16.serialized.bin.bin
│   └── vae_decoder.serialized.bin.bin
├── phone_gen/
│   ├── generate.py
│   ├── taesd_decoder.onnx
│   └── tokenizer/
│       ├── vocab.json
│       └── merges.txt
├── htp_backend_extensions_lightning.json
├── htp_backend_ext_config_lightning.json
├── lib/    (QNN runtime libs)
│   └── libQnnHtpNetRunExtensions.so   (optional, enables backend extensions)
├── model/  (optional extra model libs in some flows)
├── bin/    (qnn-net-run)
└── outputs/
```

## Technical notes

- APK version and runtime speed do not always move in lockstep: speed-ups can come from updated `phone_generate.py` even when the APK version number is unchanged.
- the APK launches `phone_generate.py` without `su`, through a normal shell and a configurable Python command;
- the default layout uses `/sdcard/Download/sdxl_qnn`;
- APK `v0.2.5` explicitly exports `SDXL_QNN_USE_MMAP=1`, `SDXL_QNN_PERF_PROFILE=burst`, hard-disables the current daemon fast path (`SDXL_QNN_USE_DAEMON=0`), enables async/prestage/prewarm flags for the current Python runtime, enables live thermal logging, auto-adds `SDXL_QNN_CONFIG_FILE` when `htp_backend_extensions_lightning.json` and `lib/libQnnHtpNetRunExtensions.so` are present, routes transient `WORK_DIR` / preview / output files into the app cache to reduce shared-storage overhead, disables extra final PNG compression to shave a bit of save overhead, and correctly parses preview timing lines in the `QNN GPU ...ms` format;
- Live Preview now prefers rebuilt QNN TAESD preview assets (`taesd_decoder.serialized.bin.bin` and/or `model/libTAESDDecoder.so`) on the GPU backend, runs there at roughly **1.0 s** per step, and falls back to `phone_gen/taesd_decoder.onnx` on CPU through `onnxruntime` only when QNN preview is unavailable or fails;
- CFG above `1.0` is noticeably slower because the phone runtime still needs both cond and uncond denoising branches; with a split UNet that translates into substantially more encoder/decoder work per step even after batching optimizations;
- the **half-CFG** toggle forwards `--prog-cfg` to the phone runtime and keeps guidance enabled only for the first `ceil(steps / 2)` steps as a speed/quality compromise;
- the status parser now keeps the live `CPU / GPU / NPU` line separate from the main stage/progress text;
- stdout is parsed in real time to display progress;
- the resulting PNG is loaded through `BitmapFactory.decodeFile()`;
- gallery saving uses the Android `MediaStore` API.
