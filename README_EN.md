# Model-to-NPU Pipeline for Snapdragon

**Languages:** [English](README_EN.md) | [Русский](README_RU.md)

> **SDXL on Snapdragon 8 Elite NPU — ~30 s total** (UNet ~19 s, VAE ~1.9 s, CLIP ~9 ms cached)
> at 1024×1024, 8 steps, CFG=3.5, progressive guidance.

> [!TIP]
> End-to-end SDXL flow is available and practically validated (`checkpoint -> final phone-generated PNG`).
> Work on **SD3**, **Flux**, **Wan** and other model families has started — they will be released as the methods are developed and validated.

<p align="center">
  <b>Repository for model-to-NPU pipelines on Qualcomm Snapdragon devices</b><br>
  Current implemented pipeline: <b>SDXL on Qualcomm Hexagon NPU</b>.
</p>

---

## What is this?

This repository is intended to grow into a home for multiple **model-specific pipelines** targeting Qualcomm Snapdragon NPUs.

- each model family gets its own folder;
- the **current implemented folder** is `SDXL/`;
- an early exploratory `WAN 2.1 1.3B/` workspace exists for Wan 2.1 T2V 1.3B model scouting;
- work on **SD3**, **Flux**, **Wan** and other model families has started — they will be released as the optimization methods are developed and validated;
- the **current Android app** lives in `APK/`;
- shared deployment helpers and assets live in `scripts/`, `tokenizer/`, and top-level helpers.

Right now the implemented and documented pipeline is **Stable Diffusion XL** running **natively on the phone NPU** (Hexagon HTP). The working SDXL path uses CLIP-L, CLIP-G, Split UNet (encoder + decoder), and VAE on the device.

**Current tested model combination:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310) + [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) (ByteDance)

> **Performance note:** the public beta timings, APK screenshots, and example outputs in this repository assume the **Lightning LoRA is already baked into the UNet**. That merge is not just a convenience step — it is the practical speed path used here.
> **Resolution note:** the currently documented exports, context binaries, previews, and example images are built specifically for **1024×1024**.

## Current status

- **Repository direction:** multi-model Snapdragon NPU pipelines (SDXL, SD3, Flux, Wan, ...)
- **Currently implemented family:** `SDXL/`
- **Exploratory Wan workspace:** `WAN 2.1 1.3B/` (candidate scouting, download helpers, phone probing, 480p-first plan)
- **Current phone app target:** SDXL
- **Status of scripts:** full practical SDXL loop (checkpoint → image) re-validated on current layout
- **Status of docs:** updated to the current known layout

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
| Android NDK | r26+ (for `.so` build and `qnn-multi-context-server`) |
| PyTorch | 2.x |
| Windows | 10/11 |

## Performance

Measured on OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM):

### Current (v0.4.0) — Variable resolution + self-contained APK

Variable resolution support (512×512 to 1536×1536, any multiple of 8). Per-resolution QNN context directories. APK resolution picker. `build_termux_prefix.py` for standalone prefix extraction.

### v0.3.0 — Persistent multi-context server

`seed=44`, `steps=8`, `CFG=3.5`, `--prog-cfg`, Live Preview OFF:

| Stage | Time | Notes |
| ----- | ---- | ----- |
| CLIP-L + CLIP-G | ~9 ms | cached result (first run ~2.8 s) |
| UNet (8 steps) | ~19.3 s | ~2411 ms/step via persistent server + RUN_CHAIN |
| VAE decoder | ~1.9 s | FP16 |
| **Total (warm)** | **~30.4 s** | |

Peak RAM: **~12 GB** out of 16 GB
Resolution: **1024×1024** (fixed)

### Previous versions

| Version | Total | UNet | CLIP | VAE | Notes |
| ------- | ----- | ---- | ---- | --- | ----- |
| v0.2.5 | 75.6 s | 66.6 s | 2.8 s | 3.0 s | burst + native accel, per-step qnn-net-run |
| v0.2.0 | 79.7 s | 72.4 s | 2.9 s | 3.4 s | sustained_high_performance |
| v0.1.3 | 104.4 s | 91.5 s | 2.0 s | 9.0 s | mmap enabled |
| v0.1.0 | 273.6 s | — | — | — | first public screenshot |

For detailed historical data and archived experiments, see [HISTORY_EN.md](HISTORY_EN.md).

## How UNet ~19 s was achieved — optimization deep dive

The UNet went from **66.6 s** (v0.2.5) to **~19.3 s** (v0.3.0) — a **3.4× speedup**. Here is exactly what changed and why each piece matters.

### 1. Persistent multi-context QNN server (biggest win)

**Before (v0.2.5):** every UNet step spawned a new `qnn-net-run` process. Each process had to:

- `fork()+exec()` — process creation overhead (~15–30 ms each);
- `dlopen()` QNN backend libraries every time;
- deserialize the context binary from disk (~1–3 s per context on first load);
- allocate and register `rpcmem` shared DSP memory;
- execute the graph;
- tear everything down and exit.

With 8 steps × 2 contexts (encoder + decoder) = **16 process spawns per image**, the cumulative overhead was enormous.

**After (v0.3.0):** a single **persistent C process** (`qnn-multi-context-server`) starts once, loads all context binaries, and keeps them alive. It speaks a simple stdin/stdout protocol:

```text
LOAD <id> <path>     → OK <graph> <inputs> <outputs>
RUN <id> <inputs> <outdir>  → OK <ms>
RUN_CHAIN <enc> <dec> ...   → OK <ms>
QUIT                        → OK
```

The server loads contexts once at startup, allocates `rpcmem` once, and all subsequent graph executions skip the entire process lifecycle. This alone eliminated ~47 s of pure overhead.

### 2. RUN_CHAIN — in-memory encoder→decoder piping

**Before:** after the encoder finished, its 11 skip-connection outputs (~82.5 MB total) were written to disk as raw files, then the decoder process read them back. This meant ~165 MB of disk I/O per step.

**After:** the `RUN_CHAIN` command runs encoder and decoder back-to-back inside the same server process. Skip connections are piped via `memcpy` between the encoder's output buffers and decoder's input buffers — **no intermediate file I/O at all**. The 11 skip connections + mid + temb stay in server-allocated `rpcmem` buffers.

Note: zero-copy pointer swap was attempted but QNN HTP requires registered `rpcmem` handles for each buffer, so `memcpy` is the minimum viable approach (see [HISTORY_EN.md](HISTORY_EN.md#zero-copy-pointer-swap-failed)).

### 3. FLOAT_32 direct fread

**Before:** tensor outputs from `qnn-net-run` were written as text (one float per line). Python then parsed each line with `float()`. For a 2.5 GB decoder, this text parsing was significant.

**After:** the server writes output tensors as raw binary (little-endian float32). Python reads them with a single `fread`-equivalent (`numpy.fromfile`). The FLOAT_32 detection in `phone_generate.py` checks the dtype string and routes through the fast binary path.

### 4. Eager preload — overlapped context loading

**Before:** contexts were loaded sequentially: CLIP first, then UNet encoder, then UNet decoder. The UNet contexts together took ~8 s to deserialize.

**After:** a background thread (`_eager_preload_unet()`) sends `LOAD` commands for UNet encoder and decoder contexts to the server **while CLIP is still running**. By the time CLIP finishes and UNet iteration begins, both UNet contexts are already warm in the server. This overlaps ~8 s of context loading with the CLIP pipeline.

### 5. CLIP result caching

**Before:** every generation run tokenized the prompt and ran CLIP-L + CLIP-G through the QNN backend (~2.8 s total).

**After:** CLIP results (hidden states + pooled output) are cached to disk keyed by prompt hash. On cache hit, CLIP completes in ~9 ms (just file reads). Since many test/iteration runs use the same prompt, this saves ~2.8 s on every repeat run.

### 6. QNN burst mode + native runtime accelerator

Already present in v0.2.5 but still contributing:

- **Burst mode:** sets QNN performance profile to `burst` (maximum HTP clock for short sustained workloads).
- **Native C accelerator:** `libsdxl_runtime_accel.so` accelerates scheduler math and tensor layout operations that would otherwise run in pure Python/numpy.

### Theoretical minimum analysis

With the current architecture:

- **UNet compute:** ~2411 ms/step × 8 steps = ~19.3 s — this is the actual NPU silicon time and cannot be reduced without fewer steps or faster hardware.
- **VAE:** ~1.9 s — already near-optimal.
- **CLIP:** ~9 ms cached, ~2.8 s cold — cached is effectively free.
- **Server overhead per step:** ~5–10 ms (memcpy + protocol) — negligible.
- **Theoretical warm-run minimum:** ~21–22 s (UNet + VAE + minimal orchestration).
- **Current actual:** ~30.4 s — the remaining ~9 s gap is Python orchestration, numpy scheduler math, and file writes for the final PNG.

## Architecture

```text
          ┌──────────────────────────────────────────────────────────────────┐
          │                      Phone (NPU)                                │
          │                                                                 │
          │  ┌─────────────────────────────────────────────────────────┐    │
          │  │        qnn-multi-context-server (persistent C process)  │    │
          │  │                                                         │    │
Prompt ──▶│  │  CLIP-L ──┐                                             │    │
          │  │  (FP16)   ├──▶ concat [1,77,2048]                       │    │
          │  │  CLIP-G ──┘    + pooled [1,1280]                        │    │
          │  │  (FP16)        + time_ids [1,6]                         │    │
          │  │                    │                                     │    │
          │  │         ┌─────────▼──────────┐                          │    │
          │  │         │  RUN_CHAIN × 8     │                          │    │
          │  │         │  encoder ──memcpy──▶ decoder                   │    │
          │  │         │  (11 skip conns     │                          │    │
          │  │         │   in server memory) │                          │    │
          │  │         └─────────┬──────────┘                          │    │
          │  │                   ▼                                      │    │
          │  │              VAE decoder ──▶ PNG                         │    │
          │  └─────────────────────────────────────────────────────────┘    │
          └──────────────────────────────────────────────────────────────────┘
```

**Split UNet:** The full FP16 UNet (~5 GB) exceeds the HTP allocation limit (~3.5 GB), so it is split into encoder (conv_in + down_blocks + mid_block, 2.52 GB) and decoder (up_blocks + conv_out, 2.69 GB). The encoder passes 11 skip-connections + mid + temb to the decoder via in-memory `memcpy` (RUN_CHAIN).

**Scheduler:** EulerDiscrete, trailing spacing (Lightning requirement), pure numpy.

**Tokenizer:** Pure Python BPE (no HuggingFace/transformers), identical to the CLIP tokenizer.

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

```bash
# Build from checkpoint (early stages)
python scripts/build_all.py --checkpoint path/to/model.safetensors
```

Or the end-to-end wrapper:

```powershell
pwsh SDXL/run_end_to_end.ps1 -ContextsDir path/to/context_binaries
```

Step by step:

```bash
# 1. Convert checkpoint to diffusers format
python SDXL/convert_sdxl_checkpoint_to_diffusers.py

# 2. Merge Lightning LoRA into UNet
python SDXL/bake_lora_into_unet.py

# 3. Export all components to ONNX
python SDXL/export_clip_vae_to_onnx.py
python SDXL/export_sdxl_to_onnx.py

# 4. Convert to QNN
python SDXL/debug/convert_clip_vae_to_qnn.py
python SDXL/debug/convert_lightning_to_qnn.py

# 5. Build Android model libraries (.so)
python SDXL/debug/build_android_model_lib_windows.py

# 6. Build the persistent multi-context QNN server
python scripts/build_qnn_multi_context_server.py
```

### 3. Deploy to phone

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/context_binaries \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn_sdk/lib/aarch64-android \
  --qnn-bin-dir /path/to/qnn_sdk/bin/aarch64-android
```

### 4. Termux setup (on phone)

```bash
pkg install python python-numpy python-pillow
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
- `SDXL_QNN_PERF_PROFILE=burst`
- persistent `qnn-multi-context-server` with RUN_CHAIN

#### Via APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

The APK provides a full GUI: prompt, negative prompt, CFG, steps, seed, contrast stretching, progress bar, live CPU / GPU / NPU temperatures, and save to gallery.

## Minimal phone file structure

```text
/sdcard/Download/sdxl_qnn/
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
│   └── libQnnHtpNetRunExtensions.so       (optional, auto-used when present)
├── bin/
│   └── qnn-multi-context-server           (persistent QNN server)
└── outputs/                               (generated PNGs)
```

## Project structure

```text
├── README.md                 ← language landing page
├── README_RU.md              ← Russian documentation
├── README_EN.md              ← you are here
├── HISTORY_EN.md             ← historical performance archive
├── HISTORY_RU.md             ← historical archive (Russian)
├── LICENSE                   ← PolyForm Noncommercial License 1.0.0
├── NOTICE                    ← required attribution / notice lines
├── phone_generate.py         ← standalone generator (runs on phone)
├── tokenizer/                ← BPE tokenizer files (CLIP)
├── examples/                 ← phone-side layout examples and samples
├── scripts/
│   ├── deploy_to_phone.py
│   ├── build_qnn_multi_context_server.py
│   ├── build_all.py
│   └── ...
├── NPU/
│   ├── qnn_multi_context_server.c  ← persistent server C source
│   └── build/                      ← compiled server binary
├── SDXL/                     ← SDXL conversion, build, and lab scripts
│   ├── debug/                ← experimental/diagnostic scripts
│   └── ...
├── WAN 2.1 1.3B/             ← Wan 2.1 T2V exploration workspace
└── APK/                      ← Android application
```

## Limitations

- **Resolution is fixed** at 1024×1024 — others need full re-conversion
- **The documented speed path assumes the Lightning LoRA has been baked into the UNet**
- **VAE FP16** slightly compresses color range → percentile contrast stretching is applied
- **TAESD live preview is optional** — uses QNN GPU or falls back to ONNX
- **CFG > 1.0 is expensive** — roughly 2× the no-CFG path
- **Termux required** — Python runtime for `phone_generate.py`
- Tested only on **OnePlus 13 (SM8750)**

## Known issues

- First run of each component is slower (context loading)
- Low RAM may cause process kill — close other apps
- On Android 11+, the APK may need "all files access" permission
- numpy and torch use different RNGs — the same seed produces different but valid images

## License

This repository is distributed under the **PolyForm Noncommercial License
1.0.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

In short:

- you may use, study, modify, and fork the project for **non-commercial**
  purposes;
- redistributions must include the PolyForm terms together with the
  `Required Notice:` lines from [`NOTICE`](NOTICE);
- third-party dependencies keep their own licenses.

Dependencies:

- Qualcomm QAIRT SDK — proprietary Qualcomm license
- SDXL-Lightning LoRA (ByteDance) — Apache 2.0
- Stable Diffusion XL — CreativeML Open RAIL-M
