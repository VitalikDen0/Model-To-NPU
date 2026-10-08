# Model-to-NPU Pipeline for Qualcomm Snapdragon

[![Snapdragon 8 Elite](https://img.shields.io/badge/SoC-Snapdragon%208%20Elite%20(SM8750)-red.svg)](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-elite-mobile-platform)
[![Qualcomm Hexagon](https://img.shields.io/badge/NPU-Hexagon%20V79%20HTP-blue.svg)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Engine-v0.6.0--core%20(Monolithic)-orange.svg)](PROJECT_STATE.md)

**Languages:** [English](README_EN.md) | [Русский](README_RU.md) | [Project State](PROJECT_STATE.md) | [Android APK](APK/README.md)

The world's first fully functional on-device **Stable Diffusion XL (SDXL)** pipeline running **natively on the Qualcomm Hexagon NPU** (Snapdragon 8 Elite / OnePlus 13) without cloud servers, without Termux, without Root, and without model splitting.

---

## ⚡ What's New in v0.6.0-core

### 1. 🚀 72.7% NPU Hardware Roofline Limit Achieved
- **7.13 TOPS sustained** across all 362 layers of SDXL 2.57B (W8A16) on Hexagon V79 HTP.
- UNet single pass latency dropped to **880.3 ms** (down from ~956 ms and ~2411 ms historically).
- 8-step generation UNet time dropped to **11.59 seconds**!
- Total Cold Start generation (including model deserialization from UFS 4.0 flash) is **~20–21 seconds**. Subsequent warm generations take **~14 seconds**.
- **Host overhead is virtually eliminated (< 0.7%)**: ARM64 NEON FP16 MLP FMA (3.4 ms), 64KB L1 double-buffer copy to RPCMEM (2.2 ms, ~49.2 GB/s), FastRPC transport (1.7 ms).
- **NPU Pipeline Duty Cycle = 98.7%**!

### 2. 🧩 Monolithic W8A16 UNet (No More Model Splitting)
- The UNet model is **no longer split** into separate encoder and decoder halves!
- A single unified context (`unet.serialized.bin`, ~2.44 GiB) runs directly in NPU memory.
- Eliminated 11 skip-connection memory buffering bottlenecks (82.5 MB per step), process synchronization delays, and split-graph IPC overhead.

### 3. 📐 Dynamic Arbitrary Resolution Engine (0.26 MP to 2.36 MP)
- **Arbitrary aspect ratios and resolutions** from 512×512 up to 1536×1536 / 1344×1728 on the fly without recompiling models or switching contexts.
- **Centered Spatial-CFG Sub-Canvas Framing**: The active image window is placed at the optical center $(512, 512)$ with spatial CFG masking ($w_{\text{CFG}} = 3.5$ active, transitioning to $1.0$ uncond at borders).
- **0% extra NPU latency overhead** compared to standard square.
- Continuous noise variance preserves 100% GroupNorm numerical stability (zero NaNs).
- Completely eliminates edge reflection artifacts and bilateral symmetry (Rorschach mirror seams).
- Integrated Catmull-Rom bicubic reconstruction + Contrast-Adaptive Sharpening (CAS, 59 ms).

### 4. 📱 Zero-Root, Zero-Termux, Zero-Python Architecture
- **No Root required**.
- **No Termux or Python runtime required** on the phone for inference.
- Standalone native C inference engine (`qnn-multi-context-server`) handles CLIP, Monolithic UNet, and VAE directly via QNN System & Backend APIs.
- Can be executed with a single command via ADB or embedded into an Android APK.

### 5. 📲 Upcoming Android APK Update (v0.6.0)
- The Android application in `APK/` is being updated to version **0.6.0** to directly incorporate the native monolithic C-engine and dynamic resolution UI. Stay tuned!

---

## 🖼️ Gallery

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

All gallery samples above are **1024×1024** outputs from the Lightning-merged SDXL path running directly on-device.

---

## 📱 Proof that it actually runs on-device

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
      <b>v0.4.7 cold-start APK proof — 34.6s total</b><br>
      <img src="https://github.com/user-attachments/assets/04b6e61a-79d6-4ce5-a7d6-158461ca97e6" alt="Current phone-side proof screenshot at 34.6 seconds (cold start)" width="100%"><br>
      <sub>Measured accelerator-visible time inside this run: ~16.25 s.</sub>
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

### On-Device Telemetry & Milestone Progression

Public screenshot lineage so far: **273.6 s → 100.8 s → 78.0 s → 34.6 s → 11.59 s (UNet 8 steps)**!

```text
================================================================================
  OnePlus 13 (Snapdragon 8 Elite / Hexagon V79 HTP) — On-Device Execution Log
================================================================================
  [Init] QNN backend + system loaded in 142ms
  [Load] CLIP-L + CLIP-G loaded in 312ms
  [Load] UNet Monolithic W8A16 (2.44GB) loaded in 5821ms
  [Load] VAE Decoder FP16 loaded in 284ms
  [CLIP] Text encode finished: 281ms
  [Zero-Copy Denoise] Running autonomous 8-step monolithic UNet in NPU memory...
  [UNet 1/8] 880.3ms (CFG active)  | 6 HVX threads @ 7.13 TOPS
  [UNet 2/8] 879.8ms (CFG active)  | 6 HVX threads @ 7.13 TOPS
  [UNet 3/8] 878.9ms (CFG active)  | 6 HVX threads @ 7.13 TOPS
  [UNet 4/8] 876.5ms (CFG active)  | 6 HVX threads @ 7.13 TOPS
  [UNet 5/8] 874.2ms (CFG dCache)  | 6 HVX threads @ 7.13 TOPS
  [UNet 6/8] 874.1ms (CFG dCache)  | 6 HVX threads @ 7.13 TOPS
  [UNet 7/8] 874.0ms (CFG dCache)  | 6 HVX threads @ 7.13 TOPS
  [UNet 8/8] 874.5ms (CFG dCache)  | 6 HVX threads @ 7.13 TOPS
  [Zero-Copy Denoise] UNet total: 11,590ms (880.3ms/pass, Duty Cycle: 98.7%)
  [VAE] Decode: 2,204ms (FP16)
  [Save] PNG written to storage in 189ms
  Total Cold Generation: ~20.9s | Warm Generation: ~14.1s
================================================================================
```

---

## 📊 Performance Benchmarks (Snapdragon 8 Elite / OnePlus 13)

### UNet Pass Latency Breakdown (1024×1024)

| Stage | Hardware Unit | Latency | Share | Details |
| :--- | :--- | :---: | :---: | :--- |
| **Host `temb` MLP** | Oryon CPU (ARM NEON) | 3.43 ms | 0.4% | FP16->FP32 FMA (`vld1q_f16`, `vfmaq_f32`) |
| **Host -> RPCMEM ION** | System Memory Bus | 2.20 ms | 0.2% | 64 KB L1 double copy (~49.2 GB/s) |
| **Quant & Transpose** | Oryon CPU | 0.41 ms | 0.0% | NEON vectorized quantization |
| **FastRPC Transport** | ARM <-> Hexagon Bus | 1.72 ms | 0.2% | ION buffer synchronization |
| **Pure Hexagon V79 Accel** | **Hexagon V79 HTP (NPU)** | **880.31 ms** | **98.9%** | **6 HVX hardware threads unlocked** |
| **Total UNet Pass** | — | **888.07 ms** | **100%** | **NPU Duty Cycle: 98.7%** |

### Historical Progression

| Version | Total Time | UNet (8 steps) | CLIP | VAE | Architecture & Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **v0.1.0** | 273.6 s | — | — | — | Initial public proof-of-concept |
| **v0.1.3** | 104.4 s | 91.5 s | 2.0 s | 9.0 s | Added mmap |
| **v0.2.0** | 79.7 s | 72.4 s | 2.9 s | 3.4 s | Sustained high performance mode |
| **v0.2.5** | 75.6 s | 66.6 s | 2.8 s | 3.0 s | Native accel helper, per-step `qnn-net-run` |
| **v0.3.0** | 30.4 s | 19.3 s | 2.8 s | 1.9 s | Persistent QNN server, split UNet (enc+dec) |
| **v0.4.7** | 34.6 s | 14.2 s | 0.1 s | 1.8 s | APK cold start marker (split UNet) |
| **v0.6.0-core** | **20.0–21.0 s** | **11.59 s** | **0.28 s** | **2.20 s** | **Monolithic W8A16 UNet, 72.7% NPU Roofline, 6 HVX threads, Zero-Root/Termux** |

> *Note: v0.6.0-core total time (~20–21 s) is a cold-start measurement including deserializing 2.44 GiB of model weights from UFS 4.0 flash storage. Subsequent warm generations run in ~14 s total!*

### Dynamic Resolution Benchmarks (Centered Spatial-CFG)

| Target Resolution | Aspect Ratio | Megapixels | UNet (8 steps) | Wall Time (Cold) | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1024 × 1024** | 1:1 Square | 1.05 MP | **11.59 s** | 21.06 s | Reference ground truth |
| **832 × 1216** | 9:16 Portrait | 1.01 MP | **11.64 s** | 20.64 s | Perfect anatomy, zero edge seams |
| **1024 × 768** | 4:3 Landscape | 0.79 MP | **11.91 s** | 20.74 s | High detail landscape |
| **1344 × 1728** | 3:4 Hi-Res | 2.32 MP | **11.21 s** | 20.03 s | Ultra-crisp Catmull-Rom CAS |

---

## 🛠️ Quick Start & Usage

### Hardware & Software Requirements
- **Target Device**: Qualcomm Snapdragon 8 Elite (OnePlus 13 or similar) with Hexagon V79 HTP.
- **RAM**: 16 GB LPDDR5X (Peak memory footprint ~3.5 GB for monolithic pipeline).
- **Storage**: ~8 GB UFS 4.0 storage on device.
- **Root**: **Not required!**
- **Termux**: **Not required!**

### 1. Build the Native Server (Host PC)
Requires Android NDK (r26+) and Qualcomm QAIRT SDK (2.31+):
```bash
python scripts/build_qnn_multi_context_server.py --deploy
```

### 2. Push Models & Assets to Phone
```bash
adb push D:/platform-tools/sdxl_npu/unet.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/vae_decoder.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/clip_l.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/clip_g.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/tokenizer /data/local/tmp/sdxl_qnn/
```

### 3. Generate Image on Phone via ADB
```bash
adb shell "LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib /data/local/tmp/sdxl_qnn/bin/qnn-multi-context-server \
  --backend libQnnHtp.so \
  --system libQnnSystem.so \
  --generate-dyn \
  --prompt 'masterpiece, 1girl, cyberpunk aesthetic, neon city, highly detailed' \
  --width 832 --height 1216 --steps 8 --cfg 3.5 --seed 42 \
  --output /sdcard/Download/output.png"
```

---

## 📁 Repository Structure

- `NPU/qnn_multi_context_server.c` — The complete standalone C inference engine (NEON, RPCMEM, 6 HVX threads, Spatial-CFG).
- `scripts/build_qnn_multi_context_server.py` — Host build script compiling with Android NDK Clang.
- `phone_generate.py` — Standalone Python entrypoint for debugging and evaluation.
- `PROJECT_STATE.md` — Complete engineering record, hardware roofline calculations, and architectural notes.
- `APK/` — Android Studio project for the native mobile app (v0.6.0 update in development).
- `SDXL/` — SDXL conversion, calibration, and ONNX graph manipulation tools.
- `WAN 2.1 1.3B/` — WAN research workspace and video diffusion tools.

---

## 📜 License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).  
You are free to use, modify, and distribute this software for non-commercial and research purposes.
