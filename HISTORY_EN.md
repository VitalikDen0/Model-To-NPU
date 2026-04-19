# Historical Performance Data & Optimization Archive

This document preserves historical performance measurements, optimization experiments, and technical notes from earlier development phases. These records are kept for reference and transparency.

**Languages:** [English](HISTORY_EN.md) | [Русский](HISTORY_RU.md)

---

## Historical timeline

### v0.1.2 — Live Preview (TAESD)

- APK gained optional **Live Preview (TAESD)** using `phone_gen/taesd_decoder.onnx` + `onnxruntime` on the phone.

### v0.1.3 — QNN mmap, first optimizations (2026-03-31)

- Phone runtime and APK launch path now enable QNN `mmap` by default.
- Control run on OnePlus 13: **104.4 s total** (`CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`) at `1024×1024`, `8` steps, `CFG=1.0`.

### v0.2.0 — Thermal monitoring, sustained_high_performance

- Phone runtime and APK now show live **CPU / GPU / NPU** temperatures.
- Default perf profile: `sustained_high_performance`.
- Auto-enable HTP backend extensions when `libQnnHtpNetRunExtensions.so` is deployed.
- Best run: **79.7–80.6 s total** with progressive CFG on OnePlus 13.

### v0.2.1 — App-private cache

- APK now routes transient runtime files through app-private cache directories instead of shared storage.

### v0.2.2 — TAESD preview repair

- TAESD preview wiring repaired for QNN path.
- APK preview timing parsing handles `QNN GPU` preview lines again.

### v0.2.3 — Historical fast path (pre-reset)

- Split-UNet reuse pass made early guided steps decay instead of hovering near ~12 s plateau.
- A runtime-only run once reached **62.0 s total** (`CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`) with Live Preview OFF.
- That run was real, but the exact phone-side state was not archived before a factory reset, so it is now historical rather than reproducible.
- UNet step progression: CFG steps 1..4: `9.765 → 8.230 → 8.386 → 7.936 s`; no-guidance steps 5..8: `5.377 → 5.513 → 5.294 → 5.479 s`.

### v0.2.4-beta — Native C accelerator

- Optional native C accelerator for scheduler/layout hot spots.
- Transition snapshot; exact APK artifact not preserved.

### v0.2.5 — Burst mode, runtime accel staging fix

- QNN `burst` default.
- Native C accelerator staging fix for Android shared-storage `dlopen`.
- Local review: **75.6 s total** (`CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`).

---

## Optimization experiments archive

### Zero-copy pointer swap (FAILED)

Attempted to swap decoder input buffer pointers to point directly at encoder output buffers, eliminating memcpy in the RUN_CHAIN pipeline. **QNN error 6004** — QNN HTP uses registered shared memory (`rpcmem`). Tensor buffers have specific registered memory handles (`Qnn_MemHandle`). Swapping pointers to different addresses causes "Failed to find memHandle" because the new addresses aren't registered. **Conclusion:** memcpy is mandatory for piping between encoder output and decoder input in QNN HTP.

### Persistent daemon approach (REGRESSED)

Using `qnn-context-runner` as a persistent daemon for context reuse initially seemed promising but consistently regressed on the rebuilt phone:
- Daemon-all: ~111.3 s → optimized to ~63.3 s (still slower than stock ~60.1 s).
- Dummy warmup pass during CLIP: ~110.5 s (too expensive to hide).
- `QnnGraph_setConfig` for VTCM/HVX: ~120.2 s (further regression).

### Monolithic INT8 UNet (CATASTROPHICALLY SLOW)

True 8W8A quantized monolithic UNet from QAIRT 2.44 with anime-aligned calibration:
- Parity: cosine ~0.99913 vs W8A16 control (good).
- Speed: ~161-218 s/step vs ~2.55 s/step for W8A16 (**63× slower**).
- Profiler confirmed the graph executes on HTP (not CPU fallback), but is compiled into a catastrophically expensive graph: ~1.35×10¹² accelerator cycles vs ~3.73×10⁹ for W8A16.

### HVX thread ceiling

Backend extension config is graph-name sensitive. With correct graph names and `hvx_threads=8`, profile clamps to `6`. The 6-thread ceiling is not explained by thermal throttling (cooling device `cdsp_sw_hvx` shows `cur_state=0`).

### tmpfs workdir (NO IMPROVEMENT)

Moving `SDXL_QNN_WORK_DIR` to `/tmp` tmpfs did not help and actually regressed to ~69.4 s (vs ~62.0 s baseline). The residual overhead is not explained by plain ext4 workdir I/O alone.

### Batched CLIP (MIXED)

Experimental batched CLIP path improved CLIP time to ~1.83-2.03 s but worsened end-to-end runs to ~69.6-70.4 s. Kept as opt-in only (`SDXL_QNN_BATCH_CLIP=1`).

---

## Validated full loop (2026-04-06)

Checkpoint used: `waiIllustriousSDXL_v160.safetensors` (WAI Illustrious SDXL v1.60 + SDXL-Lightning 8-step LoRA baked in).

Host artifacts:
- `build/sdxl_work_wai160_20260406/diffusers_pipeline/`
- `build/sdxl_work_wai160_20260406/unet_lightning_merged/`
- `build/sdxl_work_wai160_20260406/onnx_clip_vae/`
- `build/sdxl_work_wai160_20260406/onnx_unet/unet.onnx` + `unet.onnx.data`

Validated output: `NPU/outputs/wai160_phone_native_cfg35_20260406.png`

---

## Thermal observations

In warmed-up full runs, the practical thermal envelope:
- **CPU:** ~59–70°C
- **GPU:** ~50–52°C
- **NPU:** ~57–72°C (short spikes up to ~78°C)
- An early one-line CPU spike to `88.8°C` appeared before the first run stabilized — likely a transient sensor jump.

---

## Technical notes

- TAESD preview root cause (2026-04-01): Old deployed `libTAESDDecoder.so` produced outputs clipped to `[0,1]` with only ~0.21 RGB correlation vs ONNX. Rebuilding from current ONNX restored range to `[-1.18, 1.23]`, reached ~0.9999 correlation.
- After switching phone runtime to QAIRT 2.44, preview was still broken because GPU libs/context were stale 2.31 artifacts. Both GPU runner and TAESD context needed regeneration.
- `phone_generate.py::_resolve_exec_binary()` must create `WORK_DIR/bin` before staging `qnn-net-run`.
- QAIRT packaging: `libQnnHtpV79Skel.so` may be absent from `lib/aarch64-android` and live under `lib/hexagon-v79/unsigned`.
