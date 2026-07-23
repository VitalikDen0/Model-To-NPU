# Historical Performance Data & Optimization Archive

This document preserves historical performance measurements, optimization experiments, and technical notes from earlier development phases. These records are kept for reference and transparency.

**Languages:** [English](HISTORY_EN.md) | [Русский](HISTORY_RU.md)

In the current git history, the `0.3.x` line is represented by the single tag **`v0.3.0`**.

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

### v0.2.4 — Native C accelerator / runtime groundwork

- Optional native C accelerator for scheduler/layout hot spots.
- Transition snapshot; exact APK artifact not preserved.
- The work-in-progress commit message still used the label `v0.2.4-beta`, but the published repository tag is **`v0.2.4`**.

### v0.2.5 — Burst mode, runtime accel staging fix

- QNN `burst` default.
- Native C accelerator staging fix for Android shared-storage `dlopen`.
- Local review: **75.6 s total** (`CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`).

---

## 0.4.x release archive

The `0.4.x` line was the period where the repository shifted from a single fixed-shape SDXL path into a more practical phone product line with APK packaging, runtime delivery fixes, and later the decoder-regression investigation.

### v0.4.0 — Variable resolution + self-contained APK foundations

- `phone_generate.py` stopped assuming hardcoded `1024×1024` everywhere and gained width/height-aware paths;
- resolution-scoped context directories were introduced via `context/{W}x{H}/`;
- `export_split_unet.py` and `build_all.py` gained resolution-aware export/build support;
- APK got width/height controls and exported `SDXL_QNN_WIDTH` / `SDXL_QNN_HEIGHT`;
- standalone Termux/runtime packaging groundwork was added.

### v0.4.1 — Bundled runtime payload and stale-script drift fix

- APK started bundling the current phone runtime payload (`generate.py`, server binary, optional accel pieces);
- app-side execution began preferring bundled runtime files over stale phone leftovers;
- verify/runtime inspection paths were improved so stale argparse/runtime mismatches became easier to diagnose;
- deploy flow started pushing `qnn-multi-context-server` and resolution-scoped contexts more consistently.

### v0.4.2 — No clearly published standalone milestone

- No clean public `v0.4.2` marker was found in the current git history as a stable documented release.
- The line appears to have moved directly through iterative runtime/package work into the `v0.4.3` series.

### v0.4.3 — Shared prewarm reuse and runtime hotfix series

- app-open prewarm and foreground generation were reworked toward shared reuse instead of warming throwaway helpers;
- bundled runtime delivery became more deterministic, including payload refresh behavior;
- TAESD payload staging and preview/runtime path handling were tightened;
- shared-server startup and FIFO readiness issues were addressed across several follow-up refresh commits;
- this line represents multiple fixes (`010331d`, `a3e9376`, `e987a11`, `c5bac22`) rather than a single one-shot change.

### v0.4.4 — Preset-only APK smoothing pass

- APK hid arbitrary manual `WxH` editing in favor of validated presets;
- preview/final image decode/display path was smoothed to reduce UI pressure;
- APK-side exported QNN profile was softened to reduce device-lag / crash risk.

### v0.4.5 — Stability rollback for problematic shared reuse

- foreground generation stopped relying on the more aggressive shared prewarm/server reuse path;
- the APK line rolled back toward safer behavior after late-step freeze risk;
- QNN perf profile was returned to `burst` for the foreground path.

### v0.4.6 — Stability-first APK refresh

- public APK line kept background prewarm disabled;
- deterministic runtime payload refresh behavior was retained;
- runtime staging and packaging became more conservative and reproducible;
- this line was documented around the now-familiar **34.6 s cold-start APK** proof framing.

### v0.4.7 — CFG / TAESD UX hotfix

- exact user CFG values, including `1.0`, were forwarded correctly instead of silently snapping back to the runtime default;
- TAESD/live-preview failures were surfaced as non-fatal warnings instead of silently confusing the user;
- docs and proof references were aligned to the current public state.

### v0.4.8-beta — Bundled Python runtime, dual paths, TAESD off in APK

- APK gained a bundled Python 3.13 runtime for less Termux dependency on supported flows;
- root/no-root dual base-dir handling was introduced with auto-switch behavior;
- TAESD preview was intentionally disabled in the APK line because shared HTP preview hurt the fast path and GPU backend loading in-app was still problematic.

### v0.4.8-beta2 — Runtime bug fix and better error UX

- APK runtime bug fixes landed in the Android line;
- explicit error-state handling improved, including a dedicated copy-error action;
- seed parsing and related input paths became safer.

### v0.4.8-beta3 — Decoder speed fix and honest residual-tail note

- `qnn-multi-context-server` HTP perf mode was strengthened (`DCVS` off, MAX corners, RPC latency/polling controls);
- local validation moved decoder latency from the `~820 ms` class toward roughly `~725–776 ms`;
- a residual ~`50 ms` tail versus the historical ideal marker remained and was documented explicitly instead of being hidden.

### v0.5.0 — SDXL / WAN Tab Split and APK UI Refresh

- Split SDXL and WAN 2.1 into independent UI tabs with separate settings storage;
- Introduced dynamic directory scanning for `context/` and `context/lora_slots/` for automatic slot discovery;
- LMK (Low Memory Killer) defense: added explicit NPU memory unload controls and cleanup handlers.

### v0.5.1 — Dynamic NPU Graph Surgery Readiness Release

- Android APK updated (`v0.5.1`) with LMK prevention and dynamic directory scanners;
- QNN Windows converter fix (`sanitize_onnx_types.py`): casting INT64 to INT32 prevents StridedSlice memory alignment panics;
- Published technical roadmap (`ROADMAP_DYNAMIC_NPU_SURGERY`): pivoting from static resolution buckets to dynamic weight injection & attention masking (`-10000.0` instead of $-\infty$ to preserve quantization scale);
- Architectural review: designed true zero-copy via single cross-registered `rpcmem` memory handle.

---

## Optimization experiments archive

### Zero-copy pointer swap (QNN HTP Limit & Architecture Fix)

- *Initial Attempt:* Pointer swapping between encoder output and decoder input triggered **QNN error 6004** due to unregistered `Qnn_MemHandle` buffers.
- *Architecture Solution:* Instead of swapping pointers at runtime, `qnn-multi-context-server` allocates a single shared `rpcmem` memory block at startup. It is registered as Output Buffer during Encoder setup and as Input Buffer during Decoder setup. This completely eliminates `memcpy` without pointer swapping!

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
