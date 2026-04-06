# SDXL scripts overview

A detailed map of the current `SDXL/` folder.

> [!IMPORTANT]
> `SDXL/` currently contains **more than just the shortest happy-path**. It also includes diagnostics, parity checks, calibration helpers, QAIRT/QNN workarounds, and experimental branches. That is expected for a live R&D repository.
> [!NOTE]
> After the SDXL cleanup pass, lab/diagnostic/experimental scripts were moved under `SDXL/debug/`, while the root `SDXL/` keeps the shortest practical public path.

## How to read this map

- **core-happy-path** — part of the main documented pipeline;
- **deploy-runtime** — phone/runtime/Android deployment pieces;
- **calibration-data** — calibration/raw/input-list generation;
- **verification-debug** — parity checks, comparisons, diagnostics;
- **experimental-alternative** — research branches, not the main path;
- **utility** — helper rewriters, patches, and evaluators.

## Two real branches in this repository

### 1. Public beta runtime

This is the easier path that currently matches the main documentation:

- `phone_generate.py`
- `scripts/deploy_to_phone.py`
- split-UNet context binaries (`unet_encoder_fp16` + `unet_decoder_fp16`)
- CLIP-L / CLIP-G / VAE context binaries
- the APK in `APK/`

This branch is the best match for the current `README`s.

### 2. Experimental Lightning/QNN lab branch

This is the active research branch inside `SDXL/`:

- `convert_lightning_to_qnn.py`
- `debug/run_phone_lightning.py`
- `debug/run_full_phone_pipeline.py`
- `debug/trace_unet_layer_parity.py`
- `debug/compare_*`
- `debug/generate_*_references.py`
- shell helpers for `ctxgen`

It matters, but it is **not the shortest public entry point** for first-time readers.

## Recommended minimum reading order

If you want the shortest useful route through the repository, start with:

1. `convert_sdxl_checkpoint_to_diffusers.py`
2. `bake_lora_into_unet.py`
3. `export_clip_vae_to_onnx.py`
4. `export_sdxl_to_onnx.py`
5. `scripts/build_all.py`
6. `scripts/deploy_to_phone.py`
7. `phone_generate.py`
8. `README_EN.md`
9. `SDXL/LESSONS_LEARNED.md`

## Full Python inventory

### Core happy-path

| File | Purpose | Status |
| --- | --- | --- |
| `convert_sdxl_checkpoint_to_diffusers.py` | Converts the original `.safetensors` checkpoint into a Diffusers directory. | ✅ Main step |
| `bake_lora_into_unet.py` | Permanently fuses the SDXL-Lightning LoRA into the base UNet. | ✅ Main step |
| `export_clip_vae_to_onnx.py` | Exports CLIP-L, CLIP-G, and the VAE decoder to ONNX. | ✅ Main step |
| `export_sdxl_to_onnx.py` | Exports the UNet and related SDXL components to ONNX. | ✅ Main step |
| `convert_clip_vae_to_qnn.py` | Converts CLIP/VAE ONNX models into QNN artifacts. | ⚠️ Main but dev/layout-sensitive |
| `convert_lightning_to_qnn.py` | Converts the Lightning UNet into the QNN model pipeline. | ⚠️ Main but still experimental |
| `debug/quantize_unet.py` | Quantizes the UNet (W8A16 / INT8) from calibration data. | ✅ Main technical step |
| `generate.py` | Host-side generator/orchestrator that drives parts of the pipeline over ADB. | ⚠️ Useful, but not the simplest public entry |

### Deploy / runtime

| File | Purpose | Status |
| --- | --- | --- |
| `debug/run_phone_lightning.py` | Runs the phone-side Lightning UNet branch through ADB and the QNN runtime. | ⚠️ Experimental runtime |
| `debug/run_full_phone_pipeline.py` | Runs CLIP + UNet + VAE on the phone as a research full-pipeline branch. | ⚠️ Experimental runtime |
| `build_android_model_lib_windows.py` | Builds Android `.so` files from QNN model.cpp/model.bin on Windows/NDK. | ⚠️ Important build step, but platform-specific |
| `debug/export_split_unet.py` | Exports/prepares split UNet artifacts for AI Hub and phone-side use. | ⚠️ Alternative runtime branch |
| `debug/export_and_compile_aihub.py` | Handles export/compile through Qualcomm AI Hub. | ⚠️ Alternative cloud branch |

### Calibration / data prep

| File | Purpose | Status |
| --- | --- | --- |
| `debug/generate_calibration_prompts.py` | Generates a prompt set for calibration. | ✅ Useful helper |
| `debug/make_calibration_data.py` | Builds calibration `.npz` data from prompts and diffusion inputs. | ✅ Useful helper |
| `debug/make_lightning_calibration.py` | Builds Lightning calibration data with correct `init_noise_sigma` scaling. | ⚠️ Important but closer to the dev pipeline |
| `debug/make_qnn_input_list_from_npz.py` | Converts calibration `.npz` into `input_list.txt` and `.raw` files for QNN tools. | ✅ Useful helper |
| `debug/make_qnn_extbias_input_list_from_npz.py` | Builds extended input lists for extbias/extmaps scenarios. | ⚠️ Specialized |

### Verification / debug / parity

| File | Purpose | Status |
| --- | --- | --- |
| `verify_clip_vae_onnx.py` | Compares PyTorch and ONNX for CLIP-L/CLIP-G/VAE. | ✅ Useful verification |
| `verify_e2e_onnx.py` | Runs an end-to-end ONNX sanity check without the phone runtime. | ✅ Useful verification |
| `debug/verify_vae_quick.py` | Quick VAE ONNX check. | ⚠️ Local technical check |
| `debug/compare_unet_pytorch_vs_onnx.py` | Compares PyTorch UNet output with ONNX UNet output. | ⚠️ Deep diagnostics |
| `debug/compare_onnx_vs_phone.py` | Compares ONNX results with phone-side results. | ⚠️ Deep diagnostics |
| `debug/batch_compare_onnx_vs_phone_saved_steps.py` | Batch-compares saved ONNX and phone step outputs. | ⚠️ Deep diagnostics |
| `debug/host_compare_unet_baselines.py` | Compares multiple host-side UNet baselines. | ⚠️ Research diagnostics |
| `debug/trace_unet_layer_parity.py` | Traces layer-by-layer UNet parity. | ⚠️ Advanced diagnostics |
| `debug/check_encoder_outputs.py` | Checks split-encoder outputs against a reference. | ⚠️ Internal technical check |
| `debug/generate_host_references.py` | Generates host-side reference data for step-by-step comparisons. | ⚠️ Research helper |
| `debug/generate_pc_reference.py` | Produces PC/GPU reference generations for control comparisons. | ⚠️ Research helper |
| `debug/generate_embed_cfg_references.py` | Produces references for embedding-space CFG experiments. | ⚠️ Research helper |
| `debug/measure_ram.py` | Measures runtime/phone-side memory usage. | ⚠️ Diagnostic helper |
| `debug/sdxl_speed_probe.py` | Runs end-to-end phone speed checks (and optional PC baseline) for the current runtime path. | ✅ Runtime diagnostic |
| `debug/sdxl_unet_overhead_probe.py` | Breaks down split-UNet overhead with `qnn-profile-viewer`, including `mmap`, batched CFG, and repeat-in-one-process cases. | ✅ Runtime diagnostic |

### Utility / rewrite / compatibility

| File | Purpose | Status |
| --- | --- | --- |
| `rewrite_onnx_instancenorm_to_groupnorm.py` | Rewrites `InstanceNorm` into `GroupNorm` for QNN compatibility. | ✅ Key utility |
| `rewrite_onnx_shape_reshape_to_static.py` | Makes shape/reshape behavior more static for QAIRT. | ✅ Utility |
| `rewrite_onnx_gemm_to_matmul.py` | Rewrites `Gemm` into `MatMul` as a workaround. | ✅ Utility |
| `rewrite_onnx_extmaps_bias_inputs_to_fp16.py` | Rewrites extmaps/extbias inputs to FP16 and removes unnecessary Cast nodes. | ✅ Utility |
| `qnn_onnx_converter_expanddims_patch.py` | Monkey-patch entrypoint for the QAIRT converter with required fixes. | ⚠️ Low-level workaround |
| `assess_generated_image.py` | Provides a quick no-reference quality assessment of the final image. | ✅ Useful utility |

### Experimental / alternative

| File | Purpose | Status |
| --- | --- | --- |
| `debug/test_distillation_loras.py` | Tests alternative distillation/LoRA scenarios. | ⚠️ Pure research |

## Nearby shell helpers

They are not Python, but they matter when reading the full story:

| File | Purpose |
| --- | --- |
| `debug/build_fp16_ctx.sh` | Older rooted helper for building context binaries on the phone. |
| `debug/run_ctxgen_lightning.sh` | Newer helper for phone-side `qnn-context-binary-generator`. |

## What belongs to the “final path” and what does not

### Closest to the public beta path

- `phone_generate.py`
- `scripts/deploy_to_phone.py`
- tokenizer + context binaries
- APK

### Not the final user-facing path, but active lab infrastructure

These files should **not** be treated as mandatory for an ordinary user:

- all `debug/compare_*`
- all `debug/generate_*reference*`
- `debug/trace_unet_layer_parity.py`
- `debug/check_encoder_outputs.py`
- `debug/measure_ram.py`
- `debug/test_distillation_loras.py`
- the AI Hub branch (`debug/export_and_compile_aihub.py`, `debug/export_split_unet.py`)

## Why there are so many of them — and why they are not junk

This collection grew out of real engineering work:

- some files exist because QAIRT/QNN has edge cases;
- some exist to verify parity across PyTorch, ONNX, and the phone;
- some exist because the split-UNet and Lightning/full-UNet branches evolved in parallel;
- some are simply honest R&D infrastructure, not “extra scripts”.

That is why the README now needs to clearly separate:

- **what a newcomer should run**;
- **what the author/researcher needs**;
- **what is still experimental**.
