# SDXL on Qualcomm NPU — Lessons Learned & Pitfalls

Accumulated knowledge from building a native SDXL inference pipeline
on Snapdragon 8 Elite (SM8750) Hexagon NPU. Read this before making changes!

---

## 1. ONNX Export Pitfalls

### CLIP: `hidden_states[-2]`, NOT `last_hidden_state`!

**The #1 bug that cost the most time.**

SDXL pipeline uses **penultimate layer** hidden states from both CLIPs:
```python
out = text_encoder(ids, output_hidden_states=True)
hidden = out.hidden_states[-2]   # ← THIS is what SDXL uses
# NOT out.last_hidden_state       # ← This is NaN for CLIP-L!
```

- **CLIP-L** has 13 layers (0-12). Layer 12 (last) produces **NaN** even in FP32!
  Layers 0-11 are fine. Pipeline uses layer 11 (`hidden_states[-2]`).
- **CLIP-G** has 33 layers (0-32). All layers are fine.
  Pipeline uses layer 31 (`hidden_states[-2]`).
- **Pooled output**: only from CLIP-G: `out.text_embeds` (= `out[0]`) → `[1, 1280]`
- **Final prompt_embeds**: `concat(clip_l_hidden[-2], clip_g_hidden[-2], dim=-1)` → `[1, 77, 2048]`

**Fix**: Export wrapper classes that call model with `output_hidden_states=True`
and return `hidden_states[-2]`. See `CLIPLWrapper` / `CLIPGWrapper` in `export_clip_vae_to_onnx.py`.

### CLIP-G: Protobuf >2GB limit

CLIP-G with all weights is >2GB. Two implications:
1. **ONNX save**: Must use `save_as_external_data=True` with `all_tensors_to_one_file=True`
2. **ONNX checker**: Must call `onnx.checker.check_model(path_str)` not `check_model(model_object)`
3. **ONNX load for inspection**: Use `onnx.load(path, load_external_data=False)` for metadata only

### int64 for input_ids — NOT float64

`input_ids` are `int64` token indices (range 0-49407). This is integers, not float64.
QNN converter handles int64 natively. **Do NOT** convert int64→int32 — it breaks
Reshape/Shape constants that are also stored as int64 internally in ONNX.

**HOWEVER**: at runtime, `qnn-net-run` reads all inputs as raw bytes, and QNN internally
narrows int64→int32 (308 bytes = 77×4). You must feed token IDs as **float32 values**
(e.g., `np.array([49406, 272, ...]).astype(np.float32)`) — not as int32 bit patterns.
If you feed int32 byte representations, the model reads them as float32 garbage → wrong embeddings.

### FP16 → FP32 → FP16 roundtrip is lossless

All source weights are **FP16** (including VAE — it's FP16, NOT BF16 as previously assumed).
Exporting through FP32 ONNX is fine: FP16 ⊂ FP32, so no precision loss.

### VAE: InstanceNorm → GroupNorm rewrite

QNN doesn't support InstanceNorm. The VAE uses InstanceNorm which must be rewritten
to GroupNorm (opset 18) before QNN conversion. Script: `rewrite_onnx_instancenorm_to_groupnorm.py`.

---

## 2. QNN Conversion Pitfalls

### QAIRT 2.31 monkey-patches

The QAIRT 2.31 converter has multiple bugs that require monkey-patches.
All patches are in `qnn_onnx_converter_expanddims_patch.py`:

1. **ExpandDims**: ONNX Unsqueeze/Squeeze mapped to ExpandDims/Squeeze IR ops
2. **Elementwise**: Mixed float16/float32 binary elementwise inputs alignment
3. **GroupNorm**: Output dtype fix (was forcing float32)
4. **Reshape**: dtype preservation for all reshape ops
5. **LayerNorm**: dtype preservation
6. **Transpose**: dtype preservation
7. **StridedSlice**: dtype preservation
8. **ElementwiseNeuron**: dtype preservation
9. **MatMul**: dtype preservation
10. **Softmax**: dtype preservation
11. **Convolution**: dtype preservation
12. **Concat**: dtype preservation
13. **Resize**: dtype preservation

### QNN converter saves extensionless C++ files

When using `--output_path dir/model`, the converter saves `dir/model` (no .cpp extension)
and `dir/model.bin`. The Android `ndk-build` / clang++ needs `.cpp` extension to compile.
`build_android_model_lib_windows.py` handles this automatically by copying.

### Calibration data MUST be float32

QNN converter's `--input_list` always reads data as **float32**, regardless of source dtype.
If you feed float16 `.raw` files, the converter reads garbage → broken quantization.
Always save calibration in float32.

### 3D tensor NCF → NFC transpose for cross-attention

QNN converter transposes 3D tensors `[1,77,2048]` → `[1,2048,77]` (NCF→NFC layout).
For `encoder_hidden_states` (the cross-attention input to UNet), you must transpose
the CLIP embeddings accordingly before feeding to UNet on NPU, otherwise cross-attention
gets garbage → gray monochrome images with correct dynamic range but no prompt adherence.

---

## 3. Phone-side Pitfalls

### ADSP_LIBRARY_PATH is mandatory

Without it, `qnn-context-binary-generator` and `qnn-net-run` fail with
"Device Creation failure". Must include:
```bash
export ADSP_LIBRARY_PATH='/sdcard/Download/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'
```

### `qnn-net-run --model` hangs on full UNet

Direct `qnn-net-run --model libunet.so` (without context binary) hangs on
`Composing Graphs → Finalizing Graphs` for **all** full UNet models, even known-good ones.
Always use `--retrieve_context` with pre-built context binary instead.

### Context binary generation and phone safety

Full UNet context binary generation is computationally heavy on phone.
Use gentle HTP backend extensions to avoid phone crashes:
```json
{
  "vtcm_mb": 4,
  "hvx_threads": 1,
  "perf_profile": "low_power_saver",
  "rpc_control_latency": 1000
}
```

### `graph_names` for folderized converter output

Folderized QNN converter outputs use internal graph name `model`,
not the long try-name from the output directory. Backend config must use
`"graph_names": ["model"]`, otherwise `composeGraphs()` fails.

### HTP output contract

For INT8 quantized UNet: output `noise_pred.raw` from `qnn-net-run --retrieve_context`
is **float32** (dequantized). For `--use_native_output_files`, output is **uint8** with
sidecar JSON containing scale/offset for manual dequantization.

Spatial tensors must be fed in **NHWC** layout to QNN, not NCHW.

---

## 4. SDXL Lightning Specifics

### No CFG (guidance_scale = 0)

SDXL-Lightning is a distilled model — it does NOT use classifier-free guidance.
Single UNet pass per step, no uncond/cond pair needed. `guidance_scale=0` or `1`.

### Why CFG still hurts wall-clock time so much

In this repository, CFG is expensive for two reasons at once:

1. classifier-free guidance needs both **cond** and **uncond** predictions;
2. the phone runtime uses a **split UNet** (`encoder` + `decoder`).

That means a naive CFG step is not just “run UNet twice” in the abstract — it can become:

- uncond encoder
- uncond decoder
- cond encoder
- cond decoder

So one denoising step can expand into **four QNN subprocess launches** plus extra file I/O.
The current phone runtime now batches cond/uncond work more efficiently, which reduces subprocess overhead, but the real NPU workload is still roughly doubled versus the no-CFG path.

### Scheduler: EulerDiscrete, trailing

Must use `EulerDiscreteScheduler` with `timestep_spacing="trailing"`.
Standard "leading" spacing produces wrong timesteps for Lightning.

### LoRA merge required

Lightning is distributed as a LoRA. Must be merged into base UNet weights
before ONNX export (can't do LoRA inference on NPU). See `bake_lora_into_unet.py`.

---

## 5. Mixed-Precision Quantization Findings

### All-LayerNorm + Softmax FP16 causes converter collapse

When overriding **all** LayerNorm + **all** Softmax outputs to FP16 simultaneously
in the full UNet, the QAIRT converter collapses: output scale drops to ~0.005
(from normal ~0.04) and all outputs become near-zero.

Any subset works fine — the issue is specifically at the exact full global boundary.
Even removing one small stage bucket prevents collapse.

### Override names must match ONNX source names

QAIRT quantization overrides must use **source ONNX tensor names**
(e.g., `/unet/down_blocks.1/.../Softmax_output_0`), NOT sanitized IR names
from `model_net.json` (e.g., `_unet_down_blocks_1_..._Softmax_output_0`).
Wrong names → `Processed 0 quantization encodings` → overrides silently ignored,
while simplification is already disabled.

### Mixed-precision Convert islands break phone ctxgen

Even a `Convert=24` profile (single attention block with FP16 overrides)
fails phone-side `qnn-context-binary-generator` with `EXIT:124`.
The working baseline `try22` has `Convert=0`. Any non-zero `Convert` count
correlates with phone-side ctxgen failure for full UNet.

---

## 6. Weight Dtype Summary

| Component | Safetensors dtype | Notes |
|-----------|------------------|-------|
| CLIP-L | `torch.float16` | 196 tensors |
| CLIP-G | `torch.float16` | 517 tensors |
| UNet | `torch.float16` | After Lightning LoRA merge |
| VAE | `torch.float16` | **NOT BF16** as previously documented! |

---

## 7. Verified Numerical Parity (2026-03-28)

All ONNX exports verified against PyTorch FP32 originals:

| Component | Output | cos | mae | max_abs |
|-----------|--------|-----|-----|---------|
| CLIP-L | penultimate_hidden [1,77,768] | 1.0000000000 | 2.89e-07 | 1.07e-04 |
| CLIP-G | penultimate_hidden [1,77,1280] | 1.0000000000 | 4.83e-06 | 1.10e-04 |
| CLIP-G | text_embeds [1,1280] | 1.0000000000 | 1.07e-06 | 4.48e-06 |
| prompt_embeds | concat [1,77,2048] | 1.0000000000 | 3.12e-06 | 1.10e-04 |
| VAE | image [1,3,1024,1024] | 0.9999998060 | 6.44e-05 | 5.17e-03 |

---

## 8. Performance Benchmarks

| Operation | Time | Device |
|-----------|------|--------|
| UNet 1 step (NPU, W8A16) | ~7.45s | OnePlus 13 HTP |
| UNet 8 steps total | ~59.6s | OnePlus 13 HTP |
| Context binary gen (phone) | ~5-10 min | OnePlus 13 HTP |
| QNN conversion (host) | ~1-2 min | Windows PC |
| Android .so build (host) | ~30s | Windows PC |

---

## 9. Runtime overhead findings (2026-03-30)

Deep probing of the split UNet path showed that the remaining pain is not just accelerator math.

- single encoder wall time was measured at **8312.0 ms**;
- single decoder wall time was measured at **8141.1 ms**;
- adding `--use_mmap` reduced those to **6115.0 ms** and **6011.4 ms** respectively (about **26%** gain on both halves);
- repeat×4 in one process collapsed average cost to **3527.5 ms** (encoder) and **4379.7 ms** (decoder), which strongly points at process/context lifecycle overhead.

Practical consequence:

- `v0.1.3` now enables `mmap` by default in the phone runtime and the APK launch path;
- the next major speed win is likely a persistent encoder/decoder runner rather than more quantization experiments.

---

## 10. TAESD QNN preview root cause and fix (2026-04-01)

The broken TAESD preview turned out to be a layered problem.

### Part 1: preview save path was too naive

`phone_generate.py` was always applying `img / 2 + 0.5` before saving the preview PNG.
That is correct only when the incoming tensor is in roughly `[-1, 1]`.

The phone-side QNN TAESD path had cases where the output already arrived in `[0, 1]`, so the save path could wash the image out into roughly `[0.5, 1.0]`.

### Part 2: the bigger real bug was stale / bad TAESD QNN artifacts

Even after fixing save-path normalization, the deployed QNN TAESD preview was still far from the ONNX reference.

What exposed the real issue:

- ONNX TAESD vs final VAE image: about **0.78** correlation on the same latent;
- old QNN TAESD vs ONNX TAESD: only about **0.21** correlation;
- old QNN TAESD raw output was clipped to **`[0, 1]`** instead of the expected roughly **`[-1.18, 1.23]`** range.

### What actually fixed it

The reliable fix was:

1. rebuild `libTAESDDecoder.so` from the current ONNX;
2. generate the TAESD GPU context **on the phone itself** with `qnn-context-binary-generator`;
3. deploy that rebuilt `lib + context` pair together.

After the rebuild:

- rebuilt QNN TAESD vs ONNX TAESD jumped to about **0.9999** correlation;
- rebuilt QNN TAESD vs final image on the same latent reached about **0.91** correlation;
- real 8-step phone preview-vs-final similarity rose to about **0.85**.

### Practical deployment lesson

Do **not** assume that an old TAESD `lib/context` pair is still valid just because the preview backend starts and prints timings.
If the preview looks wrong, compare the QNN output directly against the ONNX TAESD reference on the same latent.

It is very possible for the preview to be *fast* and still be *numerically wrong*.
