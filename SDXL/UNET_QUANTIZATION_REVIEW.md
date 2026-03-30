# SDXL UNet quantization review

This note summarizes the **actual SDXL Lightning UNet structure used in this repository** and marks the areas that are safest vs riskiest for quantization experiments.

It is based on the current repo code, especially:

- `SDXL/export_sdxl_to_onnx.py`
- `SDXL/export_split_unet.py`
- `SDXL/quantize_unet.py`
- `SDXL/LESSONS_LEARNED.md`

## 1. The real UNet topology in this repo

The current exported SDXL UNet is configured as:

- `in_channels=4`, `out_channels=4`
- `down_block_types = [DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D]`
- `mid_block_type = UNetMidBlock2DCrossAttn`
- `up_block_types = [CrossAttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D]`
- `block_out_channels = [320, 640, 1280]`
- `layers_per_block = 2`
- `cross_attention_dim = 2048`
- `transformer_layers_per_block = [1, 2, 10]`
- `attention_head_dim = [5, 10, 20]`
- `addition_embed_type = text_time`
- `projection_class_embeddings_input_dim = 2816`

That means the model is not just "a stack of convs". It combines:

- spatial residual blocks;
- cross-attention blocks conditioned on text embeddings;
- time embedding and added text+time conditioning;
- skip connections that must preserve both scale and semantic detail.

## 2. Split runtime boundaries

The phone runtime does **not** execute the full monolithic UNet directly.
It uses a split FP16 path:

### Encoder half

`conv_in + time_embed + down_blocks + mid_block`

The split encoder emits:

- `mid_out`
- `skip_0..skip_8`
- `temb`

Documented skip shapes at 1024×1024 (latent 128×128):

- `skip_0..2`: `[1, 320, 128, 128]`
- `skip_3`: `[1, 320, 64, 64]`
- `skip_4..5`: `[1, 640, 64, 64]`
- `skip_6`: `[1, 640, 32, 32]`
- `skip_7..8`: `[1, 1280, 32, 32]`

### Decoder half

`up_blocks + conv_norm_out + conv_out`

This means the runtime boundary itself already exposes the most sensitive internal feature tensors:

- bottleneck / mid representation;
- all skip feature maps;
- time embedding (`temb`).

These are **not good places for aggressive numeric experiments** unless the entire downstream pipeline is validated end-to-end.

## 3. What is already treated as sensitive in the repo

`SDXL/quantize_unet.py` already excludes the following node families from coarse quantization:

- `conv_in`
- `conv_out`
- `conv_norm_out`
- `time_embed`
- `add_embedding`

This is a good baseline and should be treated as a minimum safety fence, not as over-caution.

## 4. What should stay closest to original precision

These parts are the highest-risk zones and should remain FP16 / original-form first, or be excluded from aggressive experiments.

### A. Input/output boundaries

- `conv_in`
- `conv_norm_out`
- `conv_out`

Why:

- `conv_in` is the first projection from latent space into UNet feature space;
- `conv_out` is the final projection back into `noise_pred`;
- visible image quality is very sensitive to range/scale damage at these boundaries.

### B. Time/addition embedding path

- `get_time_embed`
- `time_embedding`
- `add_embedding`
- text+time addition conditioning path
- `temb`-driven residual conditioning sites

Why:

- this path controls denoising behavior across timesteps;
- small scale errors here affect the entire network, not just one local block;
- the repo already documents failures around conditioning/export handling.

### C. Attention normalization / softmax-sensitive regions

Treat these as high risk for blanket overrides:

- LayerNorm-like normalization boundaries around attention
- Softmax-heavy attention subgraphs
- any mixed-precision convert islands around attention outputs

Why:

- the repo already documents that broad LayerNorm + Softmax FP16 override strategies can collapse converter behavior;
- even when host-side export looks acceptable, phone-side context generation may fail.

### D. Skip tensors and late decoder path

Be conservative around:

- skip feature maps passed from encoder to decoder;
- the final decoder stages close to `conv_out`;
- anything that strongly affects late denoising refinements.

Why:

- these tensors carry high-frequency structure;
- visible final image detail is especially sensitive near the end of denoising.

## 5. What is relatively safer to quantize first

These are the best candidates for **careful** quantization, especially weight quantization.

### A. Interior convolution weights away from boundaries

Safer targets:

- internal convs inside mid-depth down/up blocks;
- residual block weights that are not the very first/last boundary projections.

Reason:

- weight-only or weight-dominant compression in interior blocks usually hurts less than changing boundary tensors.

### B. Deeper interior weights in the 640/1280 channels region

Potentially reasonable targets:

- middle/deeper residual weights inside `down_blocks.1`, `down_blocks.2`, `mid_block`, `up_blocks.0`, `up_blocks.1`

Reason:

- these blocks are large, so they matter for size/perf;
- but they still need calibration-backed experiments and quality checks.

### C. Per-channel weight quantization for convolution weights

The repo already leans in this direction:

- `quantize_unet.py` defaults to `per_channel=True`
- `WeightSymmetric=True`
- calibration default is `percentile`

This is a much safer starting point than flat global symmetric/asymmetric guesses.

## 6. What should NOT be changed blindly

### Do not blindly quantize activations more aggressively everywhere

The most dangerous mistake is to treat activation quantization as interchangeable with weight quantization.

Why:

- activations carry prompt-dependent and timestep-dependent state;
- late-step activations are especially visually sensitive;
- the repo already shows that calibration correctness is crucial.

### Do not trust "converter succeeded" as proof of model correctness

The repo already documents several failure modes where:

- export succeeds;
- conversion succeeds;
- outputs still become numerically collapsed or visually wrong.

### Do not rely on blanket mixed-precision overrides

Large global override sets around:

- LayerNorm
- Softmax
- convert islands

have already shown failure patterns in this repo.

## 7. Current safest experiment ladder

If exploring beyond the current `W8A16` baseline, use this order:

1. **Keep runtime baseline as `W8A16`**
2. **Use calibration data only**
3. **Keep boundary nodes excluded**
4. **Use per-channel quantized weights**
5. **Try percentile calibration before broader alternatives**
6. **Only then evaluate `8W8A` / full INT8 variants**
7. **Never promote experimental INT8 directly to runtime default without visual and numerical checks**

## 8. Recommended rule of thumb

If a tensor or subgraph is one of the following, be conservative:

- first in the network;
- last in the network;
- directly tied to timestep/text conditioning;
- directly tied to attention normalization / softmax;
- passed across the split boundary;
- close to the final denoising refinement path.

If it is an interior weight-heavy residual path with good calibration coverage, it is a much better candidate for experimentation.

## 9. Bottom line

For this repository today:

- **boundary convs and conditioning paths should stay closest to original precision**;
- **attention-adjacent normalization/softmax regions should not be mass-overridden blindly**;
- **interior convolution weights are the least dangerous place to push quantization first**;
- **activation quantization is materially riskier than weight quantization**;
- the runtime default should remain the current safer baseline until any stronger experiment proves itself with calibration-backed validation.
