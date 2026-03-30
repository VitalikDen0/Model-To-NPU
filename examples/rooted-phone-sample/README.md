# Rooted SDXL phone sample (reference artifact set)

This folder contains a **small real-world sample set** copied from a rooted phone deployment under:

`/data/local/tmp/sdxl_qnn`

It is meant for **reference, exploration, and educational use** when someone wants to inspect a working layout before reproducing the full pipeline on their own device.

## What is included

This sample intentionally includes only **lightweight and illustrative** files:

- selected generated PNG outputs;
- the current tiny TAESD preview model: `phone_gen/taesd_decoder.onnx`;
- HTP backend JSON configs;
- helper shell scripts used during rooted experiments;
- `phone_gen/generate.py` from the rooted phone-side runtime;
- `phone_gen/tokenizer/` files as they existed on the device;
- small logs such as `ctxgen_fp16.log`.

## What is intentionally NOT included

This folder does **not** redistribute the heavy or sensitive runtime payload:

- the main context binaries from `context/` (`clip_*`, `unet_*`, `vae_*`);
- QNN runtime libraries from `lib/`;
- compiled model libraries from `model/`;
- any complete proprietary SDK packages.

Those files are either too large, too environment-specific, or not appropriate to redistribute here.

Why they are missing: GitHub does not allow multi-hundred-megabyte / multi-gigabyte model payloads in a normal repository. The split UNet/CLIP contexts alone exceed the per-file limit by a wide margin, so this sample keeps only the lightweight pieces that are practical to version.

The current live-preview exception is the tiny TAESD ONNX decoder, which is small enough to keep as a reference artifact because it powers APK `v0.1.2` live previews. A legacy `context/taesd_decoder.serialized.bin.bin` may still appear in older rooted snapshots, but the current runtime no longer needs it.

## Source model reference

The rooted sample was produced during experiments around the following public model combination:

- base checkpoint: [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310)
- LoRA: [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning)

## Rights and disclaimer

This repository does **not** claim ownership of the original model weights or of the upstream model identity.

- all rights to the original model/checkpoint belong to its respective creator(s) and publisher(s);
- all rights to SDXL-Lightning belong to its respective creator(s) and publisher(s);
- this folder is provided strictly as a **reference/educational example** of a rooted phone-side deployment layout and its helper artifacts;
- if you want the original model itself, please obtain it from its official upstream source.

## Related files

- English rooted layout overview: [`../phone-sdxl-qnn-layout.md`](../phone-sdxl-qnn-layout.md)
- Russian rooted layout overview: [`../phone-sdxl-qnn-layout_RU.md`](../phone-sdxl-qnn-layout_RU.md)
- Russian version of this folder note: [`README_RU.md`](README_RU.md)
