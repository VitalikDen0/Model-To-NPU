# Live phone-side SDXL deployment example

This file documents a **real directory snapshot** collected on **2026-03-29** from a connected phone via `adb shell su -c ...`.

It is included as an example of what the toolchain has already produced on-device.  
Large binaries are **not** stored in the repository; this file is an inventory/reference only.

## Base path

`/data/local/tmp/sdxl_qnn`

## Top-level items observed

- `bin/`
- `context/`
- `lib/`
- `model/`
- `output/`
- `outputs/`
- `phone_gen/`
- `runtime_work_gen/`
- `python3` (symlink to Termux Python)
- helper scripts and QNN config JSON files

## Context binaries observed

```text
clip_l.serialized.bin.bin
clip_g.serialized.bin.bin
unet_encoder_fp16.serialized.bin.bin
unet_decoder_fp16.serialized.bin.bin
unet_lightning8step.serialized.bin.bin
vae_decoder.serialized.bin.bin
```

## Model libraries observed

```text
libclip_l.so
libclip_g.so
libunet_lightning_fp16.so
libunet_lightning8step.so
libvae_decoder.so
```

## Phone-side generator files observed

```text
/data/local/tmp/sdxl_qnn/phone_gen/generate.py
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/vocab.json
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/merges.txt
```

## Extra live artifacts currently present on device

The live phone snapshot also contains files that are useful for development/debugging but are **not part of the minimal layout**:

- `ctxgen_fp16.log`
- `ctxgen_fp16_exit.txt`
- `htp_backend_ext_config_*.json`
- `htp_backend_extensions_*.json`
- `runtime_work_gen/`
- helper shell scripts such as `tmp_gen_cfg35.sh`

## Why this file exists

- to show a concrete result already produced by the tooling;
- to help compare the documented minimal layout with a real phone deployment;
- to avoid committing multi-gigabyte binaries into git.
