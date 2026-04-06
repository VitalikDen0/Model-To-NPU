#!/usr/bin/env python3
"""
Export SDXL Lightning UNet as two halves (encoder + decoder) for AI Hub compilation.
Each half fits within the 3.5 GB HTP allocation limit.

Encoder: conv_in + time_embed + down_blocks + mid_block → 1243M params (2.49GB FP16)
Decoder: up_blocks + conv_out → 1324M params (2.65GB FP16)

Usage:
  python NPU/export_split_unet.py --export         # Export both ONNX halves
  python NPU/export_split_unet.py --compile         # Submit both to AI Hub
  python NPU/export_split_unet.py --status          # Check job statuses
  python NPU/export_split_unet.py --download        # Download both context binaries
"""
import argparse, gc, os, sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PLATFORM_TOOLS = ROOT.parent
DEFAULT_SDXL_NPU = EXTERNAL_PLATFORM_TOOLS / "sdxl_npu"
SDXL_NPU = DEFAULT_SDXL_NPU if DEFAULT_SDXL_NPU.exists() else (ROOT / "sdxl_npu")
MERGED_UNET_DIR = SDXL_NPU / "unet_lightning8step_merged"
ONNX_DIR = ROOT / "NPU" / "onnx_lightning_split"
ONNX_ENC = ONNX_DIR / "unet_encoder.onnx"    # AI Hub package dir
ONNX_DEC = ONNX_DIR / "unet_decoder.onnx"    # AI Hub package dir
CONTEXT_OUT = ROOT / "NPU" / "aihub_context"
JOB_ENC_FILE = ROOT / "NPU" / "aihub_job_encoder.txt"
JOB_DEC_FILE = ROOT / "NPU" / "aihub_job_decoder.txt"

DEVICE_NAME = "Snapdragon 8 Elite QRD"

# Skip connection shapes at 1024x1024 (latent 128x128)
# skip_0..2: [1, 320, 128, 128]
# skip_3:    [1, 320, 64, 64]
# skip_4..5: [1, 640, 64, 64]
# skip_6:    [1, 640, 32, 32]
# skip_7..8: [1, 1280, 32, 32]
SKIP_SHAPES = [
    (1, 320, 128, 128),  # skip_0 (conv_in)
    (1, 320, 128, 128),  # skip_1 (down0 res0)
    (1, 320, 128, 128),  # skip_2 (down0 res1)
    (1, 320, 64, 64),    # skip_3 (down0 downsample)
    (1, 640, 64, 64),    # skip_4 (down1 res0)
    (1, 640, 64, 64),    # skip_5 (down1 res1)
    (1, 640, 32, 32),    # skip_6 (down1 downsample)
    (1, 1280, 32, 32),   # skip_7 (down2 res0)
    (1, 1280, 32, 32),   # skip_8 (down2 res1)
]


class SDXLUNetEncoder(nn.Module):
    """Part 1: time_embed + conv_in + down_blocks + mid_block.
    
    Inputs:  sample, timestep, encoder_hidden_states, text_embeds, time_ids
    Outputs: mid_out, skip_0..skip_8, temb  (11 tensors)
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        # 1. Time embedding
        t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        emb = self.unet.time_embedding(t_emb)
        aug_emb = self.unet.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        )
        emb = emb + aug_emb

        encoder_hidden_states = self.unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        )

        # 2. conv_in
        sample = self.unet.conv_in(sample)

        # 3. down blocks
        down_block_res_samples = (sample,)
        for block in self.unet.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res = block(hidden_states=sample, temb=emb)
            down_block_res_samples += res

        # 4. mid block
        if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
            sample = self.unet.mid_block(
                sample, emb, encoder_hidden_states=encoder_hidden_states
            )
        else:
            sample = self.unet.mid_block(sample, emb)

        # Return: mid_output, 9 skip connections, temb
        return (sample,) + down_block_res_samples + (emb,)


class SDXLUNetDecoder(nn.Module):
    """Part 2: up_blocks + conv_norm_out + conv_out.
    
    Inputs:  mid_out, skip_0..skip_8, temb, encoder_hidden_states  (12 tensors)
    Outputs: noise_pred
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, mid_out, skip_0, skip_1, skip_2, skip_3,
                skip_4, skip_5, skip_6, skip_7, skip_8,
                temb, encoder_hidden_states):
        sample = mid_out
        down_block_res_samples = (skip_0, skip_1, skip_2, skip_3,
                                  skip_4, skip_5, skip_6, skip_7, skip_8)

        for i, block in enumerate(self.unet.up_blocks):
            is_final_block = i == len(self.unet.up_blocks) - 1

            n_resnets = len(block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(
                    hidden_states=sample, temb=temb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = block(
                    hidden_states=sample, temb=temb,
                    res_hidden_states_tuple=res_samples,
                )

        # 6. post-process
        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        return sample


def export_onnx():
    """Export encoder and decoder halves as separate ONNX models."""
    from diffusers import UNet2DConditionModel

    print("[export] Loading merged Lightning UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
    )
    unet.eval()

    # === Encoder ===
    print("\n[export] === Encoder (conv_in + down_blocks + mid_block) ===")
    encoder = SDXLUNetEncoder(unet)
    encoder.eval()

    sample = torch.randn(1, 4, 128, 128, dtype=torch.float32)
    timestep = torch.tensor([999.0], dtype=torch.float32)
    enc_states = torch.randn(1, 77, 2048, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, dtype=torch.float32)
    time_ids = torch.randn(1, 6, dtype=torch.float32)

    # Verify encoder runs
    with torch.no_grad():
        enc_out = encoder(sample, timestep, enc_states, text_embeds, time_ids)
    print(f"  Encoder outputs: {len(enc_out)} tensors")
    print(f"  mid_out: {enc_out[0].shape}")
    for i in range(9):
        print(f"  skip_{i}: {enc_out[1+i].shape}")
    print(f"  temb: {enc_out[10].shape}")

    ONNX_ENC.mkdir(parents=True, exist_ok=True)
    enc_path = str(ONNX_ENC / "model.onnx")
    print(f"[export] Exporting encoder to {enc_path}...")

    output_names = ["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb"]

    torch.onnx.export(
        encoder,
        (sample, timestep, enc_states, text_embeds, time_ids),
        enc_path,
        opset_version=18,
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=output_names,
        dynamic_axes=None,
    )

    # Re-save with external data
    import onnx
    model = onnx.load(enc_path)
    onnx.save(model, enc_path, save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.data")
    del model

    # Remove duplicate .data file if exists
    dup = ONNX_ENC / "model.onnx.data"
    if dup.exists():
        dup.unlink()
        print("  Removed duplicate model.onnx.data")

    enc_onnx_size = os.path.getsize(enc_path)
    enc_data_size = os.path.getsize(str(ONNX_ENC / "model.data"))
    print(f"  model.onnx: {enc_onnx_size/1e6:.1f} MB, model.data: {enc_data_size/1e9:.2f} GB")

    del encoder, enc_out
    gc.collect()

    # === Decoder ===
    print("\n[export] === Decoder (up_blocks + conv_out) ===")
    decoder = SDXLUNetDecoder(unet)
    decoder.eval()

    mid_out = torch.randn(1, 1280, 32, 32, dtype=torch.float32)
    skips = [torch.randn(*s, dtype=torch.float32) for s in SKIP_SHAPES]
    temb = torch.randn(1, 1280, dtype=torch.float32)

    with torch.no_grad():
        dec_out = decoder(mid_out, *skips, temb, enc_states)
    print(f"  Decoder output: {dec_out.shape}")

    ONNX_DEC.mkdir(parents=True, exist_ok=True)
    dec_path = str(ONNX_DEC / "model.onnx")
    print(f"[export] Exporting decoder to {dec_path}...")

    dec_input_names = ["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb", "encoder_hidden_states"]

    torch.onnx.export(
        decoder,
        (mid_out, *skips, temb, enc_states),
        dec_path,
        opset_version=18,
        input_names=dec_input_names,
        output_names=["noise_pred"],
        dynamic_axes=None,
    )

    model = onnx.load(dec_path)
    onnx.save(model, dec_path, save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.data")
    del model

    dup = ONNX_DEC / "model.onnx.data"
    if dup.exists():
        dup.unlink()
        print("  Removed duplicate model.onnx.data")

    dec_onnx_size = os.path.getsize(dec_path)
    dec_data_size = os.path.getsize(str(ONNX_DEC / "model.data"))
    print(f"  model.onnx: {dec_onnx_size/1e6:.1f} MB, model.data: {dec_data_size/1e9:.2f} GB")

    del decoder, dec_out, unet
    gc.collect()
    print("\n[export] Both halves exported successfully!")


def submit_compile(part="both"):
    """Submit encoder and/or decoder to AI Hub."""
    import qai_hub as hub

    compile_options = (
        "--target_runtime qnn_context_binary "
        "--qnn_options default_graph_htp_precision=FLOAT16"
    )

    def _submit(name, onnx_pkg, job_file):
        print(f"\n[compile] Submitting {name}...")
        print(f"  Model: {onnx_pkg}")
        print(f"  Options: {compile_options}")
        job = hub.submit_compile_job(
            model=str(onnx_pkg),
            device=hub.Device(DEVICE_NAME),
            name=f"SDXL_Lightning_{name}",
            options=compile_options,
        )
        job_id = job.job_id
        print(f"  Job ID: {job_id}")
        print(f"  URL: https://app.aihub.qualcomm.com/jobs/{job_id}/")
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(job_id)
        return job

    if part in ("both", "encoder"):
        _submit("Encoder", ONNX_ENC, JOB_ENC_FILE)
    if part in ("both", "decoder"):
        _submit("Decoder", ONNX_DEC, JOB_DEC_FILE)


def check_status():
    """Check status of both compile jobs."""
    import qai_hub as hub

    for label, job_file in [("Encoder", JOB_ENC_FILE), ("Decoder", JOB_DEC_FILE)]:
        if not job_file.exists():
            print(f"[status] {label}: no job ID found")
            continue
        job_id = job_file.read_text().strip()
        job = hub.get_job(job_id)
        s = job.get_status()
        print(f"[status] {label} ({job_id}): {s.code}")
        if s.message:
            print(f"  Message: {s.message}")


def download_contexts():
    """Download both compiled context binaries."""
    import qai_hub as hub

    CONTEXT_OUT.mkdir(parents=True, exist_ok=True)

    for label, job_file, fname in [
        ("Encoder", JOB_ENC_FILE, "unet_encoder_fp16.bin"),
        ("Decoder", JOB_DEC_FILE, "unet_decoder_fp16.bin"),
    ]:
        if not job_file.exists():
            print(f"[download] {label}: no job ID found")
            continue

        job_id = job_file.read_text().strip()
        job = hub.get_job(job_id)
        s = job.get_status()

        if not s.success:
            print(f"[download] {label} ({job_id}): not complete ({s.code})")
            continue

        out_path = str(CONTEXT_OUT / fname)
        print(f"[download] {label}: downloading to {out_path}...")
        job.download_target_model(out_path)
        size = os.path.getsize(out_path)
        print(f"  Done! Size: {size / 1e9:.2f} GB")

    print("\n[download] Deploy to phone:")
    print(f"  adb push NPU/aihub_context/unet_encoder_fp16.bin /data/local/tmp/sdxl_qnn/context/unet_encoder_fp16.serialized.bin.bin")
    print(f"  adb push NPU/aihub_context/unet_decoder_fp16.bin /data/local/tmp/sdxl_qnn/context/unet_decoder_fp16.serialized.bin.bin")


def main():
    ap = argparse.ArgumentParser(description="Export split Lightning UNet → AI Hub")
    ap.add_argument("--export", action="store_true", help="Export both ONNX halves")
    ap.add_argument("--compile", action="store_true", help="Submit both to AI Hub")
    ap.add_argument("--compile-encoder", action="store_true")
    ap.add_argument("--compile-decoder", action="store_true")
    ap.add_argument("--status", action="store_true", help="Check job statuses")
    ap.add_argument("--download", action="store_true", help="Download contexts")
    a = ap.parse_args()

    if not any([a.export, a.compile, a.compile_encoder, a.compile_decoder, a.status, a.download]):
        ap.print_help()
        return

    if a.export:
        export_onnx()
    if a.compile:
        submit_compile("both")
    if a.compile_encoder:
        submit_compile("encoder")
    if a.compile_decoder:
        submit_compile("decoder")
    if a.status:
        check_status()
    if a.download:
        download_contexts()


if __name__ == "__main__":
    main()
