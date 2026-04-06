#!/usr/bin/env python3
"""
Export standard (non-extmaps) Lightning UNet ONNX → upload to Qualcomm AI Hub
→ compile to QNN context binary for Snapdragon 8 Elite.

Usage:
  python NPU/export_and_compile_aihub.py --export   # Step 1: export ONNX
  python NPU/export_and_compile_aihub.py --compile   # Step 2: submit to AI Hub
  python NPU/export_and_compile_aihub.py --status     # Check job status
  python NPU/export_and_compile_aihub.py --download   # Download compiled context
  python NPU/export_and_compile_aihub.py --all        # Do everything
"""
import argparse, gc, json, os, sys, types, math, time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SDXL_NPU = ROOT / "sdxl_npu"
MERGED_UNET_DIR = SDXL_NPU / "unet_lightning8step_merged"
ONNX_DIR = ROOT / "NPU" / "onnx_lightning_standard"
ONNX_PACKAGE = ONNX_DIR / "unet_lightning.onnx"  # directory with .onnx extension for AI Hub
CONTEXT_OUT = ROOT / "NPU" / "aihub_context"
JOB_ID_FILE = ROOT / "NPU" / "aihub_compile_job_id.txt"

DEVICE_NAME = "Snapdragon 8 Elite QRD"


def export_onnx():
    """Export standard Lightning UNet to ONNX with external weights."""
    from diffusers import UNet2DConditionModel

    print("[export] Loading merged Lightning UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
    )
    unet.eval()

    # Standard SDXL UNet wrapper — no extmaps, clean inputs
    class SDXLUNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    wrapper = SDXLUNetWrapper(unet)
    wrapper.eval()

    # Export in FP32 — AI Hub handles FP16 via default_graph_htp_precision=FLOAT16
    sample = torch.randn(1, 4, 128, 128, dtype=torch.float32)
    timestep = torch.tensor([999.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, dtype=torch.float32)
    time_ids = torch.randn(1, 6, dtype=torch.float32)

    # Create output directory structure for AI Hub
    # AI Hub expects: <dir>.onnx/ containing model.onnx + model.data
    ONNX_PACKAGE.mkdir(parents=True, exist_ok=True)
    onnx_path = str(ONNX_PACKAGE / "model.onnx")

    print(f"[export] Exporting to {onnx_path}...")
    print("[export] This may take a few minutes for the full UNet...")

    torch.onnx.export(
        wrapper,
        (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
        onnx_path,
        opset_version=17,
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["noise_pred"],
        dynamic_axes=None,  # Fixed shape for NPU
    )

    # ONNX file will be huge (>2GB). Need to save with external data.
    import onnx
    model = onnx.load(onnx_path)
    onnx.save(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
    )

    onnx_size = os.path.getsize(onnx_path)
    data_path = str(ONNX_PACKAGE / "model.data")
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f"[export] Done!")
    print(f"  model.onnx: {onnx_size / 1e6:.1f} MB")
    print(f"  model.data: {data_size / 1e9:.2f} GB")
    print(f"  Package dir: {ONNX_PACKAGE}")

    del model, wrapper, unet
    gc.collect()


def submit_compile():
    """Submit ONNX to AI Hub for compilation to QNN context binary."""
    import qai_hub as hub

    if not ONNX_PACKAGE.exists():
        print("[compile] ERROR: ONNX package not found. Run --export first.")
        return

    print(f"[compile] Target device: {DEVICE_NAME}")
    print(f"[compile] Model: {ONNX_PACKAGE}")
    print("[compile] Uploading model to AI Hub (this may take a while for 5 GB)...")

    compile_options = (
        "--target_runtime qnn_context_binary "
        "--qnn_options default_graph_htp_precision=FLOAT16"
    )
    print(f"[compile] Options: {compile_options}")

    job = hub.submit_compile_job(
        model=str(ONNX_PACKAGE),
        device=hub.Device(DEVICE_NAME),
        name="SDXL_Lightning_UNet_FP16",
        options=compile_options,
    )

    job_id = job.job_id
    print(f"[compile] Job submitted! ID: {job_id}")
    print(f"[compile] URL: https://app.aihub.qualcomm.com/jobs/{job_id}/")

    # Save job ID for later
    JOB_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_ID_FILE.write_text(job_id)
    print(f"[compile] Job ID saved to {JOB_ID_FILE}")

    return job


def check_status():
    """Check compile job status."""
    import qai_hub as hub

    if not JOB_ID_FILE.exists():
        print("[status] No job ID found. Run --compile first.")
        return None

    job_id = JOB_ID_FILE.read_text().strip()
    print(f"[status] Checking job {job_id}...")
    job = hub.get_job(job_id)
    status = job.get_status()
    print(f"[status] Status: {status}")
    return job


def download_context():
    """Download the compiled context binary."""
    import qai_hub as hub

    if not JOB_ID_FILE.exists():
        print("[download] No job ID found. Run --compile first.")
        return

    job_id = JOB_ID_FILE.read_text().strip()
    job = hub.get_job(job_id)
    status = job.get_status()

    if not status.success:
        print(f"[download] Job not complete. Status: {status}")
        return

    CONTEXT_OUT.mkdir(parents=True, exist_ok=True)
    out_path = str(CONTEXT_OUT / "unet_lightning_fp16_aihub.bin")
    print(f"[download] Downloading context binary to {out_path}...")
    job.download_target_model(out_path)
    size = os.path.getsize(out_path)
    print(f"[download] Done! Size: {size / 1e9:.2f} GB")
    print(f"[download] Next: push to phone with:")
    print(f"  adb push \"{out_path}\" /data/local/tmp/sdxl_qnn/context/unet_lightning_fp16.serialized.bin.bin")


def main():
    ap = argparse.ArgumentParser(description="Export Lightning UNet → AI Hub → QNN context")
    ap.add_argument("--export", action="store_true", help="Export standard ONNX")
    ap.add_argument("--compile", action="store_true", help="Submit compile job to AI Hub")
    ap.add_argument("--status", action="store_true", help="Check job status")
    ap.add_argument("--download", action="store_true", help="Download compiled context")
    ap.add_argument("--all", action="store_true", help="Export + compile")
    a = ap.parse_args()

    if not any([a.export, a.compile, a.status, a.download, a.all]):
        ap.print_help()
        return

    if a.export or a.all:
        export_onnx()

    if a.compile or a.all:
        submit_compile()

    if a.status:
        check_status()

    if a.download:
        download_context()


if __name__ == "__main__":
    main()
