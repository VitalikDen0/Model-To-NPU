#!/usr/bin/env python3
"""Quick check of split encoder outputs vs PyTorch reference."""
import numpy as np
import sys, os, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdxl_npu"))
sys.path.insert(0, str(ROOT))

WORK = ROOT / "NPU" / "runtime_work_gen"
MERGED_UNET_DIR = ROOT / "sdxl_npu" / "unet_lightning8step_merged"
DIFFUSERS_DIR = ROOT / "sdxl_npu" / "diffusers_pipeline"

# Load phone encoder outputs
phone_dir = WORK / "cond_s00" / "out_enc" / "Result_0"
phone_outputs = {}
for i in range(11):
    f = phone_dir / f"output_{i}.raw"
    if f.exists():
        d = np.fromfile(str(f), np.float32)
        phone_outputs[i] = d
        print(f"output_{i}: {d.size} elems, range=[{d.min():.4f}..{d.max():.4f}], std={d.std():.4f}")

# Now run PyTorch encoder with same inputs
print("\n--- Running PyTorch reference ---")
from NPU.export_split_unet import SDXLUNetEncoder, SKIP_SHAPES
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTokenizer
import json

unet = UNet2DConditionModel.from_pretrained(
    str(MERGED_UNET_DIR), torch_dtype=torch.float32, local_files_only=True
).eval()

encoder = SDXLUNetEncoder(unet).eval()

# Reconstruct the exact same inputs used by the phone pipeline
# enc_cond.raw = encoder_hidden_states [1,77,2048] float32
enc_hs = np.fromfile(str(WORK / "enc_cond.raw"), np.float32).reshape(1, 77, 2048)
# te_cond.raw = text_embeds [1,1280] float32  
te = np.fromfile(str(WORK / "te_cond.raw"), np.float32).reshape(1, 1280)
# tid_cond.raw = time_ids [1,6] float32
tid = np.fromfile(str(WORK / "tid_cond.raw"), np.float32).reshape(1, 6)
# smp.raw = sample [1,4,128,128] float32
smp = np.fromfile(str(WORK / "cond_s00" / "smp.raw"), np.float32).reshape(1, 4, 128, 128)
# ts.raw = timestep [1] float32
ts_val = np.fromfile(str(WORK / "cond_s00" / "ts.raw"), np.float32)

print(f"enc_hs: {enc_hs.shape}, range=[{enc_hs.min():.4f}..{enc_hs.max():.4f}]")
print(f"te: {te.shape}, range=[{te.min():.4f}..{te.max():.4f}]")
print(f"tid: {tid.shape}, values={tid.flatten()}")
print(f"smp: {smp.shape}, range=[{smp.min():.4f}..{smp.max():.4f}]")
print(f"timestep: {ts_val}")

with torch.no_grad():
    out = encoder(
        torch.from_numpy(smp),
        torch.tensor(ts_val),
        torch.from_numpy(enc_hs),
        torch.from_numpy(te),
        torch.from_numpy(tid),
    )

# out = (mid_out, skip_0..8, temb) = 11 tensors
ref_names = ["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb"]
print(f"\n--- Comparison (phone vs PyTorch) ---")
for i, (name, ref_t) in enumerate(zip(ref_names, out)):
    ref = ref_t.numpy().flatten()
    if i in phone_outputs:
        ph = phone_outputs[i]
        if ph.size == ref.size:
            cos = np.dot(ph, ref) / (np.linalg.norm(ph) * np.linalg.norm(ref) + 1e-12)
            mae = np.abs(ph - ref).mean()
            print(f"output_{i} ({name}): cos={cos:.6f}, mae={mae:.6f}, "
                  f"phone=[{ph.min():.3f}..{ph.max():.3f}], ref=[{ref.min():.3f}..{ref.max():.3f}]")
        else:
            print(f"output_{i} ({name}): SIZE MISMATCH phone={ph.size} vs ref={ref.size}")
    else:
        print(f"output_{i} ({name}): phone file missing")
