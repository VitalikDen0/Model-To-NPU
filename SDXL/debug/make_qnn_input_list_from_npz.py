#!/usr/bin/env python3
"""Generate QNN converter input_list + raw tensors from UNet calibration.npz.

Output format (one sample per line, 5 inputs):
sample.raw timestep.raw encoder_hidden_states.raw text_embeds.raw time_ids.raw
"""

import argparse
from pathlib import Path
import numpy as np

INPUT_NAMES = [
    "sample",
    "timestep",
    "encoder_hidden_states",
    "text_embeds",
    "time_ids",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to calibration.npz")
    ap.add_argument("--out-dir", required=True, help="Output directory for raw files")
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--list-name", default="unet_input_list.txt")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, mmap_mode="r")
    n_total = int(data["sample"].shape[0])
    n = min(n_total, args.max_samples)

    target_dtype = np.float16 if args.dtype == "float16" else np.float32

    lines = []
    for i in range(n):
        sample_paths = []
        sample_dir = out_dir / f"sample_{i:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for name in INPUT_NAMES:
            arr = np.asarray(data[name][i:i+1], dtype=target_dtype)
            p = sample_dir / f"{name}.raw"
            arr.tofile(p)
            sample_paths.append(str(p.resolve()))

        lines.append(" ".join(sample_paths))

    list_path = out_dir / args.list_name
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] npz: {npz_path}")
    print(f"[ok] samples: {n}/{n_total}")
    print(f"[ok] out: {out_dir}")
    print(f"[ok] input_list: {list_path}")


if __name__ == "__main__":
    main()
