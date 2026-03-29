import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert SDXL single-file checkpoint (.safetensors) to Diffusers directory")
    ap.add_argument("--input", required=True, help="Path to SDXL .safetensors checkpoint")
    ap.add_argument("--output", required=True, help="Output diffusers directory")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="Load/save dtype")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    os.makedirs(args.output, exist_ok=True)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    pipe = StableDiffusionXLPipeline.from_single_file(
        args.input,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe.save_pretrained(args.output)
    print(f"[ok] wrote diffusers pipeline: {args.output}")


if __name__ == "__main__":
    main()
