#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import gc
from pathlib import Path

# Try importing torch and diffusers to ensure they are available
try:
    import torch
    import torch.nn as nn
    from diffusers import StableDiffusionXLPipeline
except ImportError as exc:
    print(f"Error: Required Python packages (torch, diffusers) are missing: {exc}")
    sys.exit(1)


class SDXLUNetEncoder(nn.Module):
    """Part 1: time_embed + conv_in + down_blocks + mid_block."""
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

        return (sample,) + down_block_res_samples + (emb,)


class SDXLUNetDecoder(nn.Module):
    """Part 2: up_blocks + conv_norm_out + conv_out."""
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

        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        return sample


def compute_skip_shapes(width: int = 1024, height: int = 1024) -> list[tuple[int, ...]]:
    """Compute skip connection shapes for a given image resolution."""
    lh, lw = height // 8, width // 8
    return [
        (1, 320, lh, lw),            # skip_0 (conv_in)
        (1, 320, lh, lw),            # skip_1 (down0 res0)
        (1, 320, lh, lw),            # skip_2 (down0 res1)
        (1, 320, lh // 2, lw // 2),  # skip_3 (down0 downsample)
        (1, 640, lh // 2, lw // 2),  # skip_4 (down1 res0)
        (1, 640, lh // 2, lw // 2),  # skip_5 (down1 res1)
        (1, 640, lh // 4, lw // 4),  # skip_6 (down1 downsample)
        (1, 1280, lh // 4, lw // 4), # skip_7 (down2 res0)
        (1, 1280, lh // 4, lw // 4), # skip_8 (down2 res1)
    ]


def sanitize_onnx_int64_to_int32(model_path: Path, output_path: Path, external_data_name: str):
    """
    Sanitize an ONNX model by converting all INT64 tensors, initializers,
    inputs, outputs, value_infos, and Cast operations to INT32.
    This completely prevents QNN parser integer alignment / pointer overflow bugs on Windows.
    """
    import onnx
    from onnx import TensorProto, numpy_helper
    import numpy as np

    print(f"[sanitize] Loading ONNX model from {model_path}...")
    model = onnx.load(str(model_path), load_external_data=True)
    
    # 1. Convert initializers
    print("[sanitize] Converting INT64 initializers to INT32...")
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            arr_i32 = arr.astype(np.int32)
            new_init = numpy_helper.from_array(arr_i32, name=init.name)
            init.CopyFrom(new_init)
            
    # 2. Convert inputs
    print("[sanitize] Converting INT64 inputs to INT32...")
    for input_val in model.graph.input:
        if input_val.type.tensor_type.elem_type == TensorProto.INT64:
            input_val.type.tensor_type.elem_type = TensorProto.INT32
            
    # 3. Convert outputs
    print("[sanitize] Converting INT64 outputs to INT32...")
    for output_val in model.graph.output:
        if output_val.type.tensor_type.elem_type == TensorProto.INT64:
            output_val.type.tensor_type.elem_type = TensorProto.INT32
            
    # 4. Convert value_info
    print("[sanitize] Converting INT64 value_infos to INT32...")
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.INT64:
            vi.type.tensor_type.elem_type = TensorProto.INT32
            
    # 5. Convert Cast node targets
    print("[sanitize] Converting Cast node targets (INT64 -> INT32)...")
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.INT64:
                    attr.i = TensorProto.INT32
                    
    # Save the sanitized model back
    print(f"[sanitize] Saving sanitized ONNX to {output_path} with size_threshold=10240...")
    onnx.save(
        model, 
        str(output_path), 
        save_as_external_data=True,
        all_tensors_to_one_file=True, 
        location=external_data_name,
        size_threshold=10240
    )
    print("[sanitize] Done!")


def parse_resolutions(res_str: str) -> list[tuple[int, int]]:
    pairs = []
    for token in res_str.split(","):
        token = token.strip().lower()
        if "x" not in token:
            raise ValueError(f"Invalid resolution format: '{token}'. Must be WxH.")
        w, h = token.split("x", 1)
        pairs.append((int(w), int(h)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Compile SDXL split UNet for a target LoRA slot")
    ap.add_argument("--lora-path", type=str, required=True, help="Path to the .safetensors LoRA file")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="Scale for the LoRA (default: 1.0)")
    ap.add_argument("--slot", type=str, default="utena_clothing", help="Dynamic LoRA slot directory name under context/lora_slots/")
    ap.add_argument("--resolution", type=str, default="1024x1024,832x1216", help="Comma-separated resolutions to build")
    ap.add_argument("--diffusers-dir", type=str, default=None, help="Path to local diffusers pipeline. If omitted, auto-detects in build/.")
    ap.add_argument("--sdk-root", type=str, default=r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130", help="QAIRT/QNN SDK root path")
    ap.add_argument("--ndk-root", type=str, default=r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358", help="Android NDK root path")
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    lora_path = Path(args.lora_path).resolve()
    if not lora_path.is_file():
        print(f"Error: LoRA file {lora_path} not found.")
        sys.exit(1)

    # 1. Resolve diffusers pipeline
    diffusers_dir = None
    if args.diffusers_dir:
        diffusers_dir = Path(args.diffusers_dir).resolve()
    else:
        # Auto-detect candidates
        candidates = [
            root_dir / "build" / "sdxl_work" / "diffusers_pipeline",
            root_dir / "build" / "sdxl_work_v170" / "diffusers_pipeline",
            root_dir / "sdxl_npu" / "diffusers_pipeline",
        ]
        for c in candidates:
            if c.is_dir() and (c / "unet").is_dir():
                diffusers_dir = c
                break

    if not diffusers_dir or not diffusers_dir.is_dir():
        print("Error: Could not locate a valid base SDXL diffusers pipeline. Please provide --diffusers-dir.")
        sys.exit(1)

    print(f"Base model pipeline: {diffusers_dir}")
    print(f"LoRA target slot:    {args.slot}")
    print(f"LoRA path:           {lora_path} (scale: {args.lora_scale})")

    # 2. Perform LoRA Fusion
    print("\n--- Step 1: Fusing LoRA weights into UNet ---")
    fused_dir = root_dir / "build" / f"temp_fused_lora_{args.slot}"
    if fused_dir.exists():
        shutil.rmtree(fused_dir, ignore_errors=True)

    print("Loading pipeline and fusing LoRA weights...")
    pipe = StableDiffusionXLPipeline.from_pretrained(str(diffusers_dir), torch_dtype=torch.float16, local_files_only=True)
    state_dict, network_alphas, metadata = pipe.lora_state_dict(
        str(lora_path),
        unet_config=pipe.unet.config,
        return_lora_metadata=True,
    )
    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet,
    )
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.save_pretrained(str(fused_dir))
    
    # Grab the fused UNet
    unet = pipe.unet
    unet.float()
    unet.eval()

    # Free pipe memory
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse resolutions
    try:
        resolutions = parse_resolutions(args.resolution)
    except ValueError as e:
        print(f"Error parsing resolutions: {e}")
        sys.exit(1)

    sdk_root = Path(args.sdk_root).resolve()
    ndk_root = Path(args.ndk_root).resolve()
    
    # Verify compile prereqs
    qnn_patch_script = root_dir / "SDXL" / "debug" / "qnn_onnx_converter_expanddims_patch.py"
    ndk_compiler_script = root_dir / "SDXL" / "debug" / "build_android_model_lib_windows.py"

    if not qnn_patch_script.exists():
        print(f"Error: QNN converter patch script missing: {qnn_patch_script}")
        sys.exit(1)
    if not ndk_compiler_script.exists():
        print(f"Error: NDK compiler helper script missing: {ndk_compiler_script}")
        sys.exit(1)

    # 3. For each resolution, export ONNX -> QNN convert -> NDK compile
    output_slot_dir = root_dir / "build" / "lora_slots" / args.slot
    output_slot_dir.mkdir(parents=True, exist_ok=True)

    for w, h in resolutions:
        print(f"\n==================================================")
        print(f" Processing Resolution: {w}x{h}")
        print(f"==================================================")

        res_tag = f"{w}x{h}"
        onnx_temp_dir = root_dir / "build" / f"temp_onnx_{args.slot}_{res_tag}"
        onnx_temp_dir.mkdir(parents=True, exist_ok=True)

        # A. Export split UNet encoder/decoder to ONNX
        latent_h, latent_w = h // 8, w // 8
        print(f"[export] Latent dimensions: {latent_w}x{latent_h}")

        # Encoder
        print("[export] Exporting split UNet Encoder...")
        encoder = SDXLUNetEncoder(unet)
        encoder.eval()
        
        sample = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float32)
        timestep = torch.tensor([999.0], dtype=torch.float32)
        enc_states = torch.randn(1, 77, 2048, dtype=torch.float32)
        text_embeds = torch.randn(1, 1280, dtype=torch.float32)
        time_ids = torch.randn(1, 6, dtype=torch.float32)

        enc_onnx_path = onnx_temp_dir / "unet_encoder.onnx"
        torch.onnx.export(
            encoder,
            (sample, timestep, enc_states, text_embeds, time_ids),
            str(enc_onnx_path),
            opset_version=18,
            input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            output_names=["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb"],
            dynamic_axes=None,
        )
        
        # Sanitize INT64 to INT32 and save as external data
        sanitize_onnx_int64_to_int32(enc_onnx_path, enc_onnx_path, "unet_encoder.data")
        del encoder
        gc.collect()

        # Decoder
        print("[export] Exporting split UNet Decoder...")
        decoder = SDXLUNetDecoder(unet)
        decoder.eval()

        mid_out = torch.randn(1, 1280, latent_h // 4, latent_w // 4, dtype=torch.float32)
        skip_shapes = compute_skip_shapes(w, h)
        skips = [torch.randn(*s, dtype=torch.float32) for s in skip_shapes]
        temb = torch.randn(1, 1280, dtype=torch.float32)

        dec_onnx_path = onnx_temp_dir / "unet_decoder.onnx"
        torch.onnx.export(
            decoder,
            (mid_out, *skips, temb, enc_states),
            str(dec_onnx_path),
            opset_version=18,
            input_names=["mid_out"] + [f"skip_{i}" for i in range(9)] + ["temb", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes=None,
        )
        
        # Sanitize INT64 to INT32 and save as external data
        sanitize_onnx_int64_to_int32(dec_onnx_path, dec_onnx_path, "unet_decoder.data")
        del decoder
        gc.collect()

        # B. Setup QAIRT compilation environment
        qairt_env = os.environ.copy()
        qairt_env["QNN_SDK_ROOT"] = str(sdk_root)
        qairt_env["QAIRT_SDK_ROOT"] = str(sdk_root)
        qairt_env["PYTHONPATH"] = str(sdk_root / "lib" / "python") + os.pathsep + qairt_env.get("PYTHONPATH", "")
        
        # Add host bin and lib to system PATH on Windows so the frontend can load DLL dependencies
        lib_windows_path = sdk_root / "lib" / "x86_64-windows-msvc"
        bin_windows_path = sdk_root / "bin" / "x86_64-windows-msvc"
        qairt_env["PATH"] = str(lib_windows_path) + os.pathsep + str(bin_windows_path) + os.pathsep + qairt_env.get("PATH", "")

        # Fix Windows path for external data dump in QAIRT converter
        temp_tmp_dir = root_dir / "build" / "temp_tmp"
        temp_tmp_dir.mkdir(parents=True, exist_ok=True)
        qairt_env["TMPDIR"] = str(temp_tmp_dir)
        qairt_env["TEMP"] = str(temp_tmp_dir)
        qairt_env["TMP"] = str(temp_tmp_dir)

        # C. QNN ONNX conversion (FP16 mode)
        work_res_dir = root_dir / "build" / "temp_qnn_work" / args.slot / res_tag
        encoder_qnn_dir = work_res_dir / "encoder_qnn_src"
        decoder_qnn_dir = work_res_dir / "decoder_qnn_src"
        encoder_qnn_dir.mkdir(parents=True, exist_ok=True)
        decoder_qnn_dir.mkdir(parents=True, exist_ok=True)

        print("[qnn-convert] Converting Encoder to QNN model source...")
        cmd_enc = [
            sys.executable, str(qnn_patch_script),
            "--input_network", str(enc_onnx_path),
            "--output_path", str(encoder_qnn_dir / "model"),
            "--float_bitwidth", "16"
        ]
        subprocess.run(cmd_enc, env=qairt_env, check=True)

        print("[qnn-convert] Converting Decoder to QNN model source...")
        cmd_dec = [
            sys.executable, str(qnn_patch_script),
            "--input_network", str(dec_onnx_path),
            "--output_path", str(decoder_qnn_dir / "model"),
            "--float_bitwidth", "16"
        ]
        subprocess.run(cmd_dec, env=qairt_env, check=True)

        # D. NDK compilation into Android shared library (.so)
        build_work_dir = work_res_dir / "build_work"
        build_work_dir.mkdir(parents=True, exist_ok=True)

        print("[ndk-compile] Building libunet_encoder_fp16.so...")
        cmd_compile_enc = [
            sys.executable, str(ndk_compiler_script),
            "--sdk-root", str(sdk_root),
            "--model-cpp", str(encoder_qnn_dir / "model.cpp"),
            "--model-bin", str(encoder_qnn_dir / "model.bin"),
            "--ndk-root", str(ndk_root),
            "--build-dir", str(build_work_dir / "encoder"),
            "--lib-name", "unet_encoder_fp16"
        ]
        subprocess.run(cmd_compile_enc, check=True)

        print("[ndk-compile] Building libunet_decoder_fp16.so...")
        cmd_compile_dec = [
            sys.executable, str(ndk_compiler_script),
            "--sdk-root", str(sdk_root),
            "--model-cpp", str(decoder_qnn_dir / "model.cpp"),
            "--model-bin", str(decoder_qnn_dir / "model.bin"),
            "--ndk-root", str(ndk_root),
            "--build-dir", str(build_work_dir / "decoder"),
            "--lib-name", "unet_decoder_fp16"
        ]
        subprocess.run(cmd_compile_dec, check=True)

        # E. Move compiled .so libraries to dynamic preset lora slot structure
        target_res_dir = output_slot_dir / res_tag
        target_res_dir.mkdir(parents=True, exist_ok=True)

        src_so_enc = build_work_dir / "encoder" / "libs" / "arm64-v8a" / "libunet_encoder_fp16.so"
        src_so_dec = build_work_dir / "decoder" / "libs" / "arm64-v8a" / "libunet_decoder_fp16.so"

        shutil.copy2(str(src_so_enc), str(target_res_dir / "libunet_encoder_fp16.so"))
        shutil.copy2(str(src_so_dec), str(target_res_dir / "libunet_decoder_fp16.so"))

        print(f"[success] Compiled model libraries saved to: {target_res_dir}")

        # Clean up resolution-specific temp folders
        shutil.rmtree(onnx_temp_dir, ignore_errors=True)
        shutil.rmtree(work_res_dir, ignore_errors=True)

    # 4. Cleanup master temp directories
    shutil.rmtree(fused_dir, ignore_errors=True)
    shutil.rmtree(root_dir / "build" / "temp_qnn_work", ignore_errors=True)

    print(f"\n==================================================")
    print(f" ALL RESOLUTIONS COMPILED SUCCESSFULLY!")
    print(f"==================================================")
    print(f"Dynamic LoRA slot '{args.slot}' binaries reside under:")
    print(f"  {output_slot_dir}")
    print()
    print("Next deploy step:")
    print("1. Push compiled .so libraries for each resolution to the phone inside:")
    print(f"   /data/local/tmp/sdxl_qnn/lora_slots/{args.slot}/<WxH>/")
    print("2. Run the on-device context compiler 'qnn-context-binary-generator' via ADB to build target '.bin.bin' context files:")
    print(f"   adb shell 'cd /data/local/tmp/sdxl_qnn && \\")
    print("     export LD_LIBRARY_PATH=lib:bin && \\")
    print("     export ADSP_LIBRARY_PATH=lib && \\")
    print(f"     bin/qnn-context-binary-generator --backend libQnnHtp.so --model lora_slots/{args.slot}/1024x1024/libunet_encoder_fp16.so --binary_file context/lora_slots/{args.slot}/1024x1024/unet_encoder_fp16.serialized.bin --config_file htp_backend_extensions_lightning.json'")
    print("==================================================")


if __name__ == "__main__":
    main()
