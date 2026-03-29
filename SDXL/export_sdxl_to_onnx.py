#!/usr/bin/env python3
"""
Export SDXL components (UNet, CLIP-L, CLIP-G, VAE) from separate safetensors
to ONNX format for QNN/NPU inference.

Each component is exported individually with fixed input shapes.
Supports multiple resolutions: 1024x1024, 832x1216, 1216x832.

Usage (from WSL):
  python3 /mnt/d/platform-tools/sdxl_npu/export_sdxl_to_onnx.py \
    --clip-l /mnt/j/ComfyUI/models/text_encoders/waiIllustriousSDXL_v160_clip-l.safetensors \
    --clip-g /mnt/j/ComfyUI/models/text_encoders/waiIllustriousSDXL_v160_clip-g.safetensors \
    --unet   /mnt/j/ComfyUI/models/unet/waiIllustriousSDXL_v160_unet.safetensors \
    --vae    /mnt/j/ComfyUI/models/vae/waiIllustriousSDXL_v160_vae.safetensors \
    --out-dir /mnt/d/platform-tools/sdxl_npu/onnx_export \
    --resolution 1024x1024
"""
import argparse
import gc
import math
import os
import sys
import types
from pathlib import Path

import torch
import numpy as np


def collect_unet_resnet_conditioning_modules(model: torch.nn.Module):
    """Collect ResnetBlock2D modules that consume timestep embeddings."""
    resnet_sites = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "ResnetBlock2D":
            continue
        if getattr(module, "time_emb_proj", None) is None:
            continue
        if getattr(module, "time_embedding_norm", None) != "default":
            raise NotImplementedError(
                f"External resnet bias export currently supports only time_embedding_norm='default', got {module.time_embedding_norm!r} at {name}"
            )
        resnet_sites.append((name, module))
    return resnet_sites


def install_external_resnet_bias_surgery(model: torch.nn.Module):
    """
    Patch each timestep-conditioned ResnetBlock2D so it can consume an externally
    supplied rank-4 bias tensor instead of creating one internally via
    `time_emb_proj(... )[:, :, None, None]`.
    """
    resnet_sites = collect_unet_resnet_conditioning_modules(model)

    for index, (name, module) in enumerate(resnet_sites):
        if getattr(module, "_copilot_external_bias_surgery", False):
            continue

        def external_bias_forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            external_temb = getattr(self, "_copilot_external_bias", None)
            if external_temb is not None:
                temb = external_temb if external_temb.dtype == hidden_states.dtype else external_temb.to(dtype=hidden_states.dtype)
            elif self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None, None]

            if self.time_embedding_norm == "default":
                if temb is not None:
                    hidden_states = hidden_states + temb
                hidden_states = self.norm2(hidden_states)
            elif self.time_embedding_norm == "scale_shift":
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            else:
                hidden_states = self.norm2(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if self.conv_shortcut is not None:
                if self.training:
                    input_tensor = input_tensor.contiguous()
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
            return output_tensor

        module.forward = types.MethodType(external_bias_forward, module)
        module._copilot_external_bias_surgery = True
        module._copilot_external_bias = None
        module._copilot_external_bias_name = f"resnet_bias_{index:02d}_{name.replace('.', '_')}"

    return resnet_sites


def infer_unet_resnet_spatial_shapes(
    model: torch.nn.Module,
    latent_h: int,
    latent_w: int,
):
    """Infer per-ResNet spatial shapes directly from UNet topology for fixed latent sizes."""
    spatial_shapes: dict[str, tuple[int, int]] = {}
    resnet_sites = collect_unet_resnet_conditioning_modules(model)

    for name, module in resnet_sites:
        if name.startswith("down_blocks.0.resnets."):
            spatial_shapes[name] = (latent_h, latent_w)
        elif name.startswith("down_blocks.1.resnets."):
            spatial_shapes[name] = (latent_h // 2, latent_w // 2)
        elif name.startswith("down_blocks.2.resnets."):
            spatial_shapes[name] = (latent_h // 4, latent_w // 4)
        elif name.startswith("mid_block.resnets."):
            spatial_shapes[name] = (latent_h // 4, latent_w // 4)
        elif name.startswith("up_blocks.0.resnets."):
            spatial_shapes[name] = (latent_h // 4, latent_w // 4)
        elif name.startswith("up_blocks.1.resnets."):
            spatial_shapes[name] = (latent_h // 2, latent_w // 2)
        elif name.startswith("up_blocks.2.resnets."):
            spatial_shapes[name] = (latent_h, latent_w)
        else:
            raise RuntimeError(f"Unsupported ResNet block name for static spatial inference: {name}")

    missing = [name for name, _ in resnet_sites if name not in spatial_shapes]
    if missing:
        raise RuntimeError(f"Failed to infer spatial shapes for ResNet blocks: {missing}")

    return spatial_shapes


def compute_external_resnet_biases(
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    text_embeds: torch.Tensor,
    time_id_tensors: torch.Tensor | tuple[torch.Tensor, ...],
):
    """Compute exact rank-4 ResNet bias tensors outside the exported UNet graph."""
    if isinstance(time_id_tensors, torch.Tensor):
        if len(time_id_tensors.shape) != 2 or time_id_tensors.shape[1] != 6:
            raise ValueError(f"Expected time_ids tensor with shape [batch, 6], got {tuple(time_id_tensors.shape)}")
        time_ids = time_id_tensors
    else:
        if len(time_id_tensors) != 6:
            raise ValueError(f"Expected 6 SDXL time_id tensors, got {len(time_id_tensors)}")
        time_ids = torch.cat(time_id_tensors, dim=1)

    added_cond_kwargs = {
        "text_embeds": text_embeds,
        "time_ids": time_ids,
    }

    with torch.no_grad():
        t_emb = model.get_time_embed(sample=sample, timestep=timestep)
        emb = model.time_embedding(t_emb, None)

        class_emb = model.get_class_embed(sample=sample, class_labels=None)
        if class_emb is not None:
            if model.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = model.get_aug_embed(
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        emb = emb + aug_emb if aug_emb is not None else emb

        if model.time_embed_act is not None:
            emb = model.time_embed_act(emb)

        biases = []
        for _, module in collect_unet_resnet_conditioning_modules(model):
            block_emb = emb
            if not module.skip_time_act:
                block_emb = module.nonlinearity(block_emb)
            block_bias = module.time_emb_proj(block_emb)[:, :, None, None]
            biases.append(block_bias.to(dtype=sample.dtype))

    return tuple(biases)


def export_text_encoder(
    safetensors_path: str,
    out_path: str,
    encoder_type: str,  # "clip_l" or "clip_g"
    opset: int = 17,
):
    """Export CLIP text encoder to ONNX."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    from safetensors.torch import load_file

    print(f"[{encoder_type}] Loading weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    # Determine model type from state dict keys
    has_projection = any("text_projection" in k for k in state_dict.keys())

    if encoder_type == "clip_g" or has_projection:
        # CLIP-G uses CLIPTextModelWithProjection (OpenCLIP ViT-bigG)
        from transformers import CLIPTextConfig
        # SDXL CLIP-G config: hidden_size=1280, intermediate_size=5120, num_layers=32, num_heads=20
        config = CLIPTextConfig(
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            projection_dim=1280,
            vocab_size=49408,
            max_position_embeddings=77,
        )
        model = CLIPTextModelWithProjection(config)
    else:
        # CLIP-L uses standard CLIPTextModel
        from transformers import CLIPTextConfig
        config = CLIPTextConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            projection_dim=768,
            vocab_size=49408,
            max_position_embeddings=77,
        )
        model = CLIPTextModel(config)

    # Try to load the state dict - may need key mapping
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Try with prefix stripping
        new_sd = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["text_model.", "transformer.", "model."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            new_sd[new_key] = v
        try:
            model.load_state_dict(new_sd, strict=False)
            print(f"  [warn] Loaded with non-strict matching")
        except Exception as e:
            print(f"  [warn] Partial load: {e}")
            model.load_state_dict(new_sd, strict=False)

    export_text_encoder_module(model, out_path, encoder_type, opset=opset, has_projection=has_projection)

    del model, state_dict
    gc.collect()


def export_text_encoder_module(
    model: torch.nn.Module,
    out_path: str,
    encoder_type: str,
    opset: int = 17,
    has_projection: bool | None = None,
    onnx_exporter: str = "torch_export",
):
    if has_projection is None:
        has_projection = hasattr(model, "text_projection")

    model.eval()
    model = model.to(torch.float32)

    dummy_input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[{encoder_type}] Exporting to {out_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids,),
        out_path,
        opset_version=opset,
        dynamo=(onnx_exporter != "legacy"),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"] if has_projection else ["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
    )
    print(f"[{encoder_type}] Done: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def export_unet(
    safetensors_path: str,
    out_path: str,
    height: int = 1024,
    width: int = 1024,
    opset: int = 17,
    timestep_input_mode: str = "rank1",
    onnx_exporter: str = "torch_export",
):
    """Export SDXL UNet to ONNX."""
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    print(f"[unet] Loading weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    # Detect config from state dict
    # SDXL UNet uses cross_attention_dim=2048 (concatenated CLIP-L 768 + CLIP-G 1280)
    config = {
        "sample_size": height // 8,
        "in_channels": 4,
        "out_channels": 4,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "up_block_types": [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ],
        "only_cross_attention": False,
        "block_out_channels": [320, 640, 1280],
        "layers_per_block": 2,
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "cross_attention_dim": 2048,
        "transformer_layers_per_block": [1, 2, 10],
        "attention_head_dim": [5, 10, 20],
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 256,
        "projection_class_embeddings_input_dim": 2816,
    }

    model = UNet2DConditionModel(**config)

    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  [warn] Strict load failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)

    export_unet_module(
        model,
        out_path,
        height=height,
        width=width,
        opset=opset,
        timestep_input_mode=timestep_input_mode,
        onnx_exporter=onnx_exporter,
    )

    del model, state_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def export_unet_module(
    model: torch.nn.Module,
    out_path: str,
    height: int = 1024,
    width: int = 1024,
    opset: int = 17,
    timestep_input_mode: str = "rank1",
    onnx_exporter: str = "torch_export",
    resnet_temb_mode: str = "internal",
):
    model.eval()
    model = model.to(torch.float16)

    latent_h = height // 8
    latent_w = width // 8

    # SDXL UNet inputs
    sample = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16)
    if timestep_input_mode == "rank2":
        timestep = torch.tensor([[1.0]], dtype=torch.float16)
    else:
        timestep = torch.tensor([1.0], dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16)
    # SDXL additions: text_embeds (1280) + time_ids (6)
    added_cond_kwargs_text_embeds = torch.randn(1, 1280, dtype=torch.float16)
    if timestep_input_mode == "rank2":
        added_cond_kwargs_time_ids = tuple(torch.randn(1, 1, dtype=torch.float16) for _ in range(6))
    else:
        added_cond_kwargs_time_ids = torch.randn(1, 6, dtype=torch.float16)

    external_resnet_biases = ()
    external_resnet_modules = ()
    external_resnet_bias_names = []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[unet] Exporting to {out_path} (resolution {width}x{height}, latent {latent_w}x{latent_h})...")

    if timestep_input_mode == "rank2":
        original_get_aug_embed = model.get_aug_embed

        def build_export_friendly_time_proj(time_values: torch.Tensor, time_proj: torch.nn.Module, out_dtype: torch.dtype):
            if len(time_values.shape) == 1:
                time_values = time_values[:, None]
            elif len(time_values.shape) < 2:
                raise ValueError(f"Unsupported export-friendly time tensor rank: {tuple(time_values.shape)}")

            if time_values.shape[-1] != 1:
                raise ValueError(
                    f"export-friendly time tensor expects trailing singleton dimension, got {tuple(time_values.shape)}"
                )

            half_dim = time_proj.num_channels // 2
            exponent = -math.log(10000) * torch.arange(
                start=0,
                end=half_dim,
                dtype=torch.float32,
                device=time_values.device,
            )
            exponent = exponent / (half_dim - time_proj.downscale_freq_shift)

            view_shape = [1] * len(time_values.shape)
            view_shape[-1] = half_dim
            emb = torch.exp(exponent).reshape(view_shape)
            t_emb = time_values.float() * emb
            t_emb = time_proj.scale * t_emb
            t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

            if time_proj.flip_sin_to_cos:
                first_half, second_half = torch.split(t_emb, half_dim, dim=-1)
                t_emb = torch.cat([second_half, first_half], dim=-1)

            if time_proj.num_channels % 2 == 1:
                t_emb = torch.nn.functional.pad(t_emb, (0, 1, 0, 0))

            return t_emb.to(dtype=out_dtype)

        def export_friendly_get_time_embed(self, sample: torch.Tensor, timestep: torch.Tensor | float | int):
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                raise TypeError("rank2 timestep export path expects tensor timestep input")
            if len(timesteps.shape) == 0:
                timesteps = timesteps[None, None].to(sample.device)
            elif len(timesteps.shape) == 1:
                timesteps = timesteps[:, None].to(sample.device)
            elif len(timesteps.shape) == 2:
                timesteps = timesteps.to(sample.device)
            else:
                raise ValueError(f"Unsupported timestep rank for export-friendly path: {tuple(timesteps.shape)}")

            return build_export_friendly_time_proj(timesteps, self.time_proj, sample.dtype)

        def export_friendly_get_aug_embed(self, emb, encoder_hidden_states, added_cond_kwargs):
            if self.config.addition_embed_type != "text_time":
                return original_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

            if "text_embeds" not in added_cond_kwargs:
                raise ValueError("text_time addition_embed_type requires text_embeds in added_cond_kwargs")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError("text_time addition_embed_type requires time_ids in added_cond_kwargs")

            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            if isinstance(time_ids, torch.Tensor):
                if len(time_ids.shape) != 2:
                    raise ValueError(f"rank2 export path expects time_ids rank 2 [batch, slots], got {tuple(time_ids.shape)}")
                slots = time_ids.shape[1]
                if slots != 6:
                    raise ValueError(f"rank2 export path expects exactly 6 SDXL time_id slots, got {slots}")
                time_id_slots = tuple(torch.split(time_ids, 1, dim=1))
            else:
                time_id_slots = tuple(time_ids)
                if len(time_id_slots) != 6:
                    raise ValueError(f"rank2 export path expects exactly 6 SDXL time_id tensors, got {len(time_id_slots)}")

            slot_time_embeds = [
                build_export_friendly_time_proj(slot_time_id, self.add_time_proj, emb.dtype)
                for slot_time_id in time_id_slots
            ]
            time_embeds = torch.cat(slot_time_embeds, dim=-1)
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            return self.add_embedding(add_embeds)

        model.get_time_embed = types.MethodType(export_friendly_get_time_embed, model)
        model.get_aug_embed = types.MethodType(export_friendly_get_aug_embed, model)
        print("[unet] Using export-friendly rank2 timestep path (avoids internal Expand/Unsqueeze pattern)")

    if resnet_temb_mode in {"external_inputs", "external_featuremaps"}:
        if timestep_input_mode != "rank2":
            raise ValueError(f"{resnet_temb_mode} resnet temb mode currently requires --timestep-input-mode rank2")
        external_resnet_modules = tuple(install_external_resnet_bias_surgery(model))
        if resnet_temb_mode == "external_featuremaps":
            spatial_shapes = infer_unet_resnet_spatial_shapes(
                model,
                latent_h,
                latent_w,
            )
            external_resnet_biases = tuple(
                torch.randn(
                    1,
                    module.out_channels,
                    spatial_shapes[name][0],
                    spatial_shapes[name][1],
                    dtype=torch.float16,
                )
                for name, module in external_resnet_modules
            )
        else:
            external_resnet_biases = tuple(
                torch.randn(1, module.out_channels, 1, 1, dtype=torch.float16)
                for _, module in external_resnet_modules
            )
        external_resnet_bias_names = [module._copilot_external_bias_name for _, module in external_resnet_modules]
        print(f"[unet] Using {resnet_temb_mode} for {len(external_resnet_modules)} ResNet conditioning sites")

    if timestep_input_mode == "rank2":
        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet, external_resnet_modules=()):
                super().__init__()
                self.unet = unet
                self.external_resnet_modules = tuple(external_resnet_modules)

            def _bind_external_resnet_biases(self, external_resnet_biases):
                if len(external_resnet_biases) != len(self.external_resnet_modules):
                    raise ValueError(
                        f"Expected {len(self.external_resnet_modules)} external ResNet bias tensors, got {len(external_resnet_biases)}"
                    )
                for (_, module), bias in zip(self.external_resnet_modules, external_resnet_biases):
                    module._copilot_external_bias = bias

            def _clear_external_resnet_biases(self):
                for _, module in self.external_resnet_modules:
                    module._copilot_external_bias = None

            def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_id_0, time_id_1, time_id_2, time_id_3, time_id_4, time_id_5, *external_resnet_biases):
                self._bind_external_resnet_biases(external_resnet_biases)
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": (time_id_0, time_id_1, time_id_2, time_id_3, time_id_4, time_id_5),
                }
                try:
                    return self.unet(
                        sample,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                finally:
                    self._clear_external_resnet_biases()
    else:
        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet, external_resnet_modules=()):
                super().__init__()
                self.unet = unet
                self.external_resnet_modules = tuple(external_resnet_modules)

            def _bind_external_resnet_biases(self, external_resnet_biases):
                if len(external_resnet_biases) != len(self.external_resnet_modules):
                    raise ValueError(
                        f"Expected {len(self.external_resnet_modules)} external ResNet bias tensors, got {len(external_resnet_biases)}"
                    )
                for (_, module), bias in zip(self.external_resnet_modules, external_resnet_biases):
                    module._copilot_external_bias = bias

            def _clear_external_resnet_biases(self):
                for _, module in self.external_resnet_modules:
                    module._copilot_external_bias = None

            def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, *external_resnet_biases):
                self._bind_external_resnet_biases(external_resnet_biases)
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                try:
                    return self.unet(
                        sample,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                finally:
                    self._clear_external_resnet_biases()

    wrapper = UNetWrapper(model, external_resnet_modules=external_resnet_modules)
    wrapper.eval()

    if timestep_input_mode == "rank2":
        export_inputs = (
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs_text_embeds,
            *added_cond_kwargs_time_ids,
            *external_resnet_biases,
        )
        input_names = [
            "sample",
            "timestep",
            "encoder_hidden_states",
            "text_embeds",
            "time_id_0",
            "time_id_1",
            "time_id_2",
            "time_id_3",
            "time_id_4",
            "time_id_5",
            *external_resnet_bias_names,
        ]
    else:
        export_inputs = (
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs_text_embeds,
            added_cond_kwargs_time_ids,
            *external_resnet_biases,
        )
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", *external_resnet_bias_names]

    torch.onnx.export(
        wrapper,
        export_inputs,
        out_path,
        opset_version=opset,
        dynamo=(onnx_exporter != "legacy"),
        input_names=input_names,
        output_names=["noise_pred"],
        dynamic_axes=None,  # Fixed shape for NPU
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[unet] Done: {out_path} ({size_mb:.1f} MB)")
    del wrapper
    gc.collect()


def export_vae_decoder(
    safetensors_path: str,
    out_path: str,
    height: int = 1024,
    width: int = 1024,
    opset: int = 17,
):
    """Export SDXL VAE decoder to ONNX."""
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file

    print(f"[vae] Loading weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    config = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "scaling_factor": 0.13025,
    }

    model = AutoencoderKL(**config)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        print("  [warn] Strict load failed, trying non-strict")
        model.load_state_dict(state_dict, strict=False)

    export_vae_decoder_module(model, out_path, height=height, width=width, opset=opset)

    del model, state_dict
    gc.collect()


def export_vae_decoder_module(
    model: torch.nn.Module,
    out_path: str,
    height: int = 1024,
    width: int = 1024,
    opset: int = 17,
    onnx_exporter: str = "torch_export",
):
    model.eval()
    model = model.to(torch.float32)

    latent_h = height // 8
    latent_w = width // 8

    # Only export decoder (latent -> image)
    decoder = model

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent, return_dict=False)[0]

    wrapper = VAEDecoderWrapper(decoder)
    wrapper.eval()
    dummy_latent = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[vae] Exporting decoder to {out_path} (latent {latent_w}x{latent_h})...")
    torch.onnx.export(
        wrapper,
        (dummy_latent,),
        out_path,
        opset_version=opset,
        dynamo=(onnx_exporter != "legacy"),
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes=None,  # Fixed shape for NPU
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[vae] Done: {out_path} ({size_mb:.1f} MB)")
    del wrapper
    gc.collect()


def validate_onnx(path: str):
    """Basic ONNX validation."""
    import onnx
    print(f"[validate] Checking {path}...")
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=False)
    print(f"[validate] {path} — OK (opset={model.opset_import[0].version}, "
          f"inputs={[i.name for i in model.graph.input]}, "
          f"outputs={[o.name for o in model.graph.output]})")
    del model
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description="Export SDXL components to ONNX")
    ap.add_argument("--clip-l", help="Path to CLIP-L safetensors")
    ap.add_argument("--clip-g", help="Path to CLIP-G safetensors")
    ap.add_argument("--unet", help="Path to UNet safetensors")
    ap.add_argument("--vae", help="Path to VAE safetensors")
    ap.add_argument("--diffusers-dir", help="Path to an existing Diffusers SDXL pipeline directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--resolution", default="1024x1024",
                    help="Resolution WxH (e.g. 1024x1024). Use comma for multiple: 1024x1024,832x1216,1216x832")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--component", default="all",
                    choices=["all", "clip_l", "clip_g", "unet", "vae"],
                    help="Export only specific component")
    ap.add_argument("--timestep-input-mode", default="rank1",
                    choices=["rank1", "rank2"],
                    help="UNet timestep export mode: rank1 keeps default diffusers path, rank2 uses export-friendly [batch,1] timestep input")
    ap.add_argument("--onnx-exporter", default="torch_export",
                    choices=["torch_export", "legacy"],
                    help="ONNX exporter backend: torch_export is current default, legacy uses classic torch.onnx exporter for compatibility probes")
    ap.add_argument("--resnet-temb-mode", default="internal",
                    choices=["internal", "external_inputs", "external_featuremaps"],
                    help="UNet timestep-conditioning path inside ResnetBlock2D: internal keeps default diffusers path, external_inputs exposes pre-expanded rank-4 ResNet bias tensors as extra ONNX inputs, external_featuremaps exposes fully expanded per-block bias feature maps to avoid broadcast inside ONNX")
    args = ap.parse_args()

    resolutions = []
    for r in args.resolution.split(","):
        w, h = r.strip().split("x")
        resolutions.append((int(w), int(h)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = args.component

    diffusers_dir = Path(args.diffusers_dir) if args.diffusers_dir else None
    if diffusers_dir is not None:
        if not diffusers_dir.is_dir():
            raise FileNotFoundError(diffusers_dir)
    else:
        missing = [
            name for name, value in {
                "--clip-l": args.clip_l,
                "--clip-g": args.clip_g,
                "--unet": args.unet,
                "--vae": args.vae,
            }.items() if not value
        ]
        if missing:
            raise SystemExit(f"Missing required arguments without --diffusers-dir: {', '.join(missing)}")

    # CLIP encoders — resolution-independent, export once
    if components in ("all", "clip_l"):
        if diffusers_dir is not None:
            from transformers import CLIPTextModel

            text_encoder = CLIPTextModel.from_pretrained(str(diffusers_dir / "text_encoder"), local_files_only=True)
            export_text_encoder_module(text_encoder, str(out_dir / "clip_l.onnx"), "clip_l", args.opset, has_projection=False, onnx_exporter=args.onnx_exporter)
            del text_encoder
            gc.collect()
        else:
            export_text_encoder(args.clip_l, str(out_dir / "clip_l.onnx"), "clip_l", args.opset)
        if not args.skip_validate:
            validate_onnx(str(out_dir / "clip_l.onnx"))

    if components in ("all", "clip_g"):
        if diffusers_dir is not None:
            from transformers import CLIPTextModelWithProjection

            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(str(diffusers_dir / "text_encoder_2"), local_files_only=True)
            export_text_encoder_module(text_encoder_2, str(out_dir / "clip_g.onnx"), "clip_g", args.opset, has_projection=True, onnx_exporter=args.onnx_exporter)
            del text_encoder_2
            gc.collect()
        else:
            export_text_encoder(args.clip_g, str(out_dir / "clip_g.onnx"), "clip_g", args.opset)
        if not args.skip_validate:
            validate_onnx(str(out_dir / "clip_g.onnx"))

    # UNet and VAE — resolution-dependent
    for w, h in resolutions:
        suffix = f"_{w}x{h}" if len(resolutions) > 1 else ""

        if components in ("all", "unet"):
            unet_out = str(out_dir / f"unet{suffix}.onnx")
            if diffusers_dir is not None:
                from diffusers import UNet2DConditionModel

                unet = UNet2DConditionModel.from_pretrained(str(diffusers_dir), subfolder="unet", local_files_only=True)
                export_unet_module(
                    unet,
                    unet_out,
                    height=h,
                    width=w,
                    opset=args.opset,
                    timestep_input_mode=args.timestep_input_mode,
                    onnx_exporter=args.onnx_exporter,
                    resnet_temb_mode=args.resnet_temb_mode,
                )
                del unet
                gc.collect()
            else:
                export_unet(
                    args.unet,
                    unet_out,
                    height=h,
                    width=w,
                    opset=args.opset,
                    timestep_input_mode=args.timestep_input_mode,
                    onnx_exporter=args.onnx_exporter,
                    resnet_temb_mode=args.resnet_temb_mode,
                )
            if not args.skip_validate:
                validate_onnx(unet_out)

        if components in ("all", "vae"):
            vae_out = str(out_dir / f"vae_decoder{suffix}.onnx")
            if diffusers_dir is not None:
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(str(diffusers_dir), subfolder="vae", local_files_only=True)
                export_vae_decoder_module(vae, vae_out, height=h, width=w, opset=args.opset, onnx_exporter=args.onnx_exporter)
                del vae
                gc.collect()
            else:
                export_vae_decoder(args.vae, vae_out, height=h, width=w, opset=args.opset)
            if not args.skip_validate:
                validate_onnx(vae_out)

    print("\n[done] All exports complete!")
    print(f"Output directory: {out_dir}")
    for f in sorted(out_dir.glob("*.onnx")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
