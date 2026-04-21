#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch


DEFAULT_MODEL_DIR = Path(r"D:\platform-tools\wan21_13b_work\official_core\official-diffusers")
DEFAULT_OUT_DIR = Path(r"D:\platform-tools\wan21_13b_work\onnx")
AIHUB_EXTERNAL_DATA_INLINE_THRESHOLD_BYTES = 1024


class ExportableRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape: Any, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(v) for v in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x.to(input_dtype)


class ExportableLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(v) for v in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.float32))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=dims, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x.to(input_dtype)


class ExportableMatMulLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "ExportableMatMulLinear":
        replacement = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        replacement.weight.data.copy_(linear.weight.detach())
        if linear.bias is not None and replacement.bias is not None:
            replacement.bias.data.copy_(linear.bias.detach())
        return replacement

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(hidden_states, self.weight.transpose(0, 1))
        if self.bias is not None:
            output = output + self.bias
        return output


class LinearizedPatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        conv3d: torch.nn.Conv3d,
        patch_size: tuple[int, int, int],
        input_shape: tuple[int, int, int, int, int],
    ):
        super().__init__()
        self.patch_size = tuple(int(v) for v in patch_size)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.batch, self.channels, self.frames, self.height, self.width = self.input_shape
        self.patch_t, self.patch_h, self.patch_w = self.patch_size
        self.post_patch_frames = self.frames // self.patch_t
        self.post_patch_height = self.height // self.patch_h
        self.post_patch_width = self.width // self.patch_w
        out_features = int(conv3d.weight.shape[0])
        self.out_features = out_features
        in_features = int(conv3d.weight[0].numel())
        self.proj: ExportableMatMulLinear | None = None
        self.framewise_conv2d: torch.nn.Conv2d | None = None

        if self.patch_t == 1:
            self.framewise_conv2d = torch.nn.Conv2d(
                self.channels,
                out_features,
                kernel_size=(self.patch_h, self.patch_w),
                stride=(self.patch_h, self.patch_w),
                bias=conv3d.bias is not None,
            )
            self.framewise_conv2d.weight.data.copy_(conv3d.weight.detach().squeeze(2).to(torch.float32))
            if conv3d.bias is not None and self.framewise_conv2d.bias is not None:
                self.framewise_conv2d.bias.data.copy_(conv3d.bias.detach().to(torch.float32))
        else:
            self.proj = ExportableMatMulLinear(in_features, out_features, bias=conv3d.bias is not None)
            self.proj.weight.data.copy_(conv3d.weight.detach().reshape(out_features, in_features).to(torch.float32))
            if conv3d.bias is not None and self.proj.bias is not None:
                self.proj.bias.data.copy_(conv3d.bias.detach().to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.framewise_conv2d is not None:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
            hidden_states = hidden_states.reshape(
                self.batch * self.frames,
                self.channels,
                self.height,
                self.width,
            )
            hidden_states = self.framewise_conv2d(hidden_states)
            hidden_states = hidden_states.reshape(
                self.batch,
                self.frames,
                self.out_features,
                self.post_patch_height,
                self.post_patch_width,
            )
            hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
            return hidden_states.reshape(
                self.batch,
                self.post_patch_frames * self.post_patch_height * self.post_patch_width,
                self.out_features,
            )

        hidden_states = hidden_states.reshape(
            self.batch,
            self.channels,
            self.post_patch_frames,
            self.patch_t,
            self.post_patch_height,
            self.patch_h,
            self.post_patch_width,
            self.patch_w,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        hidden_states = hidden_states.reshape(
            self.batch,
            self.post_patch_frames * self.post_patch_height * self.post_patch_width,
            self.channels * self.patch_t * self.patch_h * self.patch_w,
        )
        if self.proj is None:
            raise RuntimeError("linear patch projection is not initialized")
        return self.proj(hidden_states)


def _replace_rmsnorm_modules(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.RMSNorm):
            replacement = ExportableRMSNorm(
                normalized_shape=child.normalized_shape,
                eps=float(child.eps if child.eps is not None else 1e-6),
                elementwise_affine=child.elementwise_affine,
            )
            if getattr(child, "weight", None) is not None and replacement.weight is not None:
                replacement.weight.data.copy_(child.weight.detach().to(torch.float32))
            replacement = replacement.to(dtype=torch.float32)
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += _replace_rmsnorm_modules(child)
    return replaced


def _replace_layernorm_modules(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.LayerNorm):
            replacement = ExportableLayerNorm(
                normalized_shape=child.normalized_shape,
                eps=float(child.eps),
                elementwise_affine=child.elementwise_affine,
                bias=getattr(child, "bias", None) is not None,
            )
            if getattr(child, "weight", None) is not None and replacement.weight is not None:
                replacement.weight.data.copy_(child.weight.detach().to(torch.float32))
            if getattr(child, "bias", None) is not None and replacement.bias is not None:
                replacement.bias.data.copy_(child.bias.detach().to(torch.float32))
            replacement = replacement.to(dtype=torch.float32)
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += _replace_layernorm_modules(child)
    return replaced


def _replace_linear_modules(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, ExportableMatMulLinear.from_linear(child))
            replaced += 1
        else:
            replaced += _replace_linear_modules(child)
    return replaced


def _validate_onnx(path: Path) -> None:
    import onnx

    onnx.checker.check_model(str(path), full_check=False)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_export_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--export-device=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_component_dir(component_dir: Path, *, clean: bool) -> None:
    if clean and component_dir.exists():
        shutil.rmtree(component_dir)
    component_dir.mkdir(parents=True, exist_ok=True)


def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _find_small_external_initializers(onnx_path: Path, *, max_bytes: int) -> list[tuple[str, int]]:
    import onnx
    from onnx import TensorProto

    type_sizes: dict[int, int] = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    offenders: list[tuple[str, int]] = []
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        element_size = int(type_sizes.get(tensor.data_type, 0))
        if element_size <= 0:
            continue
        tensor_bytes = element_size
        for dim in tensor.dims:
            tensor_bytes *= int(dim)
        if tensor_bytes <= max_bytes:
            offenders.append((tensor.name, tensor_bytes))
    offenders.sort(key=lambda item: (item[1], item[0]))
    return offenders


def _consolidate_external_data(onnx_path: Path) -> Path | None:
    import onnx

    external_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    model = onnx.load_model(str(onnx_path), load_external_data=True)
    # AI Hub shape inference expects tiny reshape/slice/expand helper tensors to stay
    # inline in the protobuf. Exporting *all* initializers to external data (threshold=0)
    # leaves values like val_39 / val_46 / val_57 / val_70 externalized and can break
    # compile-time shape inference even though the heavyweight weights are fine.
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_path.name,
        size_threshold=AIHUB_EXTERNAL_DATA_INLINE_THRESHOLD_BYTES,
        convert_attribute=False,
    )

    small_external = _find_small_external_initializers(
        onnx_path,
        max_bytes=AIHUB_EXTERNAL_DATA_INLINE_THRESHOLD_BYTES,
    )
    if small_external:
        preview = ", ".join(f"{name} ({size}B)" for name, size in small_external[:8])
        raise RuntimeError(
            "AI Hub-friendly external-data repack failed; tiny shape/helper initializers are still external: "
            + preview
        )

    keep_names = {onnx_path.name, external_path.name}
    for child in onnx_path.parent.iterdir():
        if child.is_file() and child.name not in keep_names and child.suffix != ".json":
            child.unlink()

    return external_path if external_path.exists() else None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class WanTransformerWrapper(torch.nn.Module):
    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        patch_embedding_mode: str = "conv3d",
        input_shape: tuple[int, int, int, int, int] | None = None,
    ):
        super().__init__()
        self.transformer: Any = transformer
        self.patch_embedding_mode = str(patch_embedding_mode)
        self.input_shape = input_shape
        if self.patch_embedding_mode not in {"conv3d", "linear"}:
            raise ValueError(f"Unsupported patch embedding mode: {self.patch_embedding_mode}")
        self.linear_patch_embedding = None
        if self.patch_embedding_mode == "linear":
            if self.input_shape is None:
                raise ValueError("input_shape is required when patch_embedding_mode='linear'")
            patch_embedding = transformer.patch_embedding
            if not isinstance(patch_embedding, torch.nn.Conv3d):
                raise TypeError(f"Expected Conv3d patch embedding, got {type(patch_embedding)!r}")
            config = cast(Any, getattr(transformer, "config"))
            patch_size_any = tuple(int(v) for v in config.patch_size)
            if len(patch_size_any) != 3:
                raise ValueError(f"Expected 3D patch size, got {patch_size_any}")
            self.linear_patch_embedding = LinearizedPatchEmbedding(
                patch_embedding,
                cast(tuple[int, int, int], patch_size_any),
                self.input_shape,
            )

    def _forward_linear_patch(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.transformer

        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = (int(v) for v in transformer.config.patch_size)
        post_patch_num_frames = num_frames // patch_t
        post_patch_height = height // patch_h
        post_patch_width = width // patch_w

        rotary_emb = transformer.rope(hidden_states)
        if self.linear_patch_embedding is None:
            raise RuntimeError("linear patch embedding module is not initialized")
        hidden_states = self.linear_patch_embedding(hidden_states)

        if timestep.ndim == 2:
            timestep_seq_len = int(timestep.shape[1])
            timestep = timestep.flatten()
        else:
            timestep_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            timestep,
            encoder_hidden_states,
            None,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        for block in transformer.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if temb.ndim == 3:
            shift, scale = (transformer.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = transformer.proj_out(hidden_states)

        out_channels = int(getattr(transformer.config, "out_channels", 0) or getattr(transformer.config, "in_channels", 0))
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            patch_t,
            patch_h,
            patch_w,
            out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.contiguous().reshape(batch_size, out_channels, num_frames, height, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.patch_embedding_mode == "linear":
            return self._forward_linear_patch(hidden_states, timestep, encoder_hidden_states)
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class WanTransformerPatchTokenWrapper(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module, *, input_shape: tuple[int, int, int, int, int]):
        super().__init__()
        self.transformer: Any = transformer
        self.input_shape = tuple(int(v) for v in input_shape)

        patch_embedding = transformer.patch_embedding
        if not isinstance(patch_embedding, torch.nn.Conv3d):
            raise TypeError(f"Expected Conv3d patch embedding, got {type(patch_embedding)!r}")

        config = cast(Any, getattr(transformer, "config"))
        patch_size_any = tuple(int(v) for v in config.patch_size)
        if len(patch_size_any) != 3:
            raise ValueError(f"Expected 3D patch size, got {patch_size_any}")
        self.patch_size = cast(tuple[int, int, int], patch_size_any)
        self.patch_token_dim = int(patch_embedding.weight[0].numel())
        self.out_channels = int(getattr(config, "out_channels", 0) or getattr(config, "in_channels", 0))
        self.patch_volume = int(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

        self.patch_embedding = ExportableMatMulLinear(
            self.patch_token_dim,
            int(patch_embedding.weight.shape[0]),
            bias=patch_embedding.bias is not None,
        )
        self.patch_embedding.weight.data.copy_(patch_embedding.weight.detach().reshape(int(patch_embedding.weight.shape[0]), self.patch_token_dim).to(torch.float32))
        if patch_embedding.bias is not None and self.patch_embedding.bias is not None:
            self.patch_embedding.bias.data.copy_(patch_embedding.bias.detach().to(torch.float32))

        rope = cast(Any, getattr(transformer, "rope"))
        with torch.no_grad():
            rope_input = torch.zeros(self.input_shape, dtype=torch.float32)
            rotary_cos, rotary_sin = rope(rope_input)
        self.register_buffer("rotary_cos", rotary_cos, persistent=False)
        self.register_buffer("rotary_sin", rotary_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.transformer
        hidden_states = self.patch_embedding(hidden_states)
        rotary_emb = (self.rotary_cos, self.rotary_sin)

        if timestep.ndim == 2:
            timestep_seq_len = int(timestep.shape[1])
            timestep = timestep.flatten()
        else:
            timestep_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            timestep,
            encoder_hidden_states,
            None,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        for block in transformer.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if temb.ndim == 3:
            shift, scale = (transformer.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return transformer.proj_out(hidden_states)


class WanTransformerCoreInputsWrapper(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module, *, input_shape: tuple[int, int, int, int, int]):
        super().__init__()
        self.transformer: Any = transformer
        self.input_shape = tuple(int(v) for v in input_shape)

        rope = cast(Any, getattr(transformer, "rope"))
        with torch.no_grad():
            rope_input = torch.zeros(self.input_shape, dtype=torch.float32)
            rotary_cos, rotary_sin = rope(rope_input)
        self.register_buffer("rotary_cos", rotary_cos, persistent=False)
        self.register_buffer("rotary_sin", rotary_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.transformer
        rotary_emb = (self.rotary_cos, self.rotary_sin)

        for block in transformer.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if temb.ndim == 3:
            shift, scale = (transformer.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return transformer.proj_out(hidden_states)


class WanVAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae: Any = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


def _latent_shape(num_frames: int, height: int, width: int) -> tuple[int, int, int, int, int]:
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    return (1, 16, latent_frames, latent_h, latent_w)


def _patch_token_shape(
    latent_shape: tuple[int, int, int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[int, int, int]:
    batch, channels, frames, height, width = latent_shape
    patch_t, patch_h, patch_w = patch_size
    return (
        batch,
        (frames // patch_t) * (height // patch_h) * (width // patch_w),
        channels * patch_t * patch_h * patch_w,
    )


def export_transformer(
    model_dir: Path,
    out_path: Path,
    *,
    device: torch.device,
    transformer_interface: str,
    patch_embedding_mode: str,
    height: int,
    width: int,
    num_frames: int,
    max_sequence_length: int,
    opset: int,
    exporter: str,
    do_constant_folding: bool,
    consolidate_external_data: bool,
) -> dict[str, Any]:
    from diffusers import WanTransformer3DModel

    _release_torch_memory()

    transformer = WanTransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    transformer.eval()
    transformer.to(dtype=torch.float16)
    replaced_count = _replace_rmsnorm_modules(transformer)
    layernorm_replaced_count = _replace_layernorm_modules(transformer)
    linear_replaced_count = _replace_linear_modules(transformer)
    print(f"[info] replaced RMSNorm modules for export: {replaced_count}")
    print(f"[info] replaced LayerNorm modules for export: {layernorm_replaced_count}")
    print(f"[info] replaced Linear modules for export: {linear_replaced_count}")

    hidden_shape = _latent_shape(num_frames, height, width)
    patch_size = tuple(int(v) for v in transformer.config.patch_size)
    token_count = int((hidden_shape[2] // patch_size[0]) * (hidden_shape[3] // patch_size[1]) * (hidden_shape[4] // patch_size[2]))
    inner_dim = int(transformer.config.num_attention_heads) * int(transformer.config.attention_head_dim)
    output_patch_dim = int(transformer.config.out_channels) * int(patch_size[0]) * int(patch_size[1]) * int(patch_size[2])
    hidden_input_shape: tuple[int, ...]
    hidden_output_shape: tuple[int, ...]
    effective_patch_embedding_mode = patch_embedding_mode
    input_names: list[str]
    export_inputs: tuple[torch.Tensor, ...]
    if transformer_interface == "core_inputs":
        wrapper = WanTransformerCoreInputsWrapper(transformer, input_shape=hidden_shape).eval().to(
            device=device,
            dtype=torch.float16,
        )
        hidden_input_shape = (1, token_count, inner_dim)
        hidden_output_shape = (1, token_count, output_patch_dim)
        input_names = ["hidden_states", "temb", "timestep_proj", "encoder_hidden_states"]
        hidden_states = torch.randn(hidden_input_shape, dtype=torch.float16, device=device)
        temb = torch.randn((1, inner_dim), dtype=torch.float16, device=device)
        timestep_proj = torch.randn((1, 6, inner_dim), dtype=torch.float16, device=device)
        encoder_hidden_states = torch.randn((1, max_sequence_length, inner_dim), dtype=torch.float16, device=device)
        export_inputs = (hidden_states, temb, timestep_proj, encoder_hidden_states)
        effective_patch_embedding_mode = "host_precomputed"
    elif transformer_interface == "patch_tokens":
        wrapper = WanTransformerPatchTokenWrapper(transformer, input_shape=hidden_shape).eval().to(
            device=device,
            dtype=torch.float16,
        )
        hidden_input_shape = _patch_token_shape(hidden_shape, cast(tuple[int, int, int], patch_size))
        hidden_output_shape = (hidden_input_shape[0], hidden_input_shape[1], output_patch_dim)
        input_names = ["hidden_states", "timestep", "encoder_hidden_states"]
        hidden_states = torch.randn(hidden_input_shape, dtype=torch.float16, device=device)
        timestep = torch.tensor([999.0], dtype=torch.float32, device=device)
        encoder_hidden_states = torch.randn((1, max_sequence_length, 4096), dtype=torch.float16, device=device)
        export_inputs = (hidden_states, timestep, encoder_hidden_states)
        effective_patch_embedding_mode = "linear"
    else:
        wrapper = WanTransformerWrapper(
            transformer,
            patch_embedding_mode=patch_embedding_mode,
            input_shape=hidden_shape,
        ).eval().to(
            device=device,
            dtype=torch.float16,
        )
        hidden_input_shape = hidden_shape
        hidden_output_shape = hidden_shape
        input_names = ["hidden_states", "timestep", "encoder_hidden_states"]
        hidden_states = torch.randn(hidden_input_shape, dtype=torch.float16, device=device)
        timestep = torch.tensor([999.0], dtype=torch.float32, device=device)
        encoder_hidden_states = torch.randn((1, max_sequence_length, 4096), dtype=torch.float16, device=device)
        export_inputs = (hidden_states, timestep, encoder_hidden_states)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        export_inputs,
        str(out_path),
        opset_version=opset,
        dynamo=(exporter != "legacy"),
        input_names=input_names,
        output_names=["noise_pred"],
        dynamic_axes=None,
        do_constant_folding=do_constant_folding,
    )
    export_seconds = time.time() - t0

    external_data_path: Path | None = None
    if consolidate_external_data:
        t1 = time.time()
        external_data_path = _consolidate_external_data(out_path)
        print(f"[info] consolidated external data in {time.time() - t1:.1f}s -> {external_data_path}")

    metadata = {
        "component": "transformer",
        "model_dir": str(model_dir),
        "input_names": input_names,
        "output_names": ["noise_pred"],
        "shapes": {"hidden_states": list(hidden_input_shape), "noise_pred": list(hidden_output_shape)},
        "dtypes": {"hidden_states": "float16", "noise_pred": "float16"},
        "config": {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "max_sequence_length": max_sequence_length,
            "export_device": str(device),
            "transformer_interface": transformer_interface,
            "patch_embedding_mode": effective_patch_embedding_mode,
            "export_seconds": export_seconds,
            "do_constant_folding": do_constant_folding,
            "consolidated_external_data": bool(external_data_path),
            "rmsnorm_replaced": replaced_count,
            "layernorm_replaced": layernorm_replaced_count,
            "linear_replaced": linear_replaced_count,
            "patch_size": list(patch_size),
            "latent_shape": list(hidden_shape),
            "num_layers": int(transformer.config.num_layers),
            "num_attention_heads": int(transformer.config.num_attention_heads),
            "attention_head_dim": int(transformer.config.attention_head_dim),
        },
    }
    if transformer_interface == "core_inputs":
        metadata["shapes"].update(
            {
                "temb": [1, inner_dim],
                "timestep_proj": [1, 6, inner_dim],
                "encoder_hidden_states": [1, max_sequence_length, inner_dim],
            }
        )
        metadata["dtypes"].update(
            {
                "temb": "float16",
                "timestep_proj": "float16",
                "encoder_hidden_states": "float16",
            }
        )
    else:
        metadata["shapes"].update(
            {
                "timestep": [1],
                "encoder_hidden_states": [1, max_sequence_length, 4096],
            }
        )
        metadata["dtypes"].update(
            {
                "timestep": "float32",
                "encoder_hidden_states": "float16",
            }
        )
    if external_data_path is not None:
        metadata["external_data"] = str(external_data_path)

    del wrapper, transformer, export_inputs
    _release_torch_memory()
    return metadata


def export_vae_decoder(
    model_dir: Path,
    out_path: Path,
    *,
    device: torch.device,
    height: int,
    width: int,
    num_frames: int,
    opset: int,
    exporter: str,
    do_constant_folding: bool,
    consolidate_external_data: bool,
) -> dict[str, Any]:
    from diffusers import AutoencoderKLWan

    _release_torch_memory()

    vae = AutoencoderKLWan.from_pretrained(
        str(model_dir),
        subfolder="vae",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    vae.eval()
    vae.to(dtype=torch.float16)

    wrapper = WanVAEDecoderWrapper(vae).eval().to(device=device, dtype=torch.float16)

    latent_shape = _latent_shape(num_frames, height, width)
    latents = torch.randn(latent_shape, dtype=torch.float16, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (latents,),
        str(out_path),
        opset_version=opset,
        dynamo=(exporter != "legacy"),
        input_names=["latents"],
        output_names=["video"],
        dynamic_axes=None,
        do_constant_folding=do_constant_folding,
    )
    export_seconds = time.time() - t0

    external_data_path: Path | None = None
    if consolidate_external_data:
        t1 = time.time()
        external_data_path = _consolidate_external_data(out_path)
        print(f"[info] consolidated external data in {time.time() - t1:.1f}s -> {external_data_path}")

    metadata = {
        "component": "vae_decoder",
        "model_dir": str(model_dir),
        "input_names": ["latents"],
        "output_names": ["video"],
        "shapes": {
            "latents": list(latent_shape),
            "video": [1, 3, num_frames, height, width],
        },
        "dtypes": {
            "latents": "float16",
            "video": "float16",
        },
        "config": {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "export_device": str(device),
            "export_seconds": export_seconds,
            "do_constant_folding": do_constant_folding,
            "consolidated_external_data": bool(external_data_path),
            "latents_mean": list(vae.config.latents_mean),
            "latents_std": list(vae.config.latents_std),
            "scale_factor_temporal": int(vae.config.scale_factor_temporal),
            "scale_factor_spatial": int(vae.config.scale_factor_spatial),
            "z_dim": int(vae.config.z_dim),
        },
    }
    if external_data_path is not None:
        metadata["external_data"] = str(external_data_path)

    del wrapper, vae, latents
    _release_torch_memory()
    return metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export Wan 2.1 1.3B components to fixed-shape ONNX for phone QNN experiments.")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--exporter", choices=["torch_export", "legacy"], default="legacy")
    ap.add_argument("--export-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--component", choices=["all", "transformer", "vae"], default="all")
    ap.add_argument(
        "--transformer-patch-mode",
        choices=["conv3d", "linear"],
        default="conv3d",
        help="Export transformer patch embedding either as the original Conv3d path or as patchify+Linear for QNN-friendlier graphs.",
    )
    ap.add_argument(
        "--transformer-interface",
        choices=["full", "patch_tokens", "core_inputs"],
        default="full",
        help="Export either the full latent-to-latent transformer or a host-assisted patch-token core that avoids early/late 5D layout ops inside QNN.",
    )
    ap.add_argument("--cpu-threads", type=int, default=0, help="Optional torch CPU thread count override; 0 keeps the default.")
    ap.add_argument("--do-constant-folding", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--consolidate-external-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pack large ONNX weights into one external .data file while keeping tiny "
            "shape/helper initializers inline for AI Hub shape inference."
        ),
    )
    ap.add_argument("--clean-component-dir", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fallback-to-cpu-on-oom", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-validate", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise SystemExit("height and width must be divisible by 16")
    if (args.num_frames - 1) % 4 != 0:
        raise SystemExit("num_frames must satisfy (num_frames - 1) % 4 == 0")
    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")

    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    export_device = _resolve_export_device(args.export_device)
    print(f"[info] export device: {export_device}")
    if args.cpu_threads and args.cpu_threads > 0:
        print(f"[info] torch CPU threads: {args.cpu_threads}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"wan_t2v_1p3b_{args.width}x{args.height}_{args.num_frames}f_seq{args.max_seq_len}"
    run_root = args.out_dir / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "export_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {
            "run_tag": run_tag,
            "model_dir": str(args.model_dir),
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "max_sequence_length": args.max_seq_len,
            "opset": args.opset,
            "exporter": args.exporter,
            "export_device": str(export_device),
            "components": {},
        }

    if args.component in {"all", "transformer"}:
        transformer_dir = run_root / "transformer"
        _prepare_component_dir(transformer_dir, clean=args.clean_component_dir)
        transformer_path = transformer_dir / "model.onnx"
        try:
            transformer_meta = export_transformer(
                args.model_dir,
                transformer_path,
                device=export_device,
                transformer_interface=args.transformer_interface,
                patch_embedding_mode=args.transformer_patch_mode,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                max_sequence_length=args.max_seq_len,
                opset=args.opset,
                exporter=args.exporter,
                do_constant_folding=args.do_constant_folding,
                consolidate_external_data=args.consolidate_external_data,
            )
        except torch.OutOfMemoryError:
            if export_device.type != "cuda" or not args.fallback_to_cpu_on_oom:
                raise
            print("[warn] transformer export ran out of CUDA memory; retrying on CPU")
            _release_torch_memory()
            transformer_meta = export_transformer(
                args.model_dir,
                transformer_path,
                device=torch.device("cpu"),
                transformer_interface=args.transformer_interface,
                patch_embedding_mode=args.transformer_patch_mode,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                max_sequence_length=args.max_seq_len,
                opset=args.opset,
                exporter=args.exporter,
                do_constant_folding=args.do_constant_folding,
                consolidate_external_data=args.consolidate_external_data,
            )
        if not args.skip_validate:
            _validate_onnx(transformer_path)
        transformer_meta_path = transformer_dir / "metadata.json"
        _save_json(transformer_meta_path, transformer_meta)
        manifest["components"]["transformer"] = {
            "onnx": str(transformer_path),
            "metadata": str(transformer_meta_path),
        }
        print(f"[ok] transformer ONNX: {transformer_path}")

    if args.component in {"all", "vae"}:
        vae_dir = run_root / "vae_decoder"
        _prepare_component_dir(vae_dir, clean=args.clean_component_dir)
        vae_path = vae_dir / "model.onnx"
        try:
            vae_meta = export_vae_decoder(
                args.model_dir,
                vae_path,
                device=export_device,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                opset=args.opset,
                exporter=args.exporter,
                do_constant_folding=args.do_constant_folding,
                consolidate_external_data=args.consolidate_external_data,
            )
        except torch.OutOfMemoryError:
            if export_device.type != "cuda" or not args.fallback_to_cpu_on_oom:
                raise
            print("[warn] VAE export ran out of CUDA memory; retrying on CPU")
            _release_torch_memory()
            vae_meta = export_vae_decoder(
                args.model_dir,
                vae_path,
                device=torch.device("cpu"),
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                opset=args.opset,
                exporter=args.exporter,
                do_constant_folding=args.do_constant_folding,
                consolidate_external_data=args.consolidate_external_data,
            )
        if not args.skip_validate:
            _validate_onnx(vae_path)
        vae_meta_path = vae_dir / "metadata.json"
        _save_json(vae_meta_path, vae_meta)
        manifest["components"]["vae_decoder"] = {
            "onnx": str(vae_path),
            "metadata": str(vae_meta_path),
        }
        print(f"[ok] vae decoder ONNX: {vae_path}")

    _save_json(manifest_path, manifest)
    print(f"[ok] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
