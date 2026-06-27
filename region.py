# Adapted from https://github.com/pamparamm/ComfyUI-ppm
# Adapted from https://github.com/laksjdjf/cgem156-ComfyUI/blob/main/scripts/attention_couple/node.py
# by @laksjdjf

from __future__ import annotations
from functools import partial
from typing import Any, NamedTuple
import torch
import torch.nn.functional as F
import math
from torch import Tensor, Size
import comfy.model_management
import comfy.patcher_extension
from comfy.model_patcher import ModelPatcher
from comfy.model_base import Anima, CosmosPredict2
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.sampler_helpers import convert_cond
from comfy.samplers import process_conds
from comfy_api.latest import io


COND = 0
UNCOND = 1
ANIMA_COUPLE_WRAPPER_KEY = "etn_attention_mask_anima"
ANIMA_COUPLE_PATCH_KEY = "etn_attention_mask_patch"
CONDS_COUPLE_KEY = "etn_couple_conds"
COND_UNCOND_COUPLE_KEY = "etn_couple_cond_or_uncond"
COUPLE_ACTIVE_KEY = "etn_couple_active"
NUM_TOKENS_COUPLE_KEY = "etn_couple_num_tokens"


def downsample_mask(mask: Tensor, batch: int, target_size: int, original_shape: Size) -> Tensor:
    h, w = original_shape[2], original_shape[3]
    hm, wm = mask.shape[2], mask.shape[3]
    if (h, w) == (hm, wm):  # Mask is already in latent resolution
        base_factor = 1
    elif (h * 8, w * 8) == (hm, wm):  # Mask is in image resolution, downsample by 8
        base_factor = 8
    else:
        raise ValueError(f"Bad mask size. Expected {w}x{h}, got {wm}x{hm}.")

    result = mask
    for factor in [1, 2, 4, 8]:
        size = (math.ceil(h / factor), math.ceil(w / factor))
        if size[0] * size[1] == target_size and base_factor * factor > 1:
            result = F.interpolate(mask, size=size, mode="nearest")
            break

    num_conds = mask.shape[0]
    result = result.view(num_conds, target_size, 1)
    result = result.repeat_interleave(batch, dim=0)
    return result


def reshape_mask(mask: Tensor, size: tuple[int, int], batch: int, target_size: int) -> Tensor:
    result = F.interpolate(mask, size=size, mode="nearest")
    result = result.view(mask.shape[0], target_size, 1)
    return result.repeat_interleave(batch, dim=0)


def lcm(a: int, b: int):
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers: list[int]):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm


class Region(NamedTuple):
    previous: "Region" | None
    mask: Tensor | None
    conditioning: list

    def preprocess(self):
        result: list[Region] = []
        current = self
        while current is not None:
            result.append(current)
            current = current.previous
        assert len(result) > 1, "At least 2 regions are required."

        result = list(reversed(result))
        if result[0].mask is None:  # BackgroundRegion
            masks_above = torch.stack([r.mask for r in result[1:]], dim=0)
            accumulated = torch.sum(masks_above, dim=0)
            result[0] = Region(None, 1.0 - accumulated, result[0].conditioning)
        return result


Regions = io.Custom("Regions")


class BackgroundRegion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_BackgroundRegion",
            display_name="Background Region",
            category="external_tooling/regions",
            inputs=[io.Conditioning.Input("conditioning")],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(cls, conditioning: list):
        return (Region(None, None, conditioning),)


class DefineRegion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_DefineRegion",
            display_name="Define Region",
            category="external_tooling/regions",
            inputs=[
                io.Mask.Input("mask"),
                io.Conditioning.Input("conditioning"),
                Regions.Input("regions", optional=True),
            ],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(cls, mask: Tensor, conditioning: list, regions: Region | None = None):
        if mask.dim() < 3:
            mask = mask.unsqueeze(0)
        return io.NodeOutput(Region(regions, mask, conditioning))


class ListRegionMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ListRegionMasks",
            display_name="List Region Masks",
            category="external_tooling/regions",
            inputs=[Regions.Input("regions")],
            outputs=[io.Mask.Output(display_name="masks")],
        )

    @classmethod
    def execute(cls, regions: Region):
        return io.NodeOutput(torch.stack([r.mask for r in regions.preprocess()], dim=0))


class AttentionMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_AttentionMask",
            display_name="Regions Attention Mask",
            category="external_tooling/regions",
            inputs=[io.Model.Input("model"), Regions.Input("regions")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model: ModelPatcher, regions: Region):
        return io.NodeOutput(AttentionMaskPatch.apply(model, regions))


class AttentionMaskPatch:
    def __init__(self, region_list: list[Region]):
        mask = torch.stack([r.mask for r in region_list], dim=0)
        mask_sum = mask.sum(dim=0, keepdim=True)
        assert mask_sum.sum() > 0, "There are areas that are zero in all masks."
        self.mask = mask / mask_sum
        self.region_conds = [r.conditioning for r in region_list]
        self.conds = [r.conditioning[0][0] for r in region_list]
        self.num_tokens = [cond.shape[1] for cond in self.conds]
        self.num_conds = len(region_list)
        self.batch_size = 0

    @staticmethod
    def apply(model: ModelPatcher, regions: Region):
        patch = AttentionMaskPatch(regions.preprocess())
        if _is_anima_couple_model(model):
            return patch.apply_anima(model)

        def attn2_patch(q: Tensor, k: Tensor, v: Tensor, extra_options: dict):
            assert k.mean() == v.mean(), "k and v must be the same."
            device, dtype = q.device, q.dtype

            if patch.conds[0].device != device or patch.conds[0].dtype != dtype:
                patch.conds = [cond.to(device, dtype=dtype) for cond in patch.conds]
            if patch.mask.device != device or patch.mask.dtype != dtype:
                patch.mask = patch.mask.to(device, dtype=dtype)

            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            patch.batch_size = q.shape[0] // num_chunks
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            lcm_tokens = lcm_for_list(patch.num_tokens + [k.shape[1]])
            conds_tensor = [
                cond.repeat(patch.batch_size, lcm_tokens // patch.num_tokens[i], 1)
                for i, cond in enumerate(patch.conds)
            ]
            conds_tensor = torch.cat(conds_tensor, dim=0)

            qs, ks = [], []
            for i, cond_or_uncond in reversed(list(enumerate(cond_or_unconds))):
                if cond_or_uncond == 1:  # uncond
                    k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
                    qs.insert(0, q_chunks[i])
                    ks.insert(0, k_target)
                else:
                    qs.insert(0, q_chunks[i].repeat(patch.num_conds, 1, 1))
                    ks.insert(0, conds_tensor)
                    for _ in range(patch.num_conds - 1):
                        cond_or_unconds.insert(i, 0)

            qs = torch.cat(qs, dim=0)
            ks = torch.cat(ks, dim=0)
            return qs, ks, ks

        def attn2_output_patch(out: Tensor, extra_options: dict):
            num_conds = patch.num_conds
            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = downsample_mask(
                patch.mask, patch.batch_size, out.shape[1], extra_options["original_shape"]
            )
            outputs: list[Tensor] = []
            pos = 0
            i = 0
            while i < len(cond_or_unconds):
                if cond_or_unconds[i] == 1:  # uncond
                    outputs.append(out[pos : pos + patch.batch_size])
                    pos += patch.batch_size
                else:
                    masked = out[pos : pos + num_conds * patch.batch_size] * mask_downsample
                    masked = masked.view(num_conds, patch.batch_size, out.shape[1], out.shape[2])
                    masked = masked.sum(dim=0)
                    outputs.append(masked)
                    pos += num_conds * patch.batch_size
                    for _ in range(num_conds - 1):
                        cond_or_unconds.pop(i)
                i += 1

            return torch.cat(outputs, dim=0)

        new_model = model.clone()
        new_model.set_model_attn2_patch(attn2_patch)
        new_model.set_model_attn2_output_patch(attn2_output_patch)
        new_model.set_attachments("etn_attention_mask", patch)
        return new_model

    def apply_anima(self, model: ModelPatcher):
        new_model = model.clone()
        _patch_cosmos_attention(new_model)

        device = comfy.model_management.get_torch_device()
        conds_converted = [convert_cond(cond)[0] for cond in self.region_conds]
        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            ANIMA_COUPLE_WRAPPER_KEY,
            _anima_couple_sample_wrapper(conds_converted, device),
        )
        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            ANIMA_COUPLE_WRAPPER_KEY,
            _anima_couple_diffusion_wrapper(self),
        )
        new_model.set_attachments("etn_attention_mask", self)
        return new_model


def _is_anima_couple_model(model: ModelPatcher) -> bool:
    model_type = type(model.model)
    return issubclass(model_type, (Anima, CosmosPredict2))


def _anima_couple_sample_wrapper(conds_converted: list, device):
    def sample_wrapper(executor, *args, **kwargs):
        if len(conds_converted) > 0:
            guider = args[0]
            extra_options: dict[str, Any] = args[2]
            seed: int = extra_options["seed"]
            noise: Tensor = args[4]
            latent_image: Tensor = args[5]
            denoise_mask: Tensor | None = args[6]

            conds_processed = process_conds(
                guider.inner_model,
                noise,
                {"positive": conds_converted},
                device,
                latent_image,
                denoise_mask,
                seed,
                latent_shapes=[latent_image.shape],
            )["positive"]

            conds_couple = [cond["model_conds"]["c_crossattn"].cond for cond in conds_processed]

            model_options: dict[str, Any] = extra_options["model_options"]
            transformer_options: dict[str, Any] = model_options.get("transformer_options", {}).copy()
            transformer_options[CONDS_COUPLE_KEY] = conds_couple
            transformer_options[NUM_TOKENS_COUPLE_KEY] = [cond.shape[1] for cond in conds_couple]
            model_options["transformer_options"] = transformer_options

        return executor(*args, **kwargs)

    return sample_wrapper


def _anima_couple_diffusion_wrapper(patch: AttentionMaskPatch):
    def diffusion_wrapper(executor, *args, **kwargs):
        anima_model = executor.class_obj
        x: Tensor = args[0]
        transformer_options: dict[str, Any] = kwargs.get("transformer_options", {}).copy()
        patch_spatial = getattr(anima_model, "patch_spatial", 1)

        activations_shape = list(x.shape)
        activations_shape[-2] = activations_shape[-2] // patch_spatial
        activations_shape[-1] = activations_shape[-1] // patch_spatial

        transformer_options["activations_shape"] = activations_shape
        transformer_options[ANIMA_COUPLE_PATCH_KEY] = patch
        kwargs["transformer_options"] = transformer_options

        return executor(*args, **kwargs)

    return diffusion_wrapper


def pre_cross_attention(
    patch: AttentionMaskPatch,
    transformer_options: dict,
    x: Tensor,
    context: Tensor,
    rope_emb: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None, dict]:
    transformer_options = transformer_options.copy()
    if CONDS_COUPLE_KEY not in transformer_options:
        transformer_options[COND_UNCOND_COUPLE_KEY] = list(transformer_options["cond_or_uncond"])
        transformer_options[COUPLE_ACTIVE_KEY] = False
        return x, context, rope_emb, transformer_options

    conds: list[Tensor] = transformer_options[CONDS_COUPLE_KEY]
    num_tokens_c: list[int] = transformer_options[NUM_TOKENS_COUPLE_KEY]
    cond_or_uncond = transformer_options["cond_or_uncond"]

    num_chunks = len(cond_or_uncond)
    batch = x.shape[0] // num_chunks
    x_chunks = x.chunk(num_chunks, dim=0)
    c_chunks = context.chunk(num_chunks, dim=0)
    lcm_tokens_c = lcm_for_list(num_tokens_c + [context.shape[1]])
    conds_c_tensor = torch.cat(
        [cond.repeat(batch, lcm_tokens_c // num_tokens_c[i], 1) for i, cond in enumerate(conds)],
        dim=0,
    )

    xs, cs = [], []
    cond_or_uncond_couple = []
    for i, cond_type in enumerate(cond_or_uncond):
        x_target = x_chunks[i]
        c_target = c_chunks[i].repeat(1, lcm_tokens_c // context.shape[1], 1)
        if cond_type == UNCOND:
            xs.append(x_target)
            cs.append(c_target)
            cond_or_uncond_couple.append(UNCOND)
        else:
            xs.append(x_target.repeat(patch.num_conds, 1, 1))
            cs.append(conds_c_tensor)
            cond_or_uncond_couple.extend([COND] * patch.num_conds)

    transformer_options[COND_UNCOND_COUPLE_KEY] = cond_or_uncond_couple
    transformer_options[COUPLE_ACTIVE_KEY] = True

    return torch.cat(xs, dim=0), torch.cat(cs, dim=0), rope_emb, transformer_options


def cross_attention_output(patch: AttentionMaskPatch, transformer_options: dict, out: Tensor):
    cond_or_uncond = transformer_options[COND_UNCOND_COUPLE_KEY]
    size = tuple(transformer_options["activations_shape"][-2:])
    batch = out.shape[0] // len(cond_or_uncond)
    mask = patch.mask.to(out.device, dtype=out.dtype)
    mask_downsample = reshape_mask(mask, size, batch, out.shape[1])

    outputs = []
    cond_outputs = []
    i_cond = 0
    for i, cond_type in enumerate(cond_or_uncond):
        pos, next_pos = i * batch, (i + 1) * batch
        if cond_type == UNCOND:
            outputs.append(out[pos:next_pos])
        else:
            pos_cond, next_pos_cond = i_cond * batch, (i_cond + 1) * batch
            cond_outputs.append(out[pos:next_pos] * mask_downsample[pos_cond:next_pos_cond])
            i_cond += 1

    if len(cond_outputs) > 0:
        outputs.append(torch.stack(cond_outputs).sum(0))

    return torch.cat(outputs, dim=0)


def _patch_cosmos_attention(model_patcher: ModelPatcher):
    cosmos_model = model_patcher.get_model_object("diffusion_model")
    for block_name, block in (
        (n, b)
        for n, b in cosmos_model.named_modules()
        if ("cross_attn" in n or "self_attn" in n) and isinstance(b, CosmosAttention)
    ):
        patch_name = f"diffusion_model.{block_name}.forward"
        if patch_name not in model_patcher.object_patches:
            model_patcher.add_object_patch(patch_name, partial(_cosmos_attention_forward_patched, block))


def _cosmos_attention_forward_patched(
    self,
    x: Tensor,
    context: Tensor | None = None,
    rope_emb: Tensor | None = None,
    transformer_options: dict | None = None,
) -> Tensor:
    transformer_options = transformer_options if transformer_options is not None else {}
    patch: AttentionMaskPatch | None = transformer_options.get(ANIMA_COUPLE_PATCH_KEY)

    if context is not None and patch is not None:
        x, context, rope_emb, transformer_options = pre_cross_attention(
            patch, transformer_options, x, context, rope_emb
        )

    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
    output = self.compute_attention(q, k, v, transformer_options=transformer_options)

    if context is not None and patch is not None and transformer_options.get(COUPLE_ACTIVE_KEY, False):
        output = cross_attention_output(patch, transformer_options, output)

    return output
