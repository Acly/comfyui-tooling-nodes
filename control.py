"""ControlNet-LLLite for Anima (DiT) — ComfyUI port (v2 architecture).

Adapted from kohya-ss/ComfyUI-Anima-LLLite
https://github.com/kohya-ss/ComfyUI-Anima-LLLite
Apache-2.0 license

Adapted from kohya-ss/sd-scripts. The on-disk weight format is the v2
named-key format (per-module key prefix = lllite_name, shared encoder under
``lllite_conditioning1.*``, depth embedding split per-module as
``{name}.depth_embed``); legacy ``lllite_modules.*`` files are rejected.

Differences vs. the sd-scripts reference (``networks/control_net_lllite_anima.py``):
  * No dependency on ``library.utils`` — uses stdlib logging.
  * Module discovery filters the LLM-Adapter sub-tree by class identity in
    addition to the path-based check (ComfyUI ships two distinct ``Attention``
    classes that share the bare class name).
  * ``LLLiteModuleDiT`` keeps a ``restore()`` method (and an idempotent
    ``apply_to()``); ComfyUI patches/unpatches the original Linear around
    every sampler call via ``set_model_unet_function_wrapper``.
  * Forward pass casts ``x`` and ``cond_emb`` to the LLLite parameter dtype
    so autocast / mixed-precision flows that hand us a different dtype than
    the LLLite weights still work.
  * CFG batch-size and sequence-length mismatches fall back to identity
    instead of asserting, so a slightly-off cond image cannot abort sampling.
  * The training-side ``AnimaControlNetLLLiteWrapper`` is omitted; ComfyUI
    integrates via ``model_function_wrapper`` in nodes.py instead.
"""

from __future__ import annotations

from copy import copy
import logging
import os
from dataclasses import dataclass
from typing import Any

import folder_paths
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io
from torch import nn

logger = logging.getLogger("comfyui-tooling-nodes")


# Class names of the modules that LLLite injects into. The LLM-Adapter uses
# a different ``Attention`` class with the same bare name; we filter it by
# path (``llm_adapter`` in the qualified name) and by the ``is_selfattn``
# attribute presence.
TARGET_ATTENTION_CLASS = "Attention"
TARGET_MLP_CLASS = "GPT2FeedForward"
LLM_ADAPTER_NAME = "llm_adapter"

LLLITE_ARCH_VERSION = "2"


# ----------------------------------------------------------------------------
# target_layers: atomic specifiers and presets
# ----------------------------------------------------------------------------

ATOMIC_SPECIFIERS: tuple[str, ...] = (
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
)

PRESETS: dict = {
    "self_attn_q": ("self_attn_q_pre",),
    "self_attn_qkv": ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q": ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> tuple[str, ...]:
    """Resolve a ``target_layers`` spec to a canonical atomic tuple.

    Accepts a preset name (``"self_attn_qkv"``) or a comma-separated list of
    atomic specifiers (``"self_attn_q_pre,mlp_fc1_pre"``). Returns the atomics
    in ``ATOMIC_SPECIFIERS`` order with duplicates removed.
    """
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


# ----------------------------------------------------------------------------
# Conditioning1 trunk (v2)
# ----------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


ASPP_DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        ch_half = cond_dim // 2

        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h


# ----------------------------------------------------------------------------
# LLLite module (v2: FiLM + SiLU + 5D path + depth embedding)
# ----------------------------------------------------------------------------


class LLLiteModuleDiT(nn.Module):
    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: float | None = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # Wrap in a list so the original Linear is not registered as a submodule
        # and its weights stay out of state_dict.
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (gamma, beta), zero-init for identity at start.
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.cond_emb: torch.Tensor | None = None
        self.org_forward = None

        # Set by the parent ControlNetLLLiteDiT after construction.
        self.layer_idx: int = -1
        self._depth_embeds_ref: list[nn.Parameter] = []

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layouts:
        #   self/cross attention q/k/v: (B, S, D) — already flattened in the Anima block
        #   mlp.layer1:                 (B, T, H, W, D) — passed un-flattened
        # Flatten the 5D case to 3D for the LLLite path and reshape on exit.
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb  # (B_c, S, cond_emb_dim)

        # Broadcast cond_emb to the runtime batch (CFG cond+uncond, multi-cond).
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        # Run the LLLite mini-MLP in its own parameter dtype, then cast the
        # correction back to ``x``'s dtype before adding. Robust to autocast
        # flows where x and LLLite weights have different dtypes.
        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        # Per-module depth embedding (zero-init so it's a no-op at train start).
        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx

        h = F.silu(self.down(x_proc))

        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self.org_forward(x + out)

        if is_5d:
            # org Linear out_features may differ from in_features — recover with -1.
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


# ----------------------------------------------------------------------------
# ControlNetLLLiteDiT
# ----------------------------------------------------------------------------


class ControlNetLLLiteDiT(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: float | None = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        # 4ch (RGB+mask) inpainting metadata. `inpaint_masked_input` records the training-time
        # RGB-masking policy for cond_image preparation; it does not alter the forward pass here.
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input

        self.conditioning1 = _Conditioning1(
            cond_dim,
            cond_emb_dim,
            cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)

        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        inpaint_info = (
            f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels != 3 else ""
        )
        logger.info(
            "ControlNet-LLLite (Anima v%s): created %d modules for target=%r "
            "(atomics=%s), cond_in_channels=%d, cond_dim=%d, cond_resblocks=%d, %s, "
            "cond_emb_dim=%d, mlp_dim=%d%s",
            LLLITE_ARCH_VERSION,
            n,
            target_layers,
            list(atomics),
            cond_in_channels,
            cond_dim,
            cond_resblocks,
            aspp_info,
            cond_emb_dim,
            mlp_dim,
            inpaint_info,
        )

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: tuple[str, ...]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            return False  # cross_attn K,V live in text-embedding space

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: tuple[str, ...],
        dropout: float | None,
        multiplier: float,
    ) -> list[LLLiteModuleDiT]:
        modules: list[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(
            a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre")
        )

        for name, module in dit.named_modules():
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__

            def _is_linear_like(module):
                return (
                hasattr(module, "in_features")
                and hasattr(module, "out_features")
                and callable(getattr(module, "forward", None))
            )

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                # The Anima-block Attention exposes is_selfattn; the LLM-Adapter
                # Attention does not — skip the latter even if path filter misses.
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not _is_linear_like(child):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(
                            full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier
                        )
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not _is_linear_like(child):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )

        return modules

    def set_cond_image(self, cond_image: torch.Tensor | None):
        """cond_image: (B, 3, H*16, W*16) in [-1, 1]; ``None`` clears."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()


# ----------------------------------------------------------------------------
# Save / load (named-key format; legacy lllite_modules.* is rejected)
# ----------------------------------------------------------------------------

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _from_saved_state_dict(lllite: ControlNetLLLiteDiT, weights_sd: dict) -> dict:
    """Rewrite a v2 named-key state dict back to the internal layout."""
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX) :]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(f"depth_embed slices missing for module idx(es) {missing}")
        out[_INTERNAL_DEPTH_KEY] = torch.stack([depth_slices[i] for i in range(n_modules)], dim=0)

    return out


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    weights_sd = safetensors.torch.load_file(file)

    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in a legacy ControlNet-LLLite weight format "
            f"(keys starting with '{_INTERNAL_MODULES_PREFIX}'). The current code uses a "
            f"named-key format (per-module key prefix = lllite_name, e.g. "
            f"'lllite_dit_blocks_0_self_attn_q_proj.down.weight'). Re-train with the current codebase."
        )

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info("loaded LLLite weights from %s: %s", file, info)
    return info


def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != ".safetensors":
        raise RuntimeError(f"Must use .safetensors files, got {file}")

    with safetensors.safe_open(file, framework="pt") as f:
        return f.metadata() or {}


# ----------------------------------------------------------------------------
# ComfyUI nodes for Anima ControlNet-LLLite
# ----------------------------------------------------------------------------


def _get_inner_dit(model) -> torch.nn.Module:
    """Reach the underlying Anima DiT (nn.Module) from a ComfyUI ModelPatcher."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model attribute (not a ModelPatcher?)")
    dit = getattr(inner, "diffusion_model", None)
    if dit is None:
        raise RuntimeError("MODEL.model has no .diffusion_model — not a UNet/DiT model?")
    return dit


def _target_cond_hw(latent_h: int, latent_w: int, patch_spatial: int = 2) -> tuple[int, int]:
    """Return the (H, W) the cond image / mask must be resized to.

    The LLLite ``conditioning1`` Conv has stride 16, so the cond image must be
    sized to ``latent_HW * 8`` in input pixel space (= ``token_HW * 16`` after
    DiT patchify with patch_spatial=2). The DiT internally pads the latent up
    to a multiple of ``patch_spatial`` (see ``MiniTrainDIT.forward`` →
    ``pad_to_patch_size``), so we mirror that rounding here — otherwise odd
    latent dims (e.g. 1032 px → 129 latent) yield a token-count mismatch that
    silently bypasses every LLLite module.
    """
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h * 8, padded_w * 8


def _prepare_cond_image(
    image: torch.Tensor,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
    patch_spatial: int = 2,
) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,3) in [0,1] → (1,3,H*8,W*8) in [-1,1]."""
    if image.ndim == 4 and image.shape[-1] == 3:
        # (B, H, W, 3) -> (B, 3, H, W)
        img = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")

    img = img[:1]  # use first frame only
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


def _prepare_mask(
    mask: torch.Tensor,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
    patch_spatial: int = 2,
) -> torch.Tensor:
    """ComfyUI MASK (B,H,W) in [0,1] → (1,1,H*8,W*8) binarized at 0.5.

    Returns the mask in ``{0.0, 1.0}`` (1 = inpaint area, 0 = keep). The caller
    is responsible for the ``*2-1`` rescale before concat with RGB.
    """
    if mask.ndim == 3:
        m = mask.unsqueeze(1)  # (B, 1, H, W)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)} (expected B,H,W or B,1,H,W)")

    m = m[:1]
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if m.shape[-2] != target_h or m.shape[-1] != target_w:
        m = F.interpolate(m.float(), size=(target_h, target_w), mode="nearest")
    m = (m >= 0.5).to(dtype=dtype)
    return m.to(device=device)


def _build_inpaint_cond_image(
    rgb_pm1: torch.Tensor, mask01: torch.Tensor, masked_input: bool
) -> torch.Tensor:
    """rgb_pm1: (1,3,H,W) in [-1,1], mask01: (1,1,H,W) in {0,1}. Returns (1,4,H,W).

    Mirrors ``_build_inpaint_cond_image`` in the sd-scripts training / inference
    code: the mask channel is rescaled to ``[-1, +1]`` (matches the RGB range),
    and if ``masked_input`` is set the RGB is zeroed where ``mask >= 0.5``.
    """
    if masked_input:
        keep = (mask01 < 0.5).to(rgb_pm1.dtype)
        rgb_pm1 = rgb_pm1 * keep
    mask_pm1 = mask01.to(rgb_pm1.dtype) * 2.0 - 1.0
    return torch.cat([rgb_pm1, mask_pm1], dim=1)


ETNControlNet = io.Custom("ETN_CONTROL_NET")


class ControlLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_control_load",
            display_name="Load ControlNet (tooling-nodes)",
            description="Loads ControlNet weights. Currently only supports Anima LLLite weights.",
            category="external_tooling",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("weights", folder_paths.get_filename_list("controlnet")),
            ],
            outputs=[
                io.Model.Output("out_model", "model"),
                ETNControlNet.Output("control_net"),
            ],
        )

    @classmethod
    def execute(cls, model: ModelPatcher, weights: str):  # type: ignore[override]
        weights_path = folder_paths.get_full_path("controlnet", weights)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LLLite weights not found: {weights}")

        # Architecture is fully determined by the trained weights — read everything
        # from metadata rather than exposing knobs that would just cause load errors.
        meta = read_lllite_metadata(weights_path)
        if "lllite.version" not in meta:
            raise RuntimeError(
                "Unrecognized model. This node currently only loads Anima LLLite weights."
            )
        ce_dim = int(meta.get("lllite.cond_emb_dim", 32))
        m_dim = int(meta.get("lllite.mlp_dim", 64))
        # v2 records the canonical atomic form under lllite.target_atomics; fall back
        # to the legacy preset key, then to the v1 default.
        tl = meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
        cond_dim = int(meta.get("lllite.cond_dim", 64))
        cond_resblocks = int(meta.get("lllite.cond_resblocks", 1))
        use_aspp = str(meta.get("lllite.use_aspp", "false")).lower() == "true"
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            aspp_dilations = ASPP_DEFAULT_DILATIONS
        cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))
        inpaint_masked_input = (
            str(meta.get("lllite.inpaint_masked_input", "false")).lower() == "true"
        )

        lllite = ControlNetLLLiteDiT(
            _get_inner_dit(model),
            cond_emb_dim=ce_dim,
            mlp_dim=m_dim,
            target_layers=tl,
            multiplier=1.0,
            cond_dim=cond_dim,
            cond_resblocks=cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
            inpaint_masked_input=inpaint_masked_input,
        )
        load_lllite_weights(lllite, weights_path, strict=False)
        lllite.eval().requires_grad_(False)
        return io.NodeOutput(model, lllite)


class ControlApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_control_apply",
            display_name="Apply ControlNet (tooling-nodes)",
            description="Applies ControlNet conditioning. Currently only supports Anima LLLite weights.",
            category="external_tooling",
            inputs=[
                io.Model.Input("model"),
                ETNControlNet.Input("control_net"),
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(  # type: ignore[override]
        cls,
        model: ModelPatcher,
        control_net: ControlNetLLLiteDiT,
        image: torch.Tensor,
        strength: float,
        start_percent: float,
        end_percent: float,
        mask: torch.Tensor | None = None,
    ):
        dit = _get_inner_dit(model)
        patch_spatial = int(getattr(dit, "patch_spatial", 2))

        lllite = control_net
        lllite.set_multiplier(strength)

        # Mask / cond_in_channels consistency: 4ch weights need a MASK, 3ch weights ignore it.
        if lllite.cond_in_channels == 4 and mask is None:
            raise ValueError("ControlNet weights require a mask input (inpaint mode)")
        if lllite.cond_in_channels != 4 and mask is not None:
            mask = None

        # Convert percent range -> sigma range (start_percent=0 → sigma_max).
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        # Capture image / mask tensors (cloned to detach from any upstream caching)
        src_image = image.detach().clone()
        src_mask = mask.detach().clone() if mask is not None else None
        is_inpaint = lllite.cond_in_channels == 4

        # Cache for the per-resolution preprocessed cond image (avoids repeat resize)
        cache: dict[str, Any] = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}

        # Capture any previously-installed wrapper BEFORE we clone — model_options
        # has a single "model_function_wrapper" slot, so without delegation a second
        # wrapper-installing node would silently no-op the first. Mirrors the
        # ChromaRadianceOptions pattern in comfy_extras/nodes_chroma_radiance.py.
        old_wrapper = model.model_options.get("model_function_wrapper")

        def _call_next(apply_model, input_x, timestep, c):
            if old_wrapper is not None:
                return old_wrapper(apply_model, {"input": input_x, "timestep": timestep, "c": c})
            return apply_model(input_x, timestep, **c)

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Step-range gate: skip LLLite entirely when current sigma is outside
            # [sigma_end, sigma_start]. percent_to_sigma maps 0.0 → sigma_max,
            # 1.0 → sigma_min, so the active window is sigma_end <= sigma <= sigma_start.
            sigma = float(timestep.max().item())
            if not (sigma_end <= sigma <= sigma_start):
                return _call_next(apply_model, input_x, timestep, c)

            # Anima latent shape: (B, C, T, H, W) — take spatial dims from the tail.
            latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
            device = input_x.device
            dtype = input_x.dtype

            # Move LLLite to the runtime device/dtype lazily.
            tag = (device, dtype)
            if cache["lllite_loaded_to"] != tag:
                lllite.to(device=device, dtype=dtype)
                cache["lllite_loaded_to"] = tag
                cache["cond_image_pp"] = None  # invalidate

            key = (latent_h, latent_w, device, dtype)
            if cache["key"] != key or cache["cond_image_pp"] is None:
                rgb = _prepare_cond_image(
                    src_image, latent_h, latent_w, device, dtype, patch_spatial
                )
                if is_inpaint:
                    assert src_mask is not None, "Cannot use inpaint control-net without a mask"
                    mk = _prepare_mask(src_mask, latent_h, latent_w, device, dtype, patch_spatial)
                    cache["cond_image_pp"] = _build_inpaint_cond_image(
                        rgb, mk, lllite.inpaint_masked_input
                    )
                else:
                    cache["cond_image_pp"] = rgb
                cache["key"] = key

            lllite.set_multiplier(strength)
            lllite.set_cond_image(cache["cond_image_pp"])
            lllite.apply_to()
            try:
                return _call_next(apply_model, input_x, timestep, c)
            finally:
                lllite.restore()
                lllite.clear_cond_image()

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)
