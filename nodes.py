from __future__ import annotations
from copy import copy
from typing import NamedTuple
from PIL import Image
import numpy as np
import base64
import torch
import torch.nn.functional as F
from io import BytesIO
from server import PromptServer, BinaryEventTypes

from comfy.clip_vision import ClipVisionModel
from comfy.sd import StyleModel


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "external_tooling"
    FUNCTION = "load_image"

    def load_image(self, image):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)


class LoadMaskBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("MASK",)
    CATEGORY = "external_tooling"
    FUNCTION = "load_mask"

    def load_mask(self, mask):
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)


class SendImageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "external_tooling"

    def send_images(self, images, format):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                [format, image, None],
                server.client_id,
            )
            results.append({
                "source": "websocket",
                "content-type": f"image/{format.lower()}",
                "type": "output",
            })

        return {"ui": {"images": results}}


class CropImage:
    """Deprecated, ComfyUI has an ImageCrop node now which does the same."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    def crop(self, image, x, y, width, height):
        out = image[:, y : y + height, x : x + width, :]
        return (out,)


def to_bchw(image: torch.Tensor):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.movedim(-1, 1)


def to_bhwc(image: torch.Tensor):
    return image.movedim(1, -1)


def mask_batch(mask: torch.Tensor):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


class ApplyMaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor):
        out = to_bchw(image)
        if out.shape[1] == 3:  # Assuming RGB images
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)
        mask = mask_batch(mask)

        assert mask.ndim == 3, f"Mask should have shape [B, H, W]. {mask.shape}"
        assert out.ndim == 4, f"Image should have shape [B, C, H, W]. {out.shape}"
        assert out.shape[-2:] == mask.shape[-2:], (
            f"Image size {out.shape[-2:]} must match mask size {mask.shape[-2:]}"
        )
        is_mask_batch = mask.shape[0] == out.shape[0]

        # Apply each mask in the batch to its corresponding image's alpha channel
        for i in range(out.shape[0]):
            alpha = mask[i] if is_mask_batch else mask[0]
            out[i, 3, :, :] = alpha

        return (to_bhwc(out),)


class _ReferenceImageData(NamedTuple):
    image: torch.Tensor
    weight: float
    range: tuple[float, float]


class ReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "range_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "reference_images": ("REFERENCE_IMAGE",),
            },
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("REFERENCE_IMAGE",)
    RETURN_NAMES = ("reference_images",)
    FUNCTION = "append"

    def append(
        self,
        image: torch.Tensor,
        weight: float,
        range_start: float,
        range_end: float,
        reference_images: list[_ReferenceImageData] | None = None,
    ):
        imgs = copy(reference_images) if reference_images is not None else []
        imgs.append(_ReferenceImageData(image, weight, (range_start, range_end)))
        return (imgs,)


class ApplyReferenceImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip_vision": ("CLIP_VISION",),
                "style_model": ("STYLE_MODEL",),
                "references": ("REFERENCE_IMAGE",),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"

    def apply(
        self,
        conditioning: list[list],
        clip_vision: ClipVisionModel,
        style_model: StyleModel,
        references: list[_ReferenceImageData],
    ):
        delimiters = {0.0, 1.0}
        delimiters |= set(r.range[0] for r in references)
        delimiters |= set(r.range[1] for r in references)
        delimiters = sorted(delimiters)
        ranges = [(delimiters[i], delimiters[i + 1]) for i in range(len(delimiters) - 1)]

        embeds = [_encode_image(r.image, clip_vision, style_model, r.weight) for r in references]
        base = conditioning[0][0]
        result = []
        for start, end in ranges:
            e = [
                embeds[i]
                for i, r in enumerate(references)
                if r.range[0] <= start and r.range[1] >= end
            ]
            options = conditioning[0][1].copy()
            options["start_percent"] = start
            options["end_percent"] = end
            result.append((torch.cat([base] + e, dim=1), options))

        return (result,)


def _encode_image(
    image: torch.Tensor, clip_vision: ClipVisionModel, style_model: StyleModel, weight: float
):
    e = clip_vision.encode_image(image)
    e = style_model.get_cond(e).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
    e = _downsample_image_cond(e, weight)
    return e


def _downsample_image_cond(cond: torch.Tensor, weight: float):
    if weight >= 1.0:
        return cond
    elif weight <= 0.0:
        return torch.zeros_like(cond)
    elif weight >= 0.6:
        factor = 2
    elif weight >= 0.3:
        factor = 3
    else:
        factor = 4

    # Downsample the clip vision embedding to make it smaller, resulting in less impact
    # compared to other conditioning.
    # See https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
    (b, t, h) = cond.shape
    m = int(np.sqrt(t))
    cond = F.interpolate(
        cond.view(b, m, m, h).transpose(1, -1),
        size=(m // factor, m // factor),
        mode="area",
    )
    return cond.transpose(1, -1).reshape(b, -1, h)
