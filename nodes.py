from __future__ import annotations
from copy import copy
from dataclasses import dataclass
import time
from typing import NamedTuple
from uuid import uuid4
from PIL import Image
import numpy as np
import base64
import torch
import torch.nn.functional as F
from io import BytesIO
from server import PromptServer, BinaryEventTypes

from comfy.clip_vision import ClipVisionModel
from comfy.sd import StyleModel
from comfy_api.latest import io


class LoadImageBase64(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadImageBase64",
            display_name="Load Image (Base64)",
            category="external_tooling",
            inputs=[io.String.Input("image", multiline=False)],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, image: str):
        _strip_prefix(image, "data:image/png;base64,")
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = None

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)


class LoadMaskBase64(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadMaskBase64",
            display_name="Load Mask (Base64)",
            category="external_tooling",
            inputs=[io.String.Input("mask", multiline=False)],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, mask: str):
        _strip_prefix(mask, "data:image/png;base64,")
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)


class SendImageWebSocket(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_SendImageWebSocket",
            display_name="Send Image (WebSocket)",
            category="external_tooling",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("format", options=["PNG", "JPEG"], default="PNG"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, format: str):
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

        return io.NodeOutput(ui={"images": results})


class ImageCache:
    @dataclass
    class Entry:
        data: bytes
        content_type: str
        timestamp: float
        retrieved: int

    def __init__(self):
        self.images: dict[str, ImageCache.Entry] = {}

    def add(self, image: Image.Image, format: str):
        key = uuid4().hex
        with BytesIO() as output:
            image.save(output, format=format, quality=95, compress_level=1)
            image_data = output.getvalue()

        self.insert(key, image_data, f"image/{format.lower()}")
        return key

    def insert(self, key: str, data: bytes, content_type: str):
        self.images[key] = ImageCache.Entry(
            data=data,
            content_type=content_type,
            timestamp=time.time(),
            retrieved=0,
        )

    def get(self, key: str, extend: bool = False):
        entry = self.images.get(key)
        if entry is None:
            return None, None
        entry.retrieved += 1
        if extend:
            entry.timestamp = time.time()
        self.prune()
        return entry.data, entry.content_type

    def prune(self):
        now = time.time()
        keys_to_delete = []
        for key, entry in self.images.items():
            d = now - entry.timestamp
            if (d > 60 and entry.retrieved > 1) or d > 600:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.images[key]

    def __contains__(self, key: str):
        return key in self.images


image_cache = ImageCache()


class LoadImageCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadImageCache",
            display_name="Load Image from Cache",
            category="external_tooling",
            inputs=[io.String.Input("id", multiline=False)],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, id: str):
        image_data, content_type = image_cache.get(id, extend=True)
        if image_data is None:
            raise ValueError(f"Image with ID {id} not found in cache.")

        img = Image.open(BytesIO(image_data))
        w, h = img.size
        c = len(img.getbands())
        normalized = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).reshape(1, h, w, c)
        match c:
            case 1:
                image = tensor.expand(1, h, w, 3)
                mask = tensor.reshape(1, h, w)
            case 3:
                image = tensor
                mask = tensor[..., 0]
            case 4:
                image = tensor[..., :3]
                mask = tensor[..., 3]

        return io.NodeOutput(image, mask)


class SaveImageCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_SaveImageCache",
            display_name="Save Image to Cache",
            category="external_tooling",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("format", options=["PNG", "JPEG"], default="PNG"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, format: str):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            key = image_cache.add(image, format)

            results.append({
                "source": "http",
                "id": key,
                "content-type": f"image/{format.lower()}",
                "type": "output",
            })
        return io.NodeOutput(ui={"images": results})


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


class ApplyMaskToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ApplyMaskToImage",
            display_name="Apply Mask to Image",
            category="external_tooling",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
            ],
            outputs=[io.Image.Output(display_name="masked")],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, mask: torch.Tensor):
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


class ReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ReferenceImage",
            display_name="Reference Image",
            category="external_tooling",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("weight", default=1.0, min=0.0, max=10.0),
                io.Float.Input("range_start", default=0.0, min=0.0, max=1.0),
                io.Float.Input("range_end", default=1.0, min=0.0, max=1.0),
                io.Custom("ReferenceImage").Input("reference_images", optional=True),
            ],
            outputs=[io.Custom("ReferenceImage").Output(display_name="reference_images")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        weight: float,
        range_start: float,
        range_end: float,
        reference_images: list[_ReferenceImageData] | None = None,
    ):
        imgs = copy(reference_images) if reference_images is not None else []
        imgs.append(_ReferenceImageData(image, weight, (range_start, range_end)))
        return (imgs,)


class ApplyReferenceImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ApplyReferenceImages",
            display_name="Apply Reference Images",
            category="external_tooling",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.ClipVision.Input("clip_vision"),
                io.StyleModel.Input("style_model"),
                io.Custom("ReferenceImage").Input("references"),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
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


def _strip_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s
