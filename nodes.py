from __future__ import annotations
from PIL import Image
import numpy as np
import base64
import torch
from io import BytesIO
from server import PromptServer, BinaryEventTypes


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
            results.append(
                {"source": "websocket", "content-type": f"image/{format.lower()}", "type": "output"}
            )

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
        assert (
            out.shape[-2:] == mask.shape[-2:]
        ), f"Image size {out.shape[-2:]} must match mask size {mask.shape[-2:]}"
        is_mask_batch = mask.shape[0] == out.shape[0]

        # Apply each mask in the batch to its corresponding image's alpha channel
        for i in range(out.shape[0]):
            alpha = mask[i] if is_mask_batch else mask[0]
            out[i, 3, :, :] = alpha

        return (to_bhwc(out),)
