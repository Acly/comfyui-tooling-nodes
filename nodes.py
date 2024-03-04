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
    CATEGORY = "_external_tooling"
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
    CATEGORY = "_external_tooling"
    FUNCTION = "load_mask"

    def load_mask(self, mask):
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img,)


class SendImageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "_external_tooling"

    def send_images(self, images):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                ["PNG", image, None],
                server.client_id,
            )
            results.append(
                # Could put some kind of ID here, but for now just match them by index
                {"source": "websocket", "content-type": "image/png", "type": "output"}
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

    CATEGORY = "_external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    def crop(self, image, x, y, width, height):
        out = image[:, y : y + height, x : x + width, :]
        return (out,)


class ApplyMaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "_external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor):
        # Move the channel to the second dimension for processing
        out = image.movedim(-1, 1)

        # Check if the images are RGB, and if so, add an alpha channel initialized to 1
        if out.shape[1] == 3:  # Assuming RGB images
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)

        # Ensure masks are unsqueezed to match the alpha channel dimension if needed
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Add a batch dimension to masks
        # For single mask, expand it to match size of image batch size.
        if mask.shape[0] == 1:
            mask = mask.repeat(out.shape[0], 1, 1)

        assert mask.ndim == 3, f"Mask should have shape [B, H, W]. {mask.shape}"
        assert out.ndim == 4, f"Image should have shsape [B, C, H, W]. {out.shape}"
        assert out.shape[-2:] == mask.shape[-2:], f"{out.shape[-2:]} != {mask.shape[-2:]}"
        assert out.shape[0] == mask.shape[0], f"{out.shape[0]} != {mask.shape[0]}"
        # Apply each mask in the batch to its corresponding image's alpha channel
        for i in range(out.shape[0]):
            out[i, 3, :, :] = mask[i]

        # Move the channel back to its original dimension
        out = out.movedim(1, -1)

        return (out,)
