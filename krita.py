import torch
import numpy as np
from pathlib import Path
from typing import NamedTuple
from PIL import Image

import server
import comfy.samplers
from .nodes import SendImageWebSocket


class Publisher(NamedTuple):
    name: str
    id: str
    workflow: dict


class WorkflowExchange:
    def __init__(self, server: server.PromptServer):
        self._server = server
        self._publishers: dict[str, Publisher] = {}
        self._subscribers: list[str] = []

    async def publish(self, publisher_name: str, publisher_id: str, workflow: dict):
        publisher = Publisher(publisher_name, publisher_id, workflow)
        for client_id in self._subscribers:
            await self._notify(client_id, publisher)
        self._publishers[publisher_id] = publisher

    async def subscribe(self, client_id: str):
        if client_id in self._subscribers:
            raise KeyError("Already subscribed")
        self._subscribers.append(client_id)
        for publisher in self._publishers.values():
            await self._notify(client_id, publisher)

    def unsubscribe(self, client_id: str):
        self._subscribers.remove(client_id)

    async def _notify(self, client_id: str, publisher: Publisher):
        data = {
            "publisher": {"name": publisher.name, "id": publisher.id},
            "workflow": publisher.workflow,
        }
        await self._server.send_json("etn_workflow_published", data, client_id)


def _placeholder_image():
    path = Path(__file__).parent / "data" / "external-image-placeholder.webp"
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image)[None,]


class KritaOutput(SendImageWebSocket):
    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "krita"


class KritaCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height", "seed")
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self):
        return (_placeholder_image(), 512, 512, 0)


class KritaSelection:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self):
        return (torch.ones(1, 512, 512),)


class KritaImageLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Image"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str):
        return (_placeholder_image(),)


class KritaMaskLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Mask"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str):
        return (torch.ones(1, 512, 512),)


class _BasicTypes(str):
    basic_types = ["INT", "FLOAT", "STRING", "BOOL"]

    def __eq__(self, other):
        return other in self.basic_types or isinstance(other, (list, _BasicTypes))

    def __str__(self):
        return "BASIC"


BasicTypes = _BasicTypes()
_param_types = [
    "auto",
    "number",
    "number (integer)",
    "toggle",
    "choice",
    "text",
    "prompt (positive)",
    "prompt (negative)",
]


class Parameter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Parameter"}),
                "type": (_param_types, {"default": "auto"}),
                "default": ("STRING", {"default": ""}),
                "min": ("FLOAT", {"default": 0.0}),
                "max": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = (BasicTypes,)
    RETURN_NAMES = ("value",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, type: str, default, min, max):
        return (default,)


class KritaStyle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Style"}),
                "sampler_preset": (["auto", "regular", "live"],),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "STRING",
        "STRING",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "model",
        "clip",
        "vae",
        "positive prompt",
        "negative prompt",
        "sampler name",
        "scheduler",
        "steps",
        "guidance",
    )
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, sampler_preset: str):
        raise NotImplementedError("This workflow must be started from Krita!")
