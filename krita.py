import torch
from typing import NamedTuple

import server
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
        empty = torch.zeroes(1, 512, 512, 3)
        return (empty, 512, 512, 0)


class KritaSelection:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self):
        empty = torch.ones(1, 512, 512)
        return (empty,)


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
        empty = torch.zeros(1, 512, 512, 3)
        return (empty,)


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
        empty = torch.ones(1, 512, 512)
        return (empty,)


class IntParameter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Parameter"}),
                "min": ("INT", {"default": 0}),
                "max": ("INT", {"default": 100}),
                "default": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, min: int, max: int, default: int):
        return (default,)


class BoolParameter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Parameter"}),
                "default": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOL",)
    RETURN_NAMES = ("value",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, default: bool):
        return (default,)


class NumberParameter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Parameter"}),
                "min": ("FLOAT", {"default": 0.0}),
                "max": ("FLOAT", {"default": 1.0}),
                "default": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, min: float, max: float, default: float):
        return (default,)


class TextParameter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Parameter"}),
                "type": (
                    ["general", "prompt (positive)", "prompt (negative)"],
                    {"default": "general"},
                ),
                "default": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "placeholder"
    CATEGORY = "krita"

    def placeholder(self, name: str, style: str, default: str):
        return (default,)
