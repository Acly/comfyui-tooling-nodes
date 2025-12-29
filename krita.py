import sys
import torch
import numpy as np
from pathlib import Path
from typing import Any, NamedTuple
from PIL import Image

import server
import comfy.samplers
from comfy.comfy_types.node_typing import IO
from comfy_api.latest import io
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

    async def unsubscribe(self, client_id: str):
        if client_id in self._subscribers:
            self._subscribers.remove(client_id)
        else:
            raise KeyError("No subscriber found with id " + client_id)

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


class _BasicTypes(str):
    """Matches IO.PRIMITIVE, but also any list of choices"""

    basic_types = IO.PRIMITIVE.split(",")  # STRING, FLOAT, INT, BOOLEAN

    def __eq__(self, other):
        return other in self.basic_types or isinstance(other, (list, _BasicTypes))

    def __ne__(self, other):
        return not self.__eq__(other)


BasicTypes = _BasicTypes("BASIC")


class KritaOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaOutput",
            display_name="Krita Output",
            category="krita",
            inputs=[
                io.Image.Input("images"),
                io.Boolean.Input("resize_canvas", default=False),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_canvas: bool = False):
        result = SendImageWebSocket.execute(images, "PNG")
        ui = getattr(result, "ui", {}) or {}
        ui = dict(ui)
        # Values in the UI dict must be lists so ComfyUI can concatenate them
        # across batched executions. Store a single structured entry.
        ui["resize_canvas"] = [{"enabled": bool(resize_canvas)}]
        result.ui = ui

        return result


class KritaSendText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaSendText",
            display_name="Send Text",
            category="krita",
            inputs=[
                io.AnyType.Input("value"),
                io.String.Input("name", default="Output"),
                io.Combo.Input("type", options=["text", "markdown", "html"], default="text"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, value: Any, name: str, type: str):
        mime = {
            "text": "text/plain",
            "markdown": "text/markdown",
            "html": "text/html",
        }[type]
        text = "None"
        if value is not None:
            try:
                text = str(value)
            except Exception as e:
                text = f"Could not convert to text: {e}"

        return io.NodeOutput(ui={"text": [{"name": name, "text": text, "content-type": mime}]})


class KritaCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaCanvas",
            display_name="Krita Canvas",
            category="krita",
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(_placeholder_image(), 512, 512, 0)


class KritaSelection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaSelection",
            display_name="Krita Selection",
            category="krita",
            outputs=[io.Mask.Output(display_name="mask"), io.Boolean.Output(display_name="active")],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.ones(1, 512, 512), False)


class KritaImageLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaImageLayer",
            display_name="Krita Image Layer",
            category="krita",
            inputs=[io.String.Input("name", default="Image")],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, name: str):
        return io.NodeOutput(_placeholder_image(), torch.ones(1, 512, 512))


class KritaMaskLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaMaskLayer",
            display_name="Krita Mask Layer",
            category="krita",
            inputs=[io.String.Input("name", default="Mask")],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, name: str):
        return io.NodeOutput(torch.ones(1, 512, 512))


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
_fmax = sys.float_info.max


class Parameter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_Parameter",
            display_name="Parameter",
            category="krita",
            inputs=[
                io.String.Input("name", default="Parameter"),
                io.Combo.Input("type", options=_param_types, default="auto"),
                io.String.Input("default", default=""),
                io.Float.Input("min", default=-1e10, min=-_fmax, max=_fmax, optional=True),
                io.Float.Input("max", default=1e10, min=-_fmax, max=_fmax, optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def execute(cls, name: str, type: str, default, min=0.0, max=1.0):
        if type == "number":
            return io.NodeOutput(float(default))
        elif type == "number (integer)":
            return io.NodeOutput(int(default))
        return io.NodeOutput(default)


class KritaStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaStyle",
            display_name="Krita Style",
            category="krita",
            inputs=[
                io.String.Input("name", default="Style"),
                io.Combo.Input("sampler_preset", options=["auto", "regular", "live"]),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.String.Output(display_name="positive prompt"),
                io.String.Output(display_name="negative prompt"),
                io.Combo.Output(
                    display_name="sampler name", options=comfy.samplers.KSampler.SAMPLERS
                ),
                io.Combo.Output(
                    display_name="scheduler", options=comfy.samplers.KSampler.SCHEDULERS
                ),
                io.Int.Output(display_name="steps"),
                io.Float.Output(display_name="guidance"),
            ],
        )

    @classmethod
    def execute(cls, name: str, sampler_preset: str):
        raise NotImplementedError("This workflow must be started from Krita!")
