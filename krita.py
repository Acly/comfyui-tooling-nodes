import sys
import torch
import numpy as np
from enum import Enum
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


class OutputBatchMode(Enum):
    default = "default"
    images = "images"
    animation = "animation"
    layers = "layers"


class KritaOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaOutput",
            display_name="Krita Output",
            category="krita",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("x", "offset x", default=0),
                io.Int.Input("y", "offset y", default=0),
                io.String.Input("name", default=""),
                io.Combo.Input(
                    "batch_mode", OutputBatchMode, "batch mode", default=OutputBatchMode.default
                ),
                io.Boolean.Input("resize_canvas", "resize canvas", default=False),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        x: int = 0,
        y: int = 0,
        name="",
        batch_mode: OutputBatchMode | str = OutputBatchMode.default,
        resize_canvas=False,
    ):
        batch_mode = batch_mode.value if isinstance(batch_mode, OutputBatchMode) else batch_mode
        info = {
            "name": name,
            "offset_x": x,
            "offset_y": y,
            "batch_mode": batch_mode,
            "resize_canvas": resize_canvas,
        }
        output = SendImageWebSocket.execute(images, "PNG")
        assert isinstance(output.ui, dict)
        output.ui["info"] = [info]
        return output


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


class SelectionContext(Enum):
    automatic = "automatic"
    entire_image = "entire image"
    mask_bounds = "mask bounds"


_selection_context_help = """
Determines the section (crop bounding box) of the image and mask to transmit:
- automatic: area around the selection determined by Krita settings
- entire image: always use the entire canvas area
- mask bounds: tight bounding box of the current selection

This affects the Selection and Canvas nodes. The offset x/y outputs indicate the top-left corner of the context area relative to the full canvas."""


class KritaSelection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_KritaSelection",
            display_name="Krita Selection",
            category="krita",
            inputs=[
                io.Combo.Input(
                    "context",
                    options=SelectionContext,
                    default=SelectionContext.entire_image,
                    tooltip=_selection_context_help,
                ),
                io.Int.Input("padding", "padding", default=0, min=0),
            ],
            outputs=[
                io.Mask.Output("mask", "mask"),
                io.Boolean.Output("active", "active"),
                io.Int.Output("x", "offset x"),
                io.Int.Output("y", "offset y"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(torch.ones(1, 512, 512), False, 0, 0)


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
