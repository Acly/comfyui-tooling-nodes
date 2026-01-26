from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from comfy_api.latest import io

IntArray = npt.NDArray[np.int_]


class TileLayout:
    def __init__(
        self, image: Tensor, min_tile_size: int, padding: int, blending: int, multiple: int
    ):
        assert all([x % multiple == 0 for x in image.shape[-3:-1]]), (
            "Image size must be divisible by multiple"
        )
        assert min_tile_size % multiple == 0, "Tile size must be divisible by multiple"
        assert blending <= padding, "Blending must be smaller than padding"

        self.image_size: IntArray = np.array(image.shape[-3:-1])
        self.padding: int = padding
        self.blending: int = blending
        self.tile_count: IntArray = np.maximum(1, self.image_size // (min_tile_size - 2 * padding))

        image_size_with_overlap = self.image_size + (self.tile_count - 1) * 2 * padding
        tile_size = np.ceil(image_size_with_overlap / self.tile_count)
        self.tile_size: IntArray = (np.ceil(tile_size / multiple) * multiple).astype(int)

    def size(self, coord: IntArray):
        return self.end(coord) - self.start(coord)

    def start(self, coord: IntArray, pad=0):
        offset = coord * (self.tile_size - 2 * self.padding)
        offset = offset + np.where(coord == 0, 0, pad)
        return offset

    def end(self, coord: IntArray, pad=0):
        end = self.start(coord) + self.tile_size
        end = end - np.where(coord == self.tile_count - 1, 0, pad)
        return end.clip(0, self.image_size)

    def coord(self, index: int):
        return np.array((index % self.tile_count[0], index // self.tile_count[0]))

    @property
    def total_count(self):
        return self.tile_count.prod()

    def rect(self, coord: IntArray):
        s = self.start(coord)
        e = self.end(coord)
        return (slice(None), slice(s[0], e[0]), slice(s[1], e[1]), slice(None))

    def tile(self, image: Tensor, index: int):
        return image[self.rect(self.coord(index))]

    def mask(self, coord: IntArray, blend: bool):
        from kornia.filters import box_blur

        size = self.size(coord)
        padding = self.padding if blend else self.padding - self.blending
        s = self.start(coord, padding) - self.start(coord)
        e = self.end(coord, padding) - self.start(coord)
        mask = torch.zeros((1, 1, size[0], size[1]), dtype=torch.float)
        mask[:, :, s[0] : e[0], s[1] : e[1]] = 1.0
        if blend and self.blending > 0:
            mask = box_blur(mask, (self.blending, self.blending))
        return mask.squeeze(0)

    def merge(self, image: Tensor, index: int, tile: Tensor):
        coord = self.coord(index)
        rect = self.rect(coord)
        mask = self.mask(coord, blend=True)
        mask = mask.reshape(*mask.shape, 1).repeat(1, 1, 1, image.shape[-1])
        image[rect] = (1 - mask) * image[rect] + mask * tile


class CreateTileLayout(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_TileLayout",
            display_name="Create Tile Layout",
            category="external_tooling/tiles",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("min_tile_size", default=512, min=64, max=8192, step=8),
                io.Int.Input("padding", default=32, min=0, max=8192, step=8),
                io.Int.Input("blending", default=8, min=0, max=256, step=8),
                io.Int.Input("multiple", default=8, min=1, max=1024, step=1),
            ],
            outputs=[io.Custom("TileLayout").Output(display_name="layout")],
        )

    @classmethod
    def execute(cls, image: Tensor, min_tile_size: int, padding: int, blending: int, multiple: int):
        return io.NodeOutput(TileLayout(image, min_tile_size, padding, blending, multiple))


class ExtractImageTile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ExtractImageTile",
            display_name="Extract Image Tile",
            category="external_tooling/tiles",
            inputs=[
                io.Image.Input("image"),
                io.Custom("TileLayout").Input("layout"),
                io.Int.Input("index", default=0, min=0),
            ],
            outputs=[io.Image.Output(display_name="tile")],
        )

    @classmethod
    def execute(cls, image: Tensor, layout: TileLayout, index: int):
        return io.NodeOutput(layout.tile(image, index))


class ExtractMaskTile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ExtractMaskTile",
            display_name="Extract Mask Tile",
            category="external_tooling/tiles",
            inputs=[
                io.Mask.Input("mask"),
                io.Custom("TileLayout").Input("layout"),
                io.Int.Input("index", default=0, min=0),
            ],
            outputs=[io.Mask.Output(display_name="tile")],
        )

    @classmethod
    def execute(cls, mask: Tensor, layout: TileLayout, index: int):
        tile = layout.tile(mask.unsqueeze(3), index)
        return io.NodeOutput(tile.squeeze(3))


class GenerateTileMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_GenerateTileMask",
            display_name="Generate Tile Mask",
            category="external_tooling/tiles",
            inputs=[
                io.Custom("TileLayout").Input("layout"),
                io.Int.Input("index", default=0, min=0),
                io.Boolean.Input("blend", default=False, optional=True),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, layout: TileLayout, index: int, blend: bool = False):
        return io.NodeOutput(layout.mask(layout.coord(index), blend=blend))


class MergeImageTile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_MergeImageTile",
            display_name="Merge Image Tile",
            category="external_tooling/tiles",
            inputs=[
                io.Image.Input("image"),
                io.Custom("TileLayout").Input("layout"),
                io.Int.Input("index", default=0, min=0),
                io.Image.Input("tile"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image: Tensor, layout: TileLayout, index: int, tile: Tensor):
        assert index < layout.total_count, f"Index {index} out of range"
        if index == 0:
            image = image.clone()
        layout.merge(image, index, tile)
        return io.NodeOutput(image)
