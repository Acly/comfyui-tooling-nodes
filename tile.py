from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

IntArray = npt.NDArray[np.int_]


class TileLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "blending": ("INT", {"default": 8, "min": 0, "max": 256, "step": 8}),
            }
        }

    CATEGORY = "external_tooling/tiles"
    RETURN_TYPES = ("TILE_LAYOUT",)
    FUNCTION = "node"

    image_size: IntArray
    tile_size: IntArray
    padding: int
    blending: int
    tile_count: IntArray

    def node(self, image: Tensor, min_tile_size: int, padding: int, blending: int):
        self.init(image, min_tile_size, padding, blending)
        return (self,)

    def init(self, image: Tensor, min_tile_size: int, padding: int, blending: int):
        assert all([x % 8 == 0 for x in image.shape[-3:-1]]), "Image size must be divisible by 8"
        assert min_tile_size % 8 == 0, "Tile size must be divisible by 8"
        assert blending < padding, "Blending must be smaller than padding"

        self.image_size = np.array(image.shape[-3:-1])
        self.padding = padding
        self.blending = blending
        self.tile_count = self.image_size // (min_tile_size - 2 * padding)

        image_size_with_overlap = self.image_size + (self.tile_count - 1) * 2 * padding
        tile_size = np.ceil(image_size_with_overlap / self.tile_count)
        self.tile_size = (np.ceil(tile_size / 8) * 8).astype(int)

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


class ExtractImageTile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": ("TILE_LAYOUT",),
                "index": ("INT", {"min": 0}),
            }
        }

    CATEGORY = "external_tooling/tiles"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "slice"

    def slice(self, image: Tensor, layout: TileLayout, index: int):
        return (layout.tile(image, index),)


class ExtractMaskTile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "layout": ("TILE_LAYOUT",),
                "index": ("INT", {"min": 0}),
            }
        }

    CATEGORY = "external_tooling/tiles"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "slice"

    def slice(self, mask: Tensor, layout: TileLayout, index: int):
        tile = layout.tile(mask.unsqueeze(3), index)
        return (tile.squeeze(3),)


class GenerateTileMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"layout": ("TILE_LAYOUT",), "index": ("INT", {"min": 0})},
            "optional": {"blend": ("BOOLEAN",)},
        }

    CATEGORY = "external_tooling/tiles"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate"

    def generate(self, layout: TileLayout, index: int, blend: bool = False):
        return (layout.mask(layout.coord(index), blend=blend),)


class MergeImageTile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": ("TILE_LAYOUT",),
                "index": ("INT", {"min": 0}),
                "tile": ("IMAGE",),
            }
        }

    CATEGORY = "external_tooling/tiles"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"

    def merge(self, image: Tensor, layout: TileLayout, index: int, tile: Tensor):
        assert index < layout.total_count, f"Index {index} out of range"
        if index == 0:
            image = image.clone()
        layout.merge(image, index, tile)
        return (image,)
