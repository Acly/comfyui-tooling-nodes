import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from .nodes import ListWrapper

IntArray = npt.NDArray[np.int_]


class TileLayout:
    image_size: IntArray
    tile_size: IntArray
    overlap: int
    tile_count: IntArray

    def __init__(self, image_size: IntArray, min_tile_size: int, overlap: int):
        assert all([x % 8 == 0 for x in image_size]), "Image size must be divisible by 8"
        assert min_tile_size % 8 == 0, "Tile size must be divisible by 8"
        assert min_tile_size > 2 * overlap, "Tile size must be larger than total overlap"

        self.image_size = image_size
        self.overlap = overlap
        self.tile_count = image_size // (min_tile_size - overlap)

        image_size_with_overlap = self.image_size + (self.tile_count - 1) * overlap
        tile_size = np.ceil(image_size_with_overlap / self.tile_count)
        self.tile_size = (np.ceil(tile_size / 8) * 8).astype(int)

    def size(self, coord: IntArray):
        return self.end(coord) - self.start(coord)

    def start(self, coord: IntArray, overlap=True):
        offset = coord * (self.tile_size - self.overlap)
        if not overlap:
            offset = offset + np.where(coord == 0, 0, self.overlap)
        return offset

    def end(self, coord: IntArray, overlap=True):
        end = self.start(coord) + self.tile_size
        if not overlap:
            end = end - np.where(coord == self.tile_count - 1, 0, self.overlap)
        return end.clip(0, self.image_size)

    def coord(self, index: int):
        return np.array((index % self.tile_count[0], index // self.tile_count[0]))

    @property
    def total_count(self):
        return self.tile_count.prod()

    @property
    def tiles(self):
        for i in range(self.total_count):
            c = self.coord(i)
            yield self.start(c), self.end(c)


class SplitImageTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("LIST", "TILE_LAYOUT")
    FUNCTION = "tile"

    def tile(self, image: Tensor, min_tile_size: int, overlap: int):
        layout = TileLayout(np.array(image.shape[-3:-1]), min_tile_size, overlap)
        tiles = (image[:, start[0] : end[0], start[1] : end[1], :] for start, end in layout.tiles)
        return (ListWrapper([{"image": t} for t in tiles]), layout)


def _gradient(dir: tuple[int, int], length: int, width: IntArray, channels: int):
    if dir[0] != 0 and dir[1] != 0:
        grad2d = _corner_gradient(dir, length)
    else:
        axis = 0 if dir[0] != 0 else 1
        beg, end = (1, 0) if dir[axis] == 1 else (0, 1)
        grad1d = torch.linspace(beg, end, length)
        grad2d = grad1d.repeat((width[~axis], 1))
        if axis == 0:
            grad2d = grad2d.T
    if channels > 1:
        grad2d = grad2d.reshape(*grad2d.shape, 1).repeat(1, 1, channels)
    return grad2d


def _corner_gradient(dir: IntArray, length: int):
    grad0 = _gradient(np.array((dir[0], 0)), length, np.array((length, length)), 1)
    grad1 = _gradient(np.array((0, dir[1])), length, np.array((length, length)), 1)
    return grad0 * grad1


class MergeImageTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tiles": ("LIST",), "layout": ("TILE_LAYOUT",)}}

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"

    def merge(self, tiles: ListWrapper, layout: TileLayout):
        assert (
            len(tiles.content) == layout.total_count
        ), f"Expected {layout.total_count} tiles, got {len(tiles.content)}"
        tiles: list[Tensor] = [t.get("image") for t in tiles.content]
        assert all([t is not None for t in tiles]), "All list elements must be an image"

        channels = tiles[0].shape[-1]
        image_shape = (layout.image_size[0], layout.image_size[1], channels)
        image = torch.zeros(image_shape, dtype=tiles[0].dtype, device=tiles[0].device)
        for n, tile in enumerate(tiles):
            if tile.dim() == 4:
                tile = tile[0]
            self._merge_tile(n, tile, image, layout)

        return (image.unsqueeze(0),)

    def _merge_tile(self, index: int, tile: Tensor, image: Tensor, layout: TileLayout):
        overlap = layout.overlap
        coord = layout.coord(index)
        s = layout.start(coord)  #  offset of tile start relative to image origin
        e = layout.end(coord)
        si = layout.start(coord, overlap=False)  # start of inner tile non-overlap region
        ei = layout.end(coord, overlap=False)  # end of inner tile non-overlap region
        size = layout.size(coord)
        #                            image
        # +------------------------------------------------------- . .
        # |
        # |                       tile
        # |             s +--------------------+
        # |               |    <- overlap ->   |
        # |               |  si                |
        # |               |    +-----------+   |
        # |               |    |           |   |
        # |               |    +-----------+   |
        # |               |                ei  |
        # |               |                    |
        # |               +--------------------+ e
        # |
        # |               <_______ size _______>
        # .
        # .
        sr = si - s  # offset where start overlap ends relative to tile start
        er = ei - s  # offset where end overlap starts relative to tile start

        # copy tile center (without overlap borders)
        image[si[0] : ei[0], si[1] : ei[1], :] += tile[sr[0] : er[0], sr[1] : er[1], :]

        # copy overlap borders with gradient falloff towards neighbor tiles
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        gradients = [_gradient(d, overlap, er - sr, channels=tile.shape[-1]) for d in directions]

        for dir, g in zip(directions, gradients):
            d = np.array(dir)
            if (s + d < 0).any() or (e + d > layout.image_size).any():
                continue  # No overlap at the image border

            if dir == (0, 1):
                image[si[0] : ei[0], ei[1] : e[1], :] += g * tile[sr[0] : er[0], er[1] : size[1], :]
            elif dir == (0, -1):
                image[si[0] : ei[0], s[1] : si[1], :] += g * tile[sr[0] : er[0], 0 : sr[1], :]
            elif dir == (1, 0):
                image[ei[0] : e[0], si[1] : ei[1], :] += g * tile[er[0] : size[0], sr[1] : er[1], :]
            elif dir == (-1, 0):
                image[s[0] : si[0], si[1] : ei[1], :] += g * tile[0 : sr[0], sr[1] : er[1], :]
            else:  # corner
                a = np.where(d == 1, ei, s)
                b = np.where(d == 1, e, si)
                at = np.where(d == 1, er, 0)
                bt = np.where(d == 1, size, sr)
                image[a[0] : b[0], a[1] : b[1], :] += g * tile[at[0] : bt[0], at[1] : bt[1], :]
