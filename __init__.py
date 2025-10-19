from comfy_api.latest import ComfyExtension, io
from . import api as api, nodes, tile, region, nsfw, translation, krita


class ExternalToolingNodes(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            nodes.LoadImageBase64,
            nodes.LoadMaskBase64,
            nodes.SendImageWebSocket,
            nodes.SendImageHTTP,
            nodes.ApplyMaskToImage,
            nodes.ReferenceImage,
            nodes.ApplyReferenceImages,
            tile.CreateTileLayout,
            tile.ExtractImageTile,
            tile.ExtractMaskTile,
            tile.GenerateTileMask,
            tile.MergeImageTile,
            region.BackgroundRegion,
            region.DefineRegion,
            region.ListRegionMasks,
            region.AttentionMask,
            nsfw.NSFWFilter,
            translation.Translate,
            krita.KritaOutput,
            krita.KritaSendText,
            krita.KritaCanvas,
            krita.KritaSelection,
            krita.KritaImageLayer,
            krita.KritaMaskLayer,
            krita.Parameter,
            krita.KritaStyle,
        ]


async def comfy_entrypoint():
    return ExternalToolingNodes()


WEB_DIRECTORY = "./js"
