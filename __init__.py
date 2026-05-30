from comfy_api.latest import ComfyExtension, io
from . import api as api, nodes, tile, region, nsfw, translation, krita


class ExternalToolingNodes(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        node_list = [
            nodes.LoadImageCache,
            nodes.SaveImageCache,
            nodes.LoadImageBase64,
            nodes.LoadMaskBase64,
            nodes.SendImageWebSocket,
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
            translation.Translate,
            krita.KritaOutput,
            krita.KritaSendText,
            krita.KritaCanvas,
            krita.KritaSelection,
            krita.KritaImageLayer,
            krita.KritaMaskLayer,
            krita.Parameter,
            krita.KritaStyle,
            krita.KritaStyleAndPrompt,
        ]
        try:  # see #66
            import nsfw

            node_list.append(nsfw.NSFWFilter)
        except (ImportError, ModuleNotFoundError):
            import traceback

            print("[comfyui-tooling-nodes] WARNING: Could not import all nodes.")
            traceback.print_exc()

        return node_list


async def comfy_entrypoint():
    return ExternalToolingNodes()


WEB_DIRECTORY = "./js"
