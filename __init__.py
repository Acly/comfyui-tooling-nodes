from . import api, nodes, tile, region

NODE_CLASS_MAPPINGS = {
    "ETN_LoadImageBase64": nodes.LoadImageBase64,
    "ETN_LoadMaskBase64": nodes.LoadMaskBase64,
    "ETN_SendImageWebSocket": nodes.SendImageWebSocket,
    "ETN_CropImage": nodes.CropImage,
    "ETN_ApplyMaskToImage": nodes.ApplyMaskToImage,
    "ETN_TileLayout": tile.TileLayout,
    "ETN_ExtractImageTile": tile.ExtractImageTile,
    "ETN_GenerateTileMask": tile.GenerateTileMask,
    "ETN_MergeImageTile": tile.MergeImageTile,
    "ETN_DefineRegion": region.DefineRegion,
    "ETN_AttentionMask": region.AttentionMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ETN_LoadImageBase64": "Load Image (Base64)",
    "ETN_LoadMaskBase64": "Load Mask (Base64)",
    "ETN_SendImageWebSocket": "Send Image (WebSocket)",
    "ETN_CropImage": "Crop Image",
    "ETN_ApplyMaskToImage": "Apply Mask to Image",
    "ETN_ListAppend": "List 🢒 Append",
    "ETN_ListElement": "List 🢒 Get Element",
    "ETN_TileLayout": "Create Tile Layout",
    "ETN_ExtractImageTile": "Extract Image Tile",
    "ETN_MergeImageTile": "Merge Image Tile",
    "ETN_GenerateTileMask": "Generate Tile Mask",
    "ETN_DefineRegion": "Define Region",
    "ETN_AttentionMask": "Regions Attention Mask",
}
