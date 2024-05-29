from . import api, nodes, tile, region

NODE_CLASS_MAPPINGS = {
    "ETN_LoadImageBase64": nodes.LoadImageBase64,
    "ETN_LoadMaskBase64": nodes.LoadMaskBase64,
    "ETN_SendImageWebSocket": nodes.SendImageWebSocket,
    "ETN_CropImage": nodes.CropImage,
    "ETN_ApplyMaskToImage": nodes.ApplyMaskToImage,
    "ETN_ListAppend": nodes.ListAppend,
    "ETN_ListElement": nodes.ListElement,
    "ETN_SplitImageTiles": tile.SplitImageTiles,
    "ETN_MergeImageTiles": tile.MergeImageTiles,
    "ETN_AttentionCouple": region.AttentionCouple,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ETN_LoadImageBase64": "Load Image (Base64)",
    "ETN_LoadMaskBase64": "Load Mask (Base64)",
    "ETN_SendImageWebSocket": "Send Image (WebSocket)",
    "ETN_CropImage": "Crop Image",
    "ETN_ApplyMaskToImage": "Apply Mask to Image",
    "ETN_ListAppend": "List 🢒 Append",
    "ETN_ListElement": "List 🢒 Get Element",
    "ETN_SplitImageTiles": "Tiles 🢒 Split Image",
    "ETN_MergeImageTiles": "Tiles 🢒 Merge Image",
    "ETN_AttentionCouple": "Region 🢒 Attention Couple",
}
