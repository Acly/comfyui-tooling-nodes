from . import nodes

NODE_CLASS_MAPPINGS = {
    "ETN_LoadImageBase64": nodes.LoadImageBase64,
    "ETN_LoadMaskBase64": nodes.LoadMaskBase64,
    "ETN_SendImageWebSocket": nodes.SendImageWebSocket,
    "ETN_CropImage": nodes.CropImage,
    "ETN_ApplyMaskToImage": nodes.ApplyMaskToImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ETN_LoadImageBase64": "Load Image (Base64)",
    "ETN_LoadMaskBase64": "Load Mask (Base64)",
    "ETN_SendImageWebSocket": "Send Image (WebSocket)",
    "ETN_CropImage": "Crop Image",
    "ETN_ApplyMaskToImage": "Apply Mask to Image",
}
