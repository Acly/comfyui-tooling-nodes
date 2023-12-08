from aiohttp import web
from typing import NamedTuple
import json

import comfy.utils
import comfy.supported_models
import comfy.model_detection
import folder_paths
import server

input_block_name = "model.diffusion_model.input_blocks.0.0.weight"

model_names = {
    "SD15": "sd15",
    "SD20": "sd20",
    "SD21UnclipL": "sd21",
    "SD21UnclipH": "sd21",
    "SDXLRefiner": "sdxl",
    "SDXL": "sdxl",
    "SSD1B": "ssd1b",
    "SVD_img2vid": "svd",
}


class FakeTensor(NamedTuple):
    shape: tuple

    @staticmethod
    def from_dict(d):
        try:
            return FakeTensor(tuple(d["shape"]))
        except KeyError:
            return d


def config_matches(a, b):
    return all(a[k] == b[k] for k in a if k in b)


def inspect_checkpoint(filename):
    # Read header of safetensors file
    path = folder_paths.get_full_path("checkpoints", filename)
    header = comfy.utils.safetensors_header(path)
    if header:
        cfg = json.loads(header.decode("utf-8"))

        # Build a fake "state_dict" from the header info to avoid reading the full weights
        for key in cfg:
            if not key == "__metadata__":
                cfg[key] = FakeTensor.from_dict(cfg[key])

        # Reuse Comfy's model detection
        unet_config = comfy.model_detection.detect_unet_config(
            cfg, "model.diffusion_model.", "F32"
        )

        # Get input count to detect inpaint models
        if input_block := cfg.get(input_block_name, None):
            input_count = input_block.shape[1]
        else:
            input_count = 4

        # Find a matching base model depending on unet config
        matching_models = (
            model
            for model in comfy.supported_models.models
            if config_matches(model.unet_config, unet_config)
        )
        base_model = next(matching_models, None)
        if base_model is None:
            return {"base_model": "unknown"}

        base_model_name = model_names.get(base_model.__name__, "unknown")
        return {
            "base_model": base_model_name,
            "is_inpaint": base_model_name == "sd15" and input_count > 4,
            "is_refiner": base_model is comfy.supported_models.SDXLRefiner,
        }
    return {"base_model": "unknown"}


@server.PromptServer.instance.routes.get("/etn/model_info")
async def model_info(request):
    try:
        info = {
            filename: inspect_checkpoint(filename)
            for filename in folder_paths.get_filename_list("checkpoints")
        }
        return web.json_response(info)
    except Exception as e:
        return web.json_response(dict(error=str(e)), status=500)
