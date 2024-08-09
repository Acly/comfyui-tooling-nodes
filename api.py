from __future__ import annotations
from aiohttp import web
from typing import NamedTuple
import json

import comfy.utils
from comfy import supported_models
from comfy import model_detection
import folder_paths
import server

from .translation import available_languages, translate

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
    "Stable_Cascade_B": "cascade-b",
    "Stable_Cascade_C": "cascade-c",
    "SD3": "sd3",
    "AuraFlow": "aura-flow",
    "HunyuanDiT": "hunyuan-dit",
    "HunyuanDiT1": "hunyuan-dit",
    "Flux": "flux",
    "FluxSchnell": "flux-schnell",
}


class FakeTensor(NamedTuple):
    shape: tuple

    @staticmethod
    def from_dict(d):
        try:
            return FakeTensor(tuple(d["shape"]))
        except KeyError:
            return d


def inspect_checkpoint(filename):
    try:
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
            prefix = model_detection.unet_prefix_from_state_dict(cfg)
            unet_args = [cfg, prefix, "F32"]
            try:  # latest ComfyUI takes 2 args
                unet_config = model_detection.detect_unet_config(*unet_args[:-1])
            except TypeError as e:  # older ComfyUI versions take 3 args
                unet_config = model_detection.detect_unet_config(*unet_args)

            # Get input count to detect inpaint models
            if input_block := cfg.get(input_block_name, None):
                input_count = input_block.shape[1]
            else:
                input_count = 4

            # Find a matching base model depending on unet config
            base_model = model_detection.model_config_from_unet_config(unet_config)
            if base_model is None:
                return {"base_model": "unknown"}

            base_model_class = base_model.__class__
            base_model_name = model_names.get(base_model_class.__name__, "unknown")
            return {
                "base_model": base_model_name,
                "is_inpaint": base_model_name in ["sd15", "sdxl"] and input_count > 4,
                "is_refiner": base_model_class is supported_models.SDXLRefiner,
            }
        return {"base_model": "unknown"}
    except Exception as e:
        return {"base_model": "unknown", "error": f"Failed to detect base model: {e}"}


if _server := getattr(server.PromptServer, "instance", None):

    @_server.routes.get("/etn/model_info")
    async def model_info(request):
        try:
            info = {
                filename: inspect_checkpoint(filename)
                for filename in folder_paths.get_filename_list("checkpoints")
            }
            return web.json_response(info)
        except Exception as e:
            return web.json_response(dict(error=str(e)), status=500)

    @_server.routes.get("/api/etn/model_info")
    async def api_model_info(request):
        return await model_info(request)

    @_server.routes.get("/api/etn/languages")
    async def languages(request):
        try:
            result = [dict(name=name, code=code) for code, name in available_languages()]
            return web.json_response(result)
        except Exception as e:
            return web.json_response(dict(error=str(e)), status=500)

    @_server.routes.get("/api/etn/translate/{lang}/{text}")
    async def translate_text(request):
        try:
            language = request.match_info.get("lang", "en")
            text = request.match_info.get("text", "")
            result = translate(f"lang:{language} {text}")
            return web.json_response(result)
        except Exception as e:
            return web.json_response(dict(error=str(e)), status=500)
