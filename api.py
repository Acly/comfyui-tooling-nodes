from __future__ import annotations
from aiohttp import web
from typing import NamedTuple
from pathlib import Path
import json
import traceback
import re
import logging

from comfy import model_detection, supported_models
import comfy.utils
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


def inspect_diffusion_model(filename: str, prefix: str | None, model_type: str):
    try:
        # Read header of safetensors file
        path = folder_paths.get_full_path(model_type, filename)
        header = comfy.utils.safetensors_header(path)
        if header:
            cfg = json.loads(header.decode("utf-8"))

            # Build a fake "state_dict" from the header info to avoid reading the full weights
            for key in cfg:
                if not key == "__metadata__":
                    cfg[key] = FakeTensor.from_dict(cfg[key])

            # Reuse Comfy's model detection
            if prefix is None:
                prefix = model_detection.unet_prefix_from_state_dict(cfg)
            try:  # latest ComfyUI takes 2 args
                unet_config = model_detection.detect_unet_config(cfg, prefix)
            except TypeError as e:  # older ComfyUI versions take 3 args
                raise TypeError(f"{e} when calling detect_unet_config - old version of ComfyUI?")

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
        traceback.print_exc()
        return {"base_model": "unknown", "error": f"Failed to detect base model: {e}"}


def inspect_models(model_type: str):
    try:
        prefix = "" if model_type in ("unet", "diffusion_models") else None
        info = {
            filename: inspect_diffusion_model(filename, prefix, model_type)
            for filename in folder_paths.get_filename_list(model_type)
        }
        return web.json_response(info)
    except Exception as e:
        traceback.print_exc()
        return web.json_response(dict(error=str(e)), status=500)


def has_invalid_folder_name(folder_name: str):
    valid_names = list(folder_paths.folder_names_and_paths.keys())
    if folder_name not in valid_names:
        return web.json_response(
            dict(error=f"Invalid folder path, must be one of {', '.join(valid_names)}"),
            status=400,
        )
    return None


def has_invalid_filename(filename: str):
    if not filename.lower().endswith((".sft", ".safetensors")):
        return web.json_response(dict(error="File extension must be .safetensors"), status=400)
    if not filename or not filename.strip() or len(filename) > 255:
        return web.json_response(dict(error="Invalid filename"), status=400)
    if any(char in filename for char in ["..", "/", "\\", "\n", "\r", "\t", "\0"]):
        return web.json_response(dict(error="Invalid filename"), status=400)
    if filename.startswith(".") or not re.match(r"^[a-zA-Z0-9_\-. ]+$", filename):
        return web.json_response(dict(error="Invalid filename"), status=400)
    return None


_server: server.PromptServer | None = getattr(server.PromptServer, "instance", None)
if _server is not None:

    @_server.routes.get("/api/etn/model_info/{folder_name}")
    async def model_info(request: web.Request):
        folder_name = request.match_info.get("folder_name", "checkpoints")
        if error := has_invalid_folder_name(folder_name):
            return error
        return inspect_models(folder_name)

    @_server.routes.get("/api/etn/model_info")
    async def api_model_info(request):
        return inspect_models("checkpoints")

    @_server.routes.get("/etn/model_info")
    async def api_model_info(request):
        return inspect_models("checkpoints")

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

    @_server.routes.put("/api/etn/upload/{folder_name}/{filename}")
    async def upload(request: web.Request):
        folder_name = request.match_info.get("folder_name", "")
        if error := has_invalid_folder_name(folder_name):
            return error

        filename = request.match_info.get("filename", "")
        if error := has_invalid_filename(filename):
            return error

        try:
            if folder_paths.get_full_path(folder_name, filename) is not None:
                return web.json_response(dict(status="cached"), status=200)

            folder = Path(folder_paths.folder_names_and_paths[folder_name][0][0])
            total_size = int(request.headers.get("Content-Length", "0"))
            logging.info(f"Uploading {filename} ({total_size/(1024**2):.1f} MB) to {folder} folder")

            with open(folder / filename, "wb") as f:
                async for chunk, _ in request.content.iter_chunks():
                    f.write(chunk)

            return web.json_response(dict(status="success"), status=201)
        except Exception as e:
            return web.json_response(dict(error=str(e)), status=500)
