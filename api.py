import server
from aiohttp import web
import json

import comfy.utils
import comfy.supported_models
import folder_paths
import server

input_block = "model.diffusion_model.input_blocks.0.0.weight"
transformer_block = "1.transformer_blocks.0.attn2.to_k.weight"


def inspect_checkpoint(filename):
    path = folder_paths.get_full_path("checkpoints", filename)
    header = comfy.utils.safetensors_header(path)
    if header:
        cfg = json.loads(header.decode("utf-8"))
        input_count = cfg[input_block]["shape"][1]
        context_dim = next(
            v["shape"][1] for k, v in cfg.items() if k.endswith(transformer_block)
        )
        base_model = next(
            model
            for model in comfy.supported_models.models
            if model.unet_config["context_dim"] == context_dim
        )
        base_model_name = base_model.__name__.lower()
        return {
            "base_model": base_model_name[:4],
            "is_inpaint": input_count > 4,
            "is_refiner": "refiner" in base_model_name,
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
