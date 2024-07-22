from weakref import ref as WeakRef
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPImageProcessor, CLIPConfig, CLIPVisionModel, PreTrainedModel
from kornia.filters import box_blur

from .nodes import to_bchw, to_bhwc


def cosine_similarity(image_embeds: Tensor, text_embeds: Tensor):
    if image_embeds.dim() == 2 and text_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(1)
    return F.cosine_similarity(image_embeds, text_embeds, dim=-1)


class CLIPSafetyChecker(PreTrainedModel):
    # https://huggingface.co/CompVis/stable-diffusion-safety-checker
    # Adapted from:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py

    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        projdim = config.projection_dim

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, projdim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, projdim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, projdim), requires_grad=False)
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, clip_input, images: Tensor, sensitivity: float):
        with torch.no_grad():
            image_batch = self.vision_model(clip_input)[1]
            image_embeds = self.visual_projection(image_batch)
            sensitivity = -0.1 + 0.14 * sensitivity

            special_cos_dist = cosine_similarity(image_embeds, self.special_care_embeds)
            special_scores_threshold = self.special_care_embeds_weights.unsqueeze(0)
            special_scores = special_cos_dist - special_scores_threshold + sensitivity

            if torch.any(special_scores > 0):
                sensitivity = sensitivity + 0.01

            cos_dist = cosine_similarity(image_embeds, self.concept_embeds)
            concept_threshold = self.concept_embeds_weights.unsqueeze(0)
            concept_scores = cos_dist - concept_threshold + sensitivity

            is_nsfw = [torch.any(concept_scores[i] > 0) for i in range(concept_scores.shape[0])]
            is_nsfw = [x.item() for x in is_nsfw]
            return self.filter_images(images, is_nsfw)

    def filter_images(self, images: Tensor, is_nsfw: list[bool]):
        if not any(is_nsfw):
            return images

        images = images.clone()
        images_to_filter = (i for i, nsfw in enumerate(is_nsfw) if nsfw)
        orig_size = images.shape[-2:]
        for idx in images_to_filter:
            filtered = images[idx].unsqueeze(0)
            filtered = F.interpolate(filtered, size=64, mode="nearest")
            filtered = box_blur(filtered, 11, separable=True)
            filtered = F.interpolate(filtered, size=orig_size, mode="bilinear")
            images[idx] = filtered.squeeze(0)
        return images


class CachedModels:
    _instance: WeakRef | None = None

    def __init__(self):
        model_dir = Path(__file__).parent / "safetychecker"
        model_file = model_dir / "model.safetensors"
        if not model_file.exists():
            self.download(
                "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/refs%2Fpr%2F41/model.safetensors",
                target=model_file,
            )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_dir)
        self.safety_checker = CLIPSafetyChecker.from_pretrained(model_dir)

    @classmethod
    def load(cls):
        models = cls._instance and cls._instance()
        if models is None:
            models = cls()
            cls._instance = WeakRef(models)
        return models

    def download(self, url: str, target: Path):
        import requests

        try:
            target_temp = target.with_suffix(".download")
            with requests.get(url, stream=True) as response:
                text = "NSFWFilter model download"
                total = int(response.headers.get("content-length", 0))
                pbar = tqdm(None, total=total, unit="b", unit_scale=True, desc=text)
                with open(target_temp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()
            target_temp.rename(target)
        except Exception as e:
            raise RuntimeError(
                f"NSFWFilter: Failed to download safety-checker model from {url} to target location {target}: {e}"
            ) from e


class NSFWFilter:
    models: CachedModels

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "check"
    CATEGORY = "external_tooling"

    def __init__(self):
        self.models = CachedModels.load()

    def check(self, image, sensitivity):
        image = to_bchw(image)
        input = self.models.feature_extractor(image, do_rescale=False, return_tensors="pt")
        filtered = self.models.safety_checker(
            images=image, clip_input=input.pixel_values, sensitivity=sensitivity
        )
        return (to_bhwc(filtered),)
