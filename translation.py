"""Text translation using Argos Translate."""

import re
from functools import cache


@cache
def available_languages():
    try:
        from argostranslate.package import update_package_index, get_available_packages

        update_package_index()
        list = get_available_packages()
        return [(l.from_code, l.from_name) for l in list if l.to_code == "en"]
    except ImportError:
        return [("NOT INSTALLED", "NOT INSTALLED")]


def translate(text: str, language: str):
    if text.strip() == "":
        return text

    target = "en"
    if language == target:
        return text

    try:
        from argostranslate.package import get_installed_packages, get_available_packages
        from argostranslate.translate import translate

        installed = get_installed_packages()
        if not any(p.from_code == language and p.to_code == target for p in installed):
            available = get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == language and p.to_code == target), None
            )
            assert pkg, f"Couldn't find package for translation from {language}"
            print("Downloading and installing translation package", pkg)
            pkg.install()

        text, embeddings = _extract_embeddings(text)
        translation = translate(text, language, target)
        return embeddings + translation

    except ImportError:
        raise ImportError(
            "Argos Translate is not installed. Please install it with `pip install argostranslate`"
        )


class Translate:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "language": ([code for code, _ in available_languages()],),
                "target": (["en"],),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"

    def translate(self, text: str, language: str, target: str):
        return (translate(text, language),)


_embedding_regex = re.compile(r"(embedding:[^\s,]+)")


def _extract_embeddings(text: str):
    matches = _embedding_regex.findall(text)
    embeddings = " ".join(matches)
    if matches:
        embeddings += " "
    for m in matches:
        text = text.replace(m, "")
    return text, embeddings
