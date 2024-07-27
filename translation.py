"""Text translation using Argos Translate.

The node takes text input and translates it to English. The text may contain any
number of language directives in the form `lang:xx` where `xx` is a two-letter
language code. Text fragments after a language directives are translated.
If the language is `en` text is passed through unmodified.
"""

import re
from functools import cache
from typing import NamedTuple


@cache
def available_languages():
    try:
        from argostranslate.package import update_package_index, get_available_packages

        update_package_index()
        list = get_available_packages()
        return [(l.from_code, l.from_name) for l in list if l.to_code == "en"]
    except ImportError:
        return [("NOT INSTALLED", "NOT INSTALLED")]


def translate_chunk(text: str, language: str):
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


def translate(text: str):
    chunks = Chunk.parse(text)
    return " ".join(translate_chunk(c.text, c.lang) for c in chunks)


class Translate:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"text": ("STRING", {"multiline": True})}}

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"

    def translate(self, text: str):
        return (translate(text),)


_lang_regex = re.compile(r"(lang:\w\w)")


class Chunk(NamedTuple):
    text: str
    lang: str

    @staticmethod
    def parse(text: str):
        languages = [code for code, name in available_languages()] + ["en"]
        chunks: list[Chunk] = []
        lang = "en"
        last = 0
        for m in _lang_regex.finditer(text):
            if m.start() > 0:
                chunks.append(Chunk(text[last : m.start()].strip(), lang))
            last = m.end()
            lang = m.group(0)[5:]
            if lang not in languages:
                raise ValueError(
                    f"Invalid language directive {m.group(0)} - {lang} is not a known language code."
                    f" Available languages: {', '.join(languages)}"
                )
        if last < len(text):
            chunks.append(Chunk(text[last:].strip(), lang))
        return [c for c in chunks if c.text != ""]


_embedding_regex = re.compile(r"(embedding:[^\s,]+)")


def _extract_embeddings(text: str):
    matches = _embedding_regex.findall(text)
    embeddings = " ".join(matches)
    if matches:
        embeddings += " "
    for m in matches:
        text = text.replace(m, "")
    return text, embeddings
