"""Filename categories from taxonomy; unknown names require acoustic analysis."""
import re
import unicodedata
from autoedit.audio.taxonomy import filename_label


def _fold(text):
    text = unicodedata.normalize("NFKD", text.casefold().replace("?", "d"))
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def category_from_stem(stem):
    label = filename_label(stem)
    return (label, 1.0) if label else ("UNCLEAR", 0.0)
