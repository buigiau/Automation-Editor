"""Broad mouth/expression classification from landmarks or a mouth ROI.

Not viseme/phoneme classification. Five usable labels plus CLOSED/NEUTRAL.
"""

from __future__ import annotations

from typing import Any


def classify_mouth(
    openness: float,
    width: float,
    roundness: float,
    smile: float,
    brow_drop: float = 0.0,
) -> tuple[str, float]:
    """Map normalized geometry to a category.

    All inputs are roughly 0..1 except smile (-1 frown .. +1 smile).
    """
    if openness < 0.12:
        return "CLOSED", min(1.0, 0.6 + (0.12 - openness) * 2)

    if smile < -0.25 and openness > 0.2:
        return "CRY", min(1.0, 0.55 + abs(smile) * 0.4)

    if smile > 0.35 and openness > 0.18 and width > 0.45:
        return "HAHA", min(1.0, 0.55 + smile * 0.4)

    # Round open mouth (O) vs tall open (A) vs wide (E)
    if roundness > 0.55 and 0.22 < openness < 0.75 and width < 0.7:
        return "O", min(1.0, 0.5 + roundness * 0.4)

    if width > 0.62 and openness < 0.45:
        return "E", min(1.0, 0.5 + width * 0.3)

    if openness > 0.35:
        return "A", min(1.0, 0.5 + openness * 0.4)

    return "NEUTRAL", 0.4


def roi_metrics(mouth_gray) -> dict[str, Any]:
    """Geometry from a grayscale mouth crop (numpy ndarray)."""
    import numpy as np

    if mouth_gray is None or mouth_gray.size == 0:
        return {
            "openness": 0.0,
            "width": 0.0,
            "roundness": 0.0,
            "smile": 0.0,
            "mean": 0.0,
        }
    img = mouth_gray.astype(np.float32)
    h, w = img.shape[:2]
    mean = float(img.mean()) / 255.0
    # Darker interior vs brighter lips: openness ~ contrast of center band
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    band = img[cy0:cy1, :]
    center = float(band.mean()) / 255.0 if band.size else mean
    openness = max(0.0, min(1.0, (mean - center) * 3.0 + (1.0 - center) * 0.3))
    # Horizontal extent of dark pixels
    thresh = img.mean() * 0.85
    dark = img < thresh
    cols = dark.any(axis=0)
    if cols.any():
        xs = np.where(cols)[0]
        width = float(xs[-1] - xs[0]) / max(1, w)
    else:
        width = 0.3
    aspect = (h * openness + 1) / (w * width + 1)
    roundness = max(0.0, min(1.0, 1.0 - abs(aspect - 1.0)))
    # Crude smile: left/right corners darker/higher than center bottom
    smile = float((img[int(h * 0.7), int(w * 0.15)] + img[int(h * 0.7), int(w * 0.85)]) / 2 - img[int(h * 0.85), int(w * 0.5)]) / 255.0
    return {
        "openness": float(openness),
        "width": float(width),
        "roundness": float(roundness),
        "smile": float(max(-1.0, min(1.0, smile * 4))),
        "mean": mean,
    }
