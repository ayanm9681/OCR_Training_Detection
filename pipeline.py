import easyocr
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass
import functools


@dataclass
class TextRegion:
    bbox: list
    text: str
    confidence: float


@functools.lru_cache(maxsize=1)
def _get_reader(languages: tuple) -> easyocr.Reader:
    return easyocr.Reader(list(languages), gpu=False)


def run_ocr(img: Image.Image, languages: tuple = ("en",)) -> list[TextRegion]:
    reader = _get_reader(languages)
    arr = np.array(img.convert("RGB"))
    raw = reader.readtext(arr)
    return [
        TextRegion(
            bbox=[[int(x), int(y)] for x, y in bbox],
            text=text,
            confidence=float(conf),
        )
        for bbox, text, conf in raw
    ]


def draw_results(img: Image.Image, regions: list[TextRegion], conf_threshold: float = 0.0) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for region in regions:
        if region.confidence < conf_threshold:
            continue
        pts = [(int(x), int(y)) for x, y in region.bbox]
        color = _conf_color(region.confidence)
        draw.polygon(pts, outline=color)
        label = f"{region.text} ({region.confidence:.0%})"
        draw.text((pts[0][0], max(0, pts[0][1] - 12)), label, fill=color)
    return out


def _conf_color(conf: float) -> tuple:
    if conf >= 0.8:
        return (0, 200, 0)
    if conf >= 0.5:
        return (255, 165, 0)
    return (220, 0, 0)


def results_to_text(regions: list[TextRegion], conf_threshold: float = 0.0) -> str:
    return "\n".join(r.text for r in regions if r.confidence >= conf_threshold)
