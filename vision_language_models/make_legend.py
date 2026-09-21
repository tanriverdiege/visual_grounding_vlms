"""Generates a legend image for draw_bounding_boxes'/draw_vertebra_corners'
(and friends') output.

Those functions in drawing.py color and shape predicted boxes and
ground-truth points by what they mean, but that convention isn't visible on
the annotated images themselves -- this renders it as a standalone reference
image instead.

Usage: python make_legend.py [output_path]  (default: ../examples/bbox_legend.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from drawing import _CORNER_COLOR, _GT_COLOR, _SPINE_COLOR, _VERTEBRA_COLOR

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Rendered at 3x and saved with matching DPI metadata, so the legend stays
# the same physical size but comes out sharp (crisp text/lines) instead of
# soft when viewed at 100% or printed.
_SCALE = 3
_DPI = 96 * _SCALE

_FONT_SIZE = 18 * _SCALE
_SWATCH_SIZE = (70 * _SCALE, 44 * _SCALE)
_PADDING = 5 * _SCALE
_GROUP_GAP = 15 * _SCALE  # horizontal gap between entries
_LINE_WIDTH = 3 * _SCALE
_DOT_RADIUS = 10 * _SCALE

# (color, label, swatch shape) -- "box" matches draw_bounding_boxes' outlined
# rectangle; "dot" a filled predicted corner; "ring" a hollow ground-truth
# corner (see _draw_ground_truth_corners in run_object_detection.py) --
# separate entries, each unambiguously labeled, rather than one combined
# swatch (which left it unclear which color/shape was which).
_ENTRIES = [
    (_SPINE_COLOR, "Spine Bounding Box", "box"),
    (_VERTEBRA_COLOR, "Vertebra Bounding Box", "box"),
    (_CORNER_COLOR, "Predicted Corner", "dot"),
    (_GT_COLOR, "Ground Truth Corner", "ring"),
]


def make_legend(output_path: str | Path) -> None:
    """Render draw_bounding_boxes'/draw_vertebra_corners' color and shape
    convention to `output_path`."""
    try:
        font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    swatch_w, swatch_h = _SWATCH_SIZE
    height = _PADDING * 2 + swatch_h
    width = _PADDING

    x = _PADDING
    image = Image.new("RGB", (1, 1), "white")  # placeholder, resized below
    draw = ImageDraw.Draw(image)

    # First pass to measure label widths, so the canvas can be sized exactly.
    entry_widths = []
    for _, label, _shape in _ENTRIES:
        label_w = draw.textlength(label, font=font)
        entry_widths.append(label_w + _PADDING + swatch_w)
    width = _PADDING + sum(entry_widths) + _GROUP_GAP * (len(_ENTRIES) - 1) + _PADDING

    image = Image.new("RGB", (round(width), height), "white")
    draw = ImageDraw.Draw(image)

    y = _PADDING
    for (color, label, shape), entry_w in zip(_ENTRIES, entry_widths):
        label_w = entry_w - _PADDING - swatch_w
        text_y = y + swatch_h // 2 - _FONT_SIZE // 2
        draw.text((x, text_y), label, fill="black", font=font)

        box_x = x + label_w + _PADDING
        if shape == "box":
            box = (box_x, y, box_x + swatch_w, y + swatch_h)
            draw.rectangle(box, outline=color, width=_LINE_WIDTH)
        else:
            cx, cy = box_x + swatch_w // 2, y + swatch_h // 2
            bbox = (cx - _DOT_RADIUS, cy - _DOT_RADIUS, cx + _DOT_RADIUS, cy + _DOT_RADIUS)
            if shape == "dot":
                draw.ellipse(bbox, fill=color)
            else:  # "ring"
                draw.ellipse(bbox, outline=color, width=_LINE_WIDTH)
        x += entry_w + _GROUP_GAP

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=(_DPI, _DPI))
    print(f"saved legend to {output_path}")


if __name__ == "__main__":
    default_output = Path(__file__).resolve().parent.parent / "examples" / "bbox_legend.png"
    make_legend(sys.argv[1] if len(sys.argv) > 1 else default_output)
