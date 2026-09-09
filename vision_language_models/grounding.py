"""Parse, rescale, save, and draw Qwen-VL-style grounding boxes.

Qwen-VL models (Qwen2-VL, Qwen3-VL) are trained to emit bounding boxes
normalized to [0, 1000] relative to the WHOLE ORIGINAL image, in one of two
text forms:

- JSON, the form their own grounding cookbook prompts for and the more
  reliable one to parse: `[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]`,
  optionally fenced in a ```json block.
- The raw special-token form the model falls back to unprompted:
  `<|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`.

Rescaling only ever needs the ORIGINAL image's width/height -- never
`image_grid_thw` or any other processor-internal resized size -- because the
[0, 1000] coordinates are already relative to the whole original image.
Source: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TOKEN_BOX_RE = re.compile(
    r"(?:<\|object_ref_start\|>(?P<label>.*?)<\|object_ref_end\|>)?"
    r"<\|box_start\|>\((?P<x1>\d+),(?P<y1>\d+)\),\((?P<x2>\d+),(?P<y2>\d+)\)<\|box_end\|>"
)

# Distinct, high-contrast colors, cycled if there are more boxes than colors.
_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
)


@dataclass
class BoundingBox:
    """One detected region, already in ORIGINAL-image pixel coordinates.

    Attributes:
        x1, y1, x2, y2: Corners in pixel coordinates of the original image
            (top-left origin), post-rescale.
        label: Free-text description of the region, if the model gave one.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None


def _rescale(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Map [0, 1000]-normalized corners to pixel corners of a `width`x`height` image."""
    return (
        round(x1 / 1000 * width),
        round(y1 / 1000 * height),
        round(x2 / 1000 * width),
        round(y2 / 1000 * height),
    )


def _parse_json_boxes(text: str, width: int, height: int) -> list[BoundingBox]:
    # Try every fenced block first (the common case when the model was
    # prompted for JSON), then the whole reply unfenced as a last resort.
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        try:
            data = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        boxes = [
            BoundingBox(
                *_rescale(*item["bbox_2d"], width=width, height=height),
                label=item.get("label"),
            )
            for item in data
            if isinstance(item, dict) and "bbox_2d" in item
        ]
        if boxes:
            return boxes
    return []


def _parse_token_boxes(text: str, width: int, height: int) -> list[BoundingBox]:
    return [
        BoundingBox(
            *_rescale(
                int(m["x1"]), int(m["y1"]), int(m["x2"]), int(m["y2"]),
                width=width, height=height,
            ),
            label=m["label"],
        )
        for m in _TOKEN_BOX_RE.finditer(text)
    ]


def parse_bounding_boxes(text: str, image_width: int, image_height: int) -> list[BoundingBox]:
    """Extract bounding boxes from a Qwen-VL-style grounding reply.

    Tries the JSON form first since it's unambiguous to parse; falls back to
    the raw special-token form only if no JSON boxes were found.

    Args:
        text: The model's decoded reply (`VLMOutput.text`).
        image_width: Width, in pixels, of the ORIGINAL image shown to the
            model (e.g. `VLMOutput.image_sizes[i][0]`).
        image_height: Height, in pixels, of that original image.

    Returns:
        One `BoundingBox` per detected region, in original-image pixel
        coordinates. Empty if neither form matched.
    """
    return _parse_json_boxes(text, image_width, image_height) or _parse_token_boxes(
        text, image_width, image_height
    )


def draw_bounding_boxes(
    image: Image.Image,
    boxes: Iterable[BoundingBox],
    output_path: str | Path | None = None,
) -> Image.Image:
    """Draw boxes on a copy of `image`.

    Args:
        image: The ORIGINAL image the boxes' coordinates are relative to.
        boxes: Boxes to draw, as returned by `parse_bounding_boxes`.
        output_path: If given, save the annotated image there.

    Returns:
        A new RGB image with the boxes drawn; `image` is left untouched.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for i, box in enumerate(boxes):
        color = _PALETTE[i % len(_PALETTE)]
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        if box.label:
            draw.text((box.x1 + 4, max(box.y1 - 12, 0)), box.label, fill=color)
    if output_path is not None:
        annotated.save(output_path)
    return annotated


def save_predictions(path: str | Path, records: Iterable[dict]) -> None:
    """Append prediction records to a JSONL file, one line per sample.

    Keeps only what's needed to redraw or re-analyze later -- decoded text,
    source image path, image size, and parsed boxes -- not the raw generation
    tensors (`VLMOutput.sequences`, `.scores`, `.logits`). Those are large and
    reconstructable by rerunning the model on the same input if ever needed,
    so keeping them here would turn a run's output from kilobytes into
    gigabytes for no benefit.

    A typical record:
        {"image_path": ..., "prompt": ..., "raw_text": VLMOutput.text,
         "image_size": [w, h], "boxes": [asdict(b) for b in boxes]}

    Args:
        path: JSONL file to append to; created if it doesn't exist.
        records: JSON-serializable dicts, one per sample.
    """
    with Path(path).open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_predictions(path: str | Path) -> list[dict]:
    """Read back records written by `save_predictions`.

    Args:
        path: The JSONL file to read.

    Returns:
        One dict per non-blank line, in file order.
    """
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    # Small smoke test: end-to-end on a canned model reply, no GPU needed.
    # Run from vision_language_models/: python grounding.py
    reply = (
        '```json\n[{"bbox_2d": [100, 200, 400, 600], "label": "cat"}]\n```'
    )
    w, h = 1000, 1000
    parsed = parse_bounding_boxes(reply, w, h)
    print("parsed:", parsed)

    img = Image.new("RGB", (w, h), "white")
    out = draw_bounding_boxes(img, parsed, output_path="/tmp/grounding_smoke_test.png")
    print("saved annotated image to /tmp/grounding_smoke_test.png, size", out.size)

    save_predictions(
        "/tmp/grounding_smoke_test.jsonl",
        [{"image_path": "example.png", "raw_text": reply, "image_size": [w, h],
          "boxes": [asdict(b) for b in parsed]}],
    )
    print("round-trip:", load_predictions("/tmp/grounding_smoke_test.jsonl"))
