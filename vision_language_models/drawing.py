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
import math
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
_TOKEN_POINT_RE = re.compile(
    r"(?:<\|object_ref_start\|>(?P<label>.*?)<\|object_ref_end\|>)?"
    r"<\|point_start\|>\((?P<x>\d+),(?P<y>\d+)\)<\|point_end\|>"
)

# Fixed color for a box labeled "spine" (see "qwen_spine_vert_detection") --
# it marks a different kind of region (the whole spine, not one vertebra) and
# should read as one consistent color, distinct from the per-vertebra boxes.
_SPINE_COLOR = "#000000"

# Fixed color for all other (per-vertebra) boxes.
_VERTEBRA_COLOR = "#ff0000"

# Order matches `kpnames` in detection_models/SpineTK/configs/csxa.yaml, and
# the "qwen_crop_vert_corners" / "qwen_all_vert_corners" prompts' own
# schemas.
_CORNER_ORDER = ("bottom_left", "bottom_right", "top_right", "top_left")
_CORNER_COLOR = "#ff0000"
# Sized so a predicted dot at this radius comfortably nests inside a
# ground-truth hollow circle of run_object_detection.py's _GT_RADIUS (see
# that module) when a prediction lands on/near its ground truth -- see this
# constant's docstring there for the full radius relationship.
_CORNER_RADIUS = 8

# Default color for a ground-truth overlay -- draw_vertebra_corners' /
# draw_all_vertebra_corners' `color` argument, and draw_bounding_boxes'
# `color` argument (which overrides its usual spine/vertebra split). Distinct
# from both _CORNER_COLOR and _SPINE_COLOR/_VERTEBRA_COLOR so a predicted and
# a ground-truth layer stay visually distinguishable when drawn on the same
# image.
_GT_COLOR = "#39ff14"  # neon green -- more visible against an X-ray than a duller green


@dataclass
class BoundingBox:
    """One detected region, already in ORIGINAL-image pixel coordinates.

    Attributes:
        x1, y1, x2, y2: Corners in pixel coordinates of the original image
            (top-left origin), post-rescale.
        label: Free-text description of the region, if the model gave one.
        partially_visible: Whether the model flagged this region as only
            partially visible (e.g. the "qwen_all_vertabrea_detection" prompt
            style's per-vertebra "partially_visible" field). None if the
            source didn't provide this.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    partially_visible: bool | None = None


@dataclass
class KeyPoint:
    """One detected point, already in ORIGINAL-image pixel coordinates.

    Used for the "qwen_crop_vert_corners" prompt style, which asks for a
    vertebra's 4 corners (Qwen-VL's "point_2d" form) instead of a box.

    Attributes:
        x, y: Pixel coordinates of the original image (top-left origin),
            post-rescale.
        label: Free-text description of the point, if the model gave one
            (e.g. "bottom_left").
    """

    x: int
    y: int
    label: str | None = None


@dataclass
class VertebraCorners:
    """One vertebra's 4 corner points, already in ORIGINAL-image pixel coordinates.

    Used for the "qwen_all_vert_corners" prompt style, which
    (unlike "qwen_crop_vert_corners") detects every vertebra in one pass over
    an uncropped spine X-ray -- so each vertebra's corners need to stay
    grouped under it, rather than returned as one flat point list the way
    `parse_keypoints` does for a single already-isolated vertebra.

    Attributes:
        label: The vertebra's label from the model (e.g. "1", "2", ...
            top-to-bottom, the same convention as "qwen_all_vert_detection").
        bottom_left, bottom_right, top_right, top_left: This vertebra's 4
            corners, or None if the model didn't return that one.
    """

    label: str | None = None
    bottom_left: KeyPoint | None = None
    bottom_right: KeyPoint | None = None
    top_right: KeyPoint | None = None
    top_left: KeyPoint | None = None

    def corners(self) -> list[KeyPoint]:
        """This vertebra's present corners, in `_CORNER_ORDER`."""
        return [
            point
            for point in (self.bottom_left, self.bottom_right, self.top_right, self.top_left)
            if point is not None
        ]


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
            # The "qwen_all_vert_detection" / "qwen_spine_detection" prompt
            # styles wrap their boxes in {"detected": [...]} (older prompt
            # revisions used {"vertebrae": [...]}) instead of a bare list.
            for key in ("detected", "vertebrae"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            continue
        boxes = [
            BoundingBox(
                *_rescale(*item["bbox_2d"], width=width, height=height),
                label=item.get("label"),
                partially_visible=item.get("partially_visible"),
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


def _rescale_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Map a [0, 1000]-normalized point to pixel coordinates of a `width`x`height` image."""
    return round(x / 1000 * width), round(y / 1000 * height)


def _parse_json_points(text: str, width: int, height: int) -> list[KeyPoint]:
    # Mirrors _parse_json_boxes, but for the "point_2d" form the
    # "qwen_crop_vert_corners" prompt style asks for instead of "bbox_2d".
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        try:
            data = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            for key in ("detected", "vertebrae"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            continue
        points = [
            KeyPoint(
                *_rescale_point(*item["point_2d"], width=width, height=height),
                label=item.get("label"),
            )
            for item in data
            if isinstance(item, dict) and "point_2d" in item
        ]
        if points:
            return points
    return []


def _parse_token_points(text: str, width: int, height: int) -> list[KeyPoint]:
    return [
        KeyPoint(
            *_rescale_point(int(m["x"]), int(m["y"]), width=width, height=height),
            label=m["label"],
        )
        for m in _TOKEN_POINT_RE.finditer(text)
    ]


def parse_keypoints(text: str, image_width: int, image_height: int) -> list[KeyPoint]:
    """Extract points from a Qwen-VL-style point-grounding reply.

    Tries the JSON form first since it's unambiguous to parse; falls back to
    the raw special-token form only if no JSON points were found. See
    `parse_bounding_boxes`, which this mirrors for the "point_2d" form used by
    the "qwen_crop_vert_corners" and "qwen_all_vert_corners" prompt styles.

    Args:
        text: The model's decoded reply (`VLMOutput.text`).
        image_width: Width, in pixels, of the ORIGINAL image shown to the
            model (e.g. `VLMOutput.image_sizes[i][0]`).
        image_height: Height, in pixels, of that original image.

    Returns:
        One `KeyPoint` per detected point, in original-image pixel
        coordinates. Empty if neither form matched.
    """
    return _parse_json_points(text, image_width, image_height) or _parse_token_points(
        text, image_width, image_height
    )


def parse_vertebra_corners(
    text: str, image_width: int, image_height: int
) -> list[VertebraCorners]:
    """Extract per-vertebra corner points from a "qwen_all_vert_corners" reply.

    Unlike `parse_keypoints` (one already-isolated vertebra's 4 corners, with
    no need to say which vertebra they belong to), a whole-image reply can
    describe more than one vertebra. An earlier version of this parser
    expected each vertebra's corners nested under a `"corners"` object per
    detected item, matching an earlier version of the "qwen_all_vert_corners"
    prompt -- empirically, that schema made the model return "detected": []
    on every image, so the prompt (and this parser) now use Qwen-VL's native
    flat `"point_2d"` form instead (like `parse_keypoints`), with each point's
    `label` carrying a `"<vertebra_number>_<corner_name>"` compound key (e.g.
    "1_bottom_left") to say which vertebra it belongs to.

    Args:
        text: The model's decoded reply (`VLMOutput.text`).
        image_width: Width, in pixels, of the ORIGINAL image shown to the
            model.
        image_height: Height, in pixels, of that original image.

    Returns:
        One `VertebraCorners` per distinct vertebra number seen, in
        original-image pixel coordinates and in first-seen order. Empty if
        nothing parsed.
    """
    points = parse_keypoints(text, image_width, image_height)

    grouped: dict[str, dict[str, KeyPoint]] = {}
    order: list[str] = []
    for point in points:
        if not point.label or "_" not in point.label:
            continue
        vertebra_number, corner_name = point.label.split("_", 1)
        corner_name = corner_name.lower()
        if corner_name not in _CORNER_ORDER:
            continue
        if vertebra_number not in grouped:
            grouped[vertebra_number] = {}
            order.append(vertebra_number)
        grouped[vertebra_number][corner_name] = KeyPoint(point.x, point.y, label=corner_name)

    return [VertebraCorners(label=number, **grouped[number]) for number in order]


def parse_medgemma_boxes(
    text: str,
    square_side: int,
    x_offset: int = 0,
    y_offset: int = 0,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[BoundingBox]:
    """Extract bounding boxes from a MedGemma-style localization reply.

    MedGemma was trained on Chest ImaGenome anatomical bounding boxes using a
    convention that differs from Qwen-VL's in three ways `parse_bounding_boxes`
    does not handle:

    - Key is `"box_2d"`, not `"bbox_2d"`.
    - Coordinate order is `[y0, x0, y1, x1]` (y first), not `[x1, y1, x2, y2]`.
    - The model is prompted to reason before answering (Google's own
      localization notebook's prompt ends "Don't give a final answer without
      reasoning... Output the final answer in the format 'Final Answer: X'"),
      so the reply is free-text reasoning followed by one fenced ```json
      block -- unlike Qwen-VL, which is prompted to emit ONLY JSON. Taking the
      LAST fenced block (rather than trying each one, as
      `_parse_json_boxes` does) is what Google's own parsing code does, and
      avoids matching a code snippet that shows up in the reasoning itself.

    It also expects the image to have been padded to a square before being
    shown to the model (see `image_prep.pad_to_square`), since coordinates
    come back normalized to [0, 1000] relative to that square, not the
    original image. Pass the offsets `pad_to_square` returned to map back.

    Args:
        text: The model's decoded reply (`VLMOutput.text`).
        square_side: Side length, in pixels, of the square image actually
            shown to the model (`max(original_width, original_height)` when
            padded with `image_prep.pad_to_square`; just the image's own
            width/height if it was already square).
        x_offset: X offset, in pixels, at which the original image was placed
            on the square canvas. 0 if the image was already square.
        y_offset: Y offset, in pixels, at which the original image was placed
            on the square canvas. 0 if the image was already square.
        image_width: If given, clamp box edges into [0, image_width]. A
            corner the model placed in the padding can otherwise land outside
            the original image after the offset is subtracted.
        image_height: If given, clamp box edges into [0, image_height].

    Returns:
        One `BoundingBox` per detected region, in original-image pixel
        coordinates. Empty if no fenced JSON block parsed.
    """
    json_start = text.find("```json")
    json_end = text.rfind("```")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return []
    try:
        data = json.loads(text[json_start + len("```json") : json_end].strip())
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []

    def to_pixels(y0: float, x0: float, y1: float, x1: float) -> tuple[int, int, int, int]:
        px0 = x0 / 1000 * square_side - x_offset
        py0 = y0 / 1000 * square_side - y_offset
        px1 = x1 / 1000 * square_side - x_offset
        py1 = y1 / 1000 * square_side - y_offset
        if image_width is not None:
            px0, px1 = (min(max(v, 0), image_width) for v in (px0, px1))
        if image_height is not None:
            py0, py1 = (min(max(v, 0), image_height) for v in (py0, py1))
        return round(px0), round(py0), round(px1), round(py1)

    return [
        BoundingBox(*to_pixels(*item["box_2d"]), label=item.get("label"))
        for item in data
        if isinstance(item, dict) and "box_2d" in item
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


def crop_for_zoom(
    image: Image.Image, box: BoundingBox, padding_frac: float = 0.15
) -> tuple[Image.Image, int, int]:
    """Crop to `box` plus padding, for a second, higher-resolution grounding pass.

    A small object in a large image gets very few of the processor's token
    budget (e.g. one vertebra in a full-body scan). Locating it coarsely
    first, then cropping to that region and re-running grounding on JUST the
    crop, spends the same token budget on far fewer original pixels -- i.e.
    more effective resolution on the thing that matters, without raising
    `max_pixels` (and its memory cost) at all.

    Args:
        image: The ORIGINAL, full image.
        box: A coarse box (from a first grounding pass) to zoom into.
        padding_frac: Fraction of the box's width/height to pad on each side,
            so the crop doesn't clip detail right at the box edge.

    Returns:
        (crop, x_offset, y_offset) -- the cropped image, and the crop's
        top-left corner in the ORIGINAL image's pixel coordinates. Pass that
        offset to `offset_boxes` to map a second grounding pass's boxes (which
        come out relative to the crop) back to original-image coordinates.
    """
    pad_x = round((box.x2 - box.x1) * padding_frac)
    pad_y = round((box.y2 - box.y1) * padding_frac)
    x1 = max(box.x1 - pad_x, 0)
    y1 = max(box.y1 - pad_y, 0)
    x2 = min(box.x2 + pad_x, image.width)
    y2 = min(box.y2 + pad_y, image.height)
    return image.crop((x1, y1, x2, y2)), x1, y1


def offset_boxes(boxes: Iterable[BoundingBox], x_offset: int, y_offset: int) -> list[BoundingBox]:
    """Translate boxes parsed against a crop back to original-image coordinates.

    Args:
        boxes: Boxes from `parse_bounding_boxes` run on a crop (e.g. one
            returned by `crop_for_zoom`).
        x_offset: The crop's left edge in original-image pixels.
        y_offset: The crop's top edge in original-image pixels.

    Returns:
        New boxes, same order, shifted into original-image coordinates.
    """
    return [
        BoundingBox(
            b.x1 + x_offset, b.y1 + y_offset, b.x2 + x_offset, b.y2 + y_offset,
            label=b.label, partially_visible=b.partially_visible,
        )
        for b in boxes
    ]


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: str,
    width: int = 3,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    """Dashed-outline rectangle, used to mark a partially-visible box."""
    x1, y1, x2, y2 = box
    for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        (xa, ya), (xb, yb) = start, end
        length = math.hypot(xb - xa, yb - ya)
        if length == 0:
            continue
        dx, dy = (xb - xa) / length, (yb - ya) / length
        pos, draw_dash = 0.0, True
        while pos < length:
            seg_end = min(pos + (dash_len if draw_dash else gap_len), length)
            if draw_dash:
                draw.line(
                    [(xa + dx * pos, ya + dy * pos), (xa + dx * seg_end, ya + dy * seg_end)],
                    fill=color, width=width,
                )
            pos, draw_dash = seg_end, not draw_dash


def _draw_one_box(
    draw: ImageDraw.ImageDraw, box: BoundingBox, color: str, width: int
) -> None:
    coords = (box.x1, box.y1, box.x2, box.y2)
    if box.partially_visible:
        _draw_dashed_rectangle(draw, coords, color, width=width)
    else:
        draw.rectangle(coords, outline=color, width=width)


def draw_bounding_boxes(
    image: Image.Image,
    boxes: Iterable[BoundingBox],
    output_path: str | Path | None = None,
    color: str | None = None,
) -> Image.Image:
    """Draw boxes on a copy of `image`.

    A box with `partially_visible=True` (see the "qwen_all_vert_detection"
    prompt style) is drawn with a dashed outline instead of the usual solid
    outline.

    A box labeled "spine" (see "qwen_spine_vert_detection", which returns one
    spine box plus one box per vertebra in the same list) is drawn first, in a
    fixed black color (`_SPINE_COLOR`); every other (per-vertebra) box is
    drawn in a fixed red color (`_VERTEBRA_COLOR`), so the whole spine and the
    individual vertebrae stay visually distinct regardless of how many boxes
    are drawn. No labels are drawn on the image.

    Args:
        image: The ORIGINAL image the boxes' coordinates are relative to.
        boxes: Boxes to draw, as returned by `parse_bounding_boxes`.
        output_path: If given, save the annotated image there.
        color: If given, overrides the spine/vertebra color split above and
            draws every box in this one color instead -- for a ground-truth
            overlay (pass `_GT_COLOR`) that should read as one consistent
            layer regardless of label, distinguishable from a predicted
            layer drawn separately in the usual colors.

    Returns:
        A new RGB image with the boxes drawn; `image` is left untouched.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    boxes = list(boxes)  # materialize: scanned twice below (spine vs. rest)
    spine_boxes = [b for b in boxes if (b.label or "").lower() == "spine"]
    other_boxes = [b for b in boxes if (b.label or "").lower() != "spine"]

    for box in spine_boxes:
        _draw_one_box(draw, box, color or _SPINE_COLOR, width=4)
    for box in other_boxes:
        _draw_one_box(draw, box, color or _VERTEBRA_COLOR, width=3)

    if output_path is not None:
        annotated.save(output_path)
    return annotated


def draw_vertebra_corners(
    image: Image.Image,
    points: Iterable[KeyPoint],
    output_path: str | Path | None = None,
    color: str = _CORNER_COLOR,
    draw_outline: bool = True,
    radius: int = _CORNER_RADIUS,
    filled: bool = True,
    line_width: int = 3,
) -> Image.Image:
    """Draw a vertebra's 4 corner points, connected into a quadrilateral.

    Unlike `draw_bounding_boxes`, this draws whatever quadrilateral the 4
    corners actually form -- including a rotated one -- rather than forcing
    an axis-aligned rectangle, since a vertebra's true outline is often
    rotated relative to the image axes (scoliosis, patient positioning).

    Args:
        image: The image the points' coordinates are relative to (typically
            a crop already isolating one vertebra, e.g. from
            `CsxaXrayImage.crop_to_bounding_box`).
        points: Points to draw, as returned by `parse_keypoints`. Matched to
            corners by `label` (e.g. "bottom_left"), case-insensitively; the
            connecting quadrilateral is only drawn once all 4 named corners
            are present, but every point is drawn as a dot regardless.
        output_path: If given, save the annotated image there.
        color: Outline/fill color. Defaults to `_CORNER_COLOR`; pass
            `_GT_COLOR` (or call this a second time on the already-
            annotated image) to layer a ground-truth overlay in a
            distinguishable color from a predicted layer.
        draw_outline: If False, only the corner dots are drawn, not the
            connecting quadrilateral -- for a ground-truth overlay that
            should read as raw annotated points, not a derived shape.
        radius: Dot radius, in pixels. Defaults to `_CORNER_RADIUS`; pass a
            larger value on a crop that's been upscaled (e.g.
            `CsxaXrayImage.crop_to_bounding_box`'s `upscale`), so the dots
            stay proportionally visible instead of looking tiny relative to
            the enlarged image.
        filled: If False, each point is drawn as a hollow (outline-only)
            circle instead of a filled dot -- for a ground-truth overlay
            that should stay visually distinct from a filled predicted dot
            even in the same color.
        line_width: Outline thickness, in pixels, for a hollow circle
            (`filled=False`); ignored when `filled=True`. A thin default can
            get hard to see against a busy X-ray, especially at a larger
            `radius`.

    Returns:
        A new RGB image with the corner points (and, if `draw_outline` and
        all 4 named corners are present, the connecting quadrilateral)
        drawn; `image` is left untouched.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    points = list(points)
    by_label = {(p.label or "").lower(): p for p in points}
    ordered = [by_label[name] for name in _CORNER_ORDER if name in by_label]

    if draw_outline and len(ordered) == len(_CORNER_ORDER):
        draw.polygon([(p.x, p.y) for p in ordered], outline=color, width=3)

    for p in points:
        bbox = (p.x - radius, p.y - radius, p.x + radius, p.y + radius)
        if filled:
            draw.ellipse(bbox, fill=color)
        else:
            draw.ellipse(bbox, outline=color, width=line_width)

    if output_path is not None:
        annotated.save(output_path)
    return annotated


def draw_all_vertebra_corners(
    image: Image.Image,
    vertebrae: Iterable[VertebraCorners],
    output_path: str | Path | None = None,
    color: str = _CORNER_COLOR,
    draw_outline: bool = True,
    radius: int = _CORNER_RADIUS,
    filled: bool = True,
    line_width: int = 3,
) -> Image.Image:
    """Draw every vertebra's corner quadrilateral on a copy of `image`.

    Applies `draw_vertebra_corners`' drawing convention to each vertebra
    independently, for the "qwen_all_vert_corners" prompt style's
    whole-image, multi-vertebra replies.

    Args:
        image: The ORIGINAL (uncropped) image the points are relative to.
        vertebrae: Vertebrae to draw, as returned by `parse_vertebra_corners`.
        output_path: If given, save the annotated image there.
        color: Outline/fill color; see `draw_vertebra_corners`.
        draw_outline: If False, only draw each vertebra's corner dots, not
            its connecting quadrilateral; see `draw_vertebra_corners`.
        radius: Dot radius, in pixels; see `draw_vertebra_corners`.
        filled: If False, draw hollow (outline-only) circles; see
            `draw_vertebra_corners`.
        line_width: Hollow-circle outline thickness; see
            `draw_vertebra_corners`.

    Returns:
        A new RGB image with every vertebra's corners drawn; `image` is left
        untouched.
    """
    annotated = image.convert("RGB").copy()
    for vertebra in vertebrae:
        annotated = draw_vertebra_corners(
            annotated, vertebra.corners(),
            color=color, draw_outline=draw_outline, radius=radius,
            filled=filled, line_width=line_width,
        )

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
