from __future__ import annotations

QWEN_CROPPED_VERT_BBOX_PROMPT_TEMPLATE = """Analyze the provided cropped X-ray image,
which contains a single vertebra.

Bounding-box definition:
Return the smallest axis-aligned rectangular bounding box that contains the
entire visible vertebra.

Detection requirements:
- Detect the single vertebra visible in this cropped image.
- Return exactly one bounding box.
- Return valid JSON only.

Use exactly this JSON schema:

{
  "detected": [
    {
      "label": "vertebra",
      "bbox_2d": [x_min, y_min, x_max, y_max]
    }
  ]
}"""

QWEN_CROPPED_VERTEBRA_CORNERS = """Analyze the provided cropped X-ray image,
which contains a single vertebra.

Point definition:
Identify the four corners of the vertebra body: bottom_left, bottom_right,
top_right, and top_left.

Detection requirements:
- Locate all four corners, even if a corner is only partially visible.
- Return exactly one point per corner, labeled with its corner name.
- Return valid JSON only.
- Coordinates are normalized to the range 0-1000, relative to this image's
  width and height.

Use exactly this JSON schema:

{
  "detected": [
    {"label": "bottom_left", "point_2d": [x, y]},
    {"label": "bottom_right", "point_2d": [x, y]},
    {"label": "top_right", "point_2d": [x, y]},
    {"label": "top_left", "point_2d": [x, y]}
  ]
}"""


QWEN_ALL_VERT_BBOX_PROMPT_TEMPLATE = """Analyze the provided X-ray and 
detect every visible vertebra.

Bounding-box definition:
For each vertebra, return the smallest axis-aligned rectangular bounding box
that contains the entire visible vertebra.

Detection requirements:
- Detect all clearly visible vertebrae.
- List the vertebrae from the top of the image to the bottom.
- Do not merge multiple vertebrae into one bounding box.
- Return exactly one bounding box per detected vertebra.
- Return valid JSON only.

Use exactly this JSON schema:

{
  "detected": [
    {
      "label": "1",
      "bbox_2d": [x_min, y_min, x_max, y_max],
    }
  ]
}"""

# Flat point_2d list (Qwen-VL's native point-grounding form -- the same one
# QWEN_CROPPED_VERTEBRA_CORNERS uses, which reliably returns non-empty
# results), with a "<vertebra_index>_<corner_name>" compound label to say
# which vertebra each point belongs to. An earlier version of this prompt
# asked for each vertebra's 4 corners nested under a "corners": {...} object
# per detected item; empirically that schema made the model return
# "detected": [] on every single image, presumably because it's not the flat,
# single-key-per-item shape Qwen-VL was actually trained to emit for
# grounding (see the working QWEN_ALL_VERT_BBOX_PROMPT_TEMPLATE and
# QWEN_CROPPED_VERTEBRA_CORNERS above for comparison) -- this flat form asks
# for exactly that shape instead.
QWEN_ALL_VERT_CORNERS_PROMPT_TEMPLATE = """Analyze the provided X-ray and
detect every visible vertebra's corners.

Point definition:
For each vertebra, identify its four corners: bottom_left, bottom_right,
top_right, and top_left.

Detection requirements:
- Detect all clearly visible vertebrae.
- Number the vertebrae "1", "2", "3", and so on, from the top of the image to
  the bottom.
- For each vertebra, return one point per corner, labeled
  "<vertebra_number>_<corner_name>", e.g. "1_bottom_left", "1_bottom_right",
  "1_top_right", "1_top_left", "2_bottom_left", and so on.
- Locate all four corners for each vertebra, even if a corner is only
  partially visible.
- Return valid JSON only.
- Coordinates are normalized to the range 0-1000, relative to this image's
  width and height.

Use exactly this JSON schema:

{
  "detected": [
    {"label": "1_bottom_left", "point_2d": [x, y]},
    {"label": "1_bottom_right", "point_2d": [x, y]},
    {"label": "1_top_right", "point_2d": [x, y]},
    {"label": "1_top_left", "point_2d": [x, y]},
    {"label": "2_bottom_left", "point_2d": [x, y]}
  ]
}"""

QWEN_SPINE_BBOX_PROMPT_TEMPLATE = """Analyze the provided X-ray and
detect the spine.
Bounding-box definition:
For the spine, return the smallest axis-aligned rectangular bounding box
that contains the spine.
Detection requirements:
- Detect the spine.
- Return exactly one bounding box.
- Return valid JSON only.
Use exactly this JSON schema:
{
  "detected": [
    {
      "label": "1",
      "bbox_2d": [x_min, y_min, x_max, y_max],
    }
  ]
}"""

QWEN_SPINE_VERT_BBOX_PROMPT_TEMPLATE = """Analyze the provided X-ray in two steps.
Step 1: Detect the spine and compute the smallest axis-aligned bounding box
that contains it.
Step 2: Within that spine bounding box, detect every individual visible
vertebra body and compute a separate bounding box for each one, in the same
format as the spine's bounding box.
Bounding-box definition:
Each bounding box must be the smallest axis-aligned rectangle that contains
the corresponding structure (the spine as a whole, or one vertebra body).
Detection requirements:
- Return exactly one bounding box for the spine, labeled "spine".
- Return exactly one bounding box per detected vertebra, labeled "vertebra".
- Return valid JSON only.
- Don't just return the spine bounding box, also return the vertabra body bounding boxes.
- Return at least one vertebra body bounding box.
Use exactly this JSON schema:
{
  "detected": [
    {
      "label": "spine",
      "bbox_2d": [x_min, y_min, x_max, y_max]
    },
    {
      "label": "vertebra",
      "bbox_2d": [x_min, y_min, x_max, y_max]
    }
  ]
}"""

# Whole-image, anatomically-labeled variants of the two "all vertebrae"
# prompts above, for LoRA fine-tuning on CSXA (see lora/prepare_data.py).
# CSXA only annotates C3-C7, so the generic "every visible vertebra, numbered
# top to bottom" prompts would train the model to silently skip C2 while
# still calling C3 "1"; these name exactly the vertebrae the targets contain.
# "<vertebra>_<corner>" labels still split on the first "_", so
# parse_vertebra_corners groups them the same way as the numbered form.
QWEN_C3_C7_CORNERS_PROMPT_TEMPLATE = """Analyze the provided lateral cervical
spine X-ray and locate the corners of vertebrae C3, C4, C5, C6, and C7.

Point definition:
For each vertebra body, identify its four corners: bottom_left,
bottom_right, top_right, and top_left.

Detection requirements:
- Only report C3, C4, C5, C6, and C7, from top to bottom.
- Label each point "<vertebra>_<corner_name>", e.g. "C3_bottom_left".
- Skip a vertebra entirely if it is not visible.
- Return valid JSON only.
- Coordinates are normalized to the range 0-1000, relative to this image's
  width and height.

Use exactly this JSON schema:

{
  "detected": [
    {"label": "C3_bottom_left", "point_2d": [x, y]},
    {"label": "C3_bottom_right", "point_2d": [x, y]},
    {"label": "C3_top_right", "point_2d": [x, y]},
    {"label": "C3_top_left", "point_2d": [x, y]},
    {"label": "C4_bottom_left", "point_2d": [x, y]}
  ]
}"""

QWEN_C3_C7_BBOX_PROMPT_TEMPLATE = """Analyze the provided lateral cervical
spine X-ray and locate vertebrae C3, C4, C5, C6, and C7.

Bounding-box definition:
For each vertebra body, return the smallest axis-aligned rectangular
bounding box that contains it.

Detection requirements:
- Only report C3, C4, C5, C6, and C7, from top to bottom.
- Label each box with its vertebra name, e.g. "C3".
- Skip a vertebra entirely if it is not visible.
- Return valid JSON only.
- Coordinates are normalized to the range 0-1000, relative to this image's
  width and height.

Use exactly this JSON schema:

{
  "detected": [
    {"label": "C3", "bbox_2d": [x_min, y_min, x_max, y_max]},
    {"label": "C4", "bbox_2d": [x_min, y_min, x_max, y_max]}
  ]
}"""

PROMPTS = {
    "qwen_crop_vert_bbox": QWEN_CROPPED_VERT_BBOX_PROMPT_TEMPLATE,
    "qwen_crop_vert_corners": QWEN_CROPPED_VERTEBRA_CORNERS,
    "qwen_all_vert_bbox": QWEN_ALL_VERT_BBOX_PROMPT_TEMPLATE,
    "qwen_all_vert_corners": QWEN_ALL_VERT_CORNERS_PROMPT_TEMPLATE,
    "qwen_spine_bbox": QWEN_SPINE_BBOX_PROMPT_TEMPLATE,
    "qwen_spine_vert_bbox": QWEN_SPINE_VERT_BBOX_PROMPT_TEMPLATE,
    "qwen_c3_c7_corners": QWEN_C3_C7_CORNERS_PROMPT_TEMPLATE,
    "qwen_c3_c7_bbox": QWEN_C3_C7_BBOX_PROMPT_TEMPLATE,
}

def get_prompt(style: str) -> str:
    """
    Returns the prompt template registered under `style`. None of the
    registered templates take a `target` (or any other placeholder) anymore
    -- each one is self-contained (whole-image or already-cropped-to-one-
    vertebra detection), so this is a plain lookup.
    Args:
        style (str): One of the keys in `PROMPTS`, e.g. "qwen_crop_vert_bbox",
            "qwen_all_vert_corners".
    Returns:
        str: The prompt template.
    """
    try:
        return PROMPTS[style]
    except KeyError:
        raise KeyError(
            f"{style!r} is not a known prompt style; available: {sorted(PROMPTS)}"
        ) from None
