"""Prompt templates for the grounding scripts in this folder.

Centralizes the prompt wording so `run_grounding.py` and
`run_grounding_zoom.py` (and anything else that grounds a VLM) pull from one
place instead of keeping their own near-duplicate copies.
"""

from __future__ import annotations

# Qwen-VL grounds reliably when explicitly asked for this JSON shape; left
# unprompted it more often falls back to the raw <|box_start|> token form
# (parse_bounding_boxes still handles that, but it's less consistent).
QWEN_PROMPT_TEMPLATE = (
    'Locate {target} in the image. Output ONLY JSON in this exact form, '
    'with coordinates normalized to 0-1000: '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}]'
)

# For use once the image has already been cropped down to a single vertebra
# (e.g. via CsxaXrayImage.crop_to_bounding_box) -- there is no `target` to
# describe since the crop already isolates it.
QWEN_PROMPT_TEMPLATE_CROPPED_IMAGE = (
    'This is a cropped X-ray image of a vertebra. Locate its '
    'bounding box in this cropped image. Output ONLY JSON in this exact form, '
    'with coordinates normalized to 0-1000: '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}]'
)

# MedGemma's own localization notebook's exact prompt -- see
# https://github.com/Google-Health/medgemma/blob/main/notebooks/cxr_anatomy_localization_with_hugging_face.ipynb
# It differs from Qwen-VL's in convention (box_2d key, [y0, x0, y1, x1] order)
# and in expecting free-text reasoning before a final fenced JSON block, not
# JSON-only output; `parse_medgemma_boxes` matches this exact shape. The model
# was trained on Chest ImaGenome's ~29 anatomical regions on frontal chest
# X-rays (lungs, clavicles, spine, cardiac silhouette, mediastinum, hilar
# structures, costophrenic angles, ...) -- targets outside that vocabulary
# (e.g. "spinal cord", which is soft tissue and invisible on a plain
# radiograph) won't localize reliably no matter how the prompt is phrased.
MEDGEMMA_PROMPT_TEMPLATE = """Instructions:
The following user query will require outputting bounding boxes.
The format of bounding boxes coordinates is [y0, x0, y1, x1] where
(y0, x0) must be top-left corner and (y1, x1) the bottom-right corner.
This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000],
meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150.
You MUST output a single parseable json list of objects enclosed into
```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}},
{{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and
sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {target}? Don't give a final answer without reasoning.
Output the final answer in the format "Final Answer: X" where X is a JSON list of objects.
The object needs a "box_2d" and "label" key. Answer:"""

# Detects every vertebra in one pass over a whole (uncropped) spine X-ray,
# rather than the single-target QWEN_PROMPT_TEMPLATE. No `target` to fill in.
QWEN_VERT_DETECTION_PROMPT_TEMPLATE = """Analyze the provided X-ray and 
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

QWEN_SPINE_DETECTION_PROMPT_TEMPLATE = """Analyze the provided X-ray and 
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

QWEN_SPINE_VERT_DETECTION_PROMPT_TEMPLATE = """Analyze the provided X-ray in two steps.

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

# Lookup by style name instead of importing each constant directly.
PROMPTS = {
    "qwen": QWEN_PROMPT_TEMPLATE,
    "qwen_cropped": QWEN_PROMPT_TEMPLATE_CROPPED_IMAGE,
    "qwen_all_vert_detection": QWEN_VERT_DETECTION_PROMPT_TEMPLATE,
    "qwen_spine_detection": QWEN_SPINE_DETECTION_PROMPT_TEMPLATE,
    "qwen_spine_vert_detection": QWEN_SPINE_VERT_DETECTION_PROMPT_TEMPLATE,
    "medgemma": MEDGEMMA_PROMPT_TEMPLATE,
}

def get_prompt(style: str, **kwargs: str) -> str:
    """
    Returns the prompt template registered under `style`, with any of
    `kwargs` filled in that the template actually has a placeholder for.
    Placeholders not present in the template (e.g. "target" for a prompt like
    "qwen_spine_detection" that doesn't need one) are silently ignored rather
    than raising, so callers don't need to know which styles are parameterized.
    Args:
        style (str): One of the keys in `PROMPTS`, e.g. "qwen", "qwen_cropped",
            "qwen_spine_detection", "medgemma".
        **kwargs: Values to substitute for any "{name}" placeholders the
            template contains, e.g. target="C3".
    Returns:
        str: The prompt, with matching placeholders substituted.
    """
    try:
        template = PROMPTS[style]
    except KeyError:
        raise KeyError(
            f"{style!r} is not a known prompt style; available: {sorted(PROMPTS)}"
        ) from None

    present_kwargs = {name: value for name, value in kwargs.items() if f"{{{name}}}" in template}
    if not present_kwargs:
        return template
    return template.format(**present_kwargs)
