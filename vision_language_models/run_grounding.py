"""End-to-end example: run a grounding prompt and draw the result.

    python run_grounding.py <config.yaml> <image_path> "<what to find>"

e.g.:

    python run_grounding.py configs/qwen_3_vl_4b_instruct.yaml \\
        ../data/example_image5.png "the main subject"

Writes `<image_stem>_annotated.png` next to the input image and appends a
record to `predictions.jsonl` in the current directory.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from grounding import draw_bounding_boxes, parse_bounding_boxes, save_predictions
from vlm import HuggingFaceVLM
from vlm_config import VLMConfig

# Qwen-VL grounds reliably when explicitly asked for this JSON shape; left
# unprompted it more often falls back to the raw <|box_start|> token form
# (parse_bounding_boxes still handles that, but it's less consistent).
_PROMPT_TEMPLATE = (
    'Locate {target} in the image. Output ONLY JSON in this exact form, '
    'with coordinates normalized to 0-1000: '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}]'
)


def run_grounding(config_path: str, image_path: str, target: str) -> None:
    vlm = HuggingFaceVLM(VLMConfig.from_yaml(config_path))

    prompt = _PROMPT_TEMPLATE.format(target=target)
    output = vlm(images_paths=[image_path], prompts=[prompt])
    print("model reply:", output.text)

    width, height = output.image_sizes[0]
    boxes = parse_bounding_boxes(output.text, width, height)
    print("parsed boxes:", boxes)

    image_path = Path(image_path)
    annotated_path = image_path.with_name(f"{image_path.stem}_annotated.png")
    draw_bounding_boxes(vlm.load_image(str(image_path)), boxes, output_path=annotated_path)
    print("annotated image saved to", annotated_path)

    save_predictions(
        "predictions.jsonl",
        [
            {
                "image_path": str(image_path),
                "prompt": prompt,
                "raw_text": output.text,
                "image_size": [width, height],
                "boxes": [asdict(b) for b in boxes],
            }
        ],
    )


if __name__ == "__main__":
    run_grounding(
        config_path= "configs/qwen_3_vl_4b_instruct.yaml",
        image_path= "../data/example_scan.jpeg",
        target= "top most vertabrea"
    )
