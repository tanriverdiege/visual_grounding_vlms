"""End-to-end example: run a grounding prompt and draw the result.

Not a CLI -- edit the call at the bottom of this file (config path, image
path(s), target) and run:

    python run_object_detection.py

Writes `<image_stem>_annotated.png` next to each input image and appends a
record per image to `predictions.jsonl` in the current directory, when
`save=True`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from grounding import (
    draw_bounding_boxes,
    parse_bounding_boxes,
    save_predictions,
)
from prompts import get_prompt
from vlm import HuggingFaceVLM
from vlm_config import VLMConfig


def run_vlm_object_detection(
        config_path: str,
        image_path: str | list[str],
        target: str | list[str | None] | None = None,
        verbose: bool = False,
        save: bool = False
    ) -> None:
    """Run grounding on one image, or a batch of images.

    Args:
        config_path: Path to the VLM's YAML config.
        image_path: A single image path, or a list of image paths to run as
            one batch.
        target: The thing to locate. A single string is reused for every
            image when `image_path` is a list; pass a list of the same
            length as `image_path` to give each image its own target.
        verbose: Print each model reply.
        save: Write the annotated image (next to its input) and append a
            record to `predictions.jsonl` for each image.
    """
    batched = isinstance(image_path, list)
    image_paths = [Path(p) for p in image_path] if batched else [Path(image_path)]
    if not image_paths:
        raise ValueError("image_path is empty; pass at least one image")

    if isinstance(target, list):
        if not batched:
            raise ValueError(
                "target is a list but image_path is a single path; pass a "
                "list of image paths too"
            )
        targets = target
    else:
        targets = [target] * len(image_paths)
    if len(targets) != len(image_paths):
        raise ValueError(
            f"got {len(image_paths)} image path(s) but {len(targets)} "
            "target(s); they must correspond one-to-one"
        )

    config = VLMConfig.from_yaml(config_path)
    prompt_template = config.prompt_template
    if prompt_template is None:
        raise ValueError(
            f"{config_path} sets no prompt_template; add one before running "
            "(this is checked before the model loads on purpose)"
        )

    vlm = HuggingFaceVLM(config)

    originals = [vlm.load_image(str(p)) for p in image_paths]
    sizes = [image.size for image in originals]
    prompts = [get_prompt(prompt_template, target=t) for t in targets]

    outputs = vlm.batch(
        images_paths=[[str(p)] for p in image_paths],
        prompts=[[p] for p in prompts],
    )

    records = []
    for path, original, (width, height), prompt, output in zip(
        image_paths, originals, sizes, prompts, outputs
    ):
        if verbose:
            print("model reply:", output.text)

        boxes = parse_bounding_boxes(output.text, width, height)

        if save:
            annotated_path = path.with_name(f"{path.stem}_annotated.png")
            print("annotated image saved to", annotated_path)
            draw_bounding_boxes(original, boxes, output_path=annotated_path)

            records.append(
                {
                    "image_path": str(path),
                    "prompt": prompt,
                    "raw_text": output.text,
                    "image_size": [width, height],
                    "boxes": [asdict(b) for b in boxes],
                }
            )

    if records:
        save_predictions("predictions.jsonl", records)


if __name__ == "__main__":
    # data/src-0-cropped-C3-pad0.5-up3.0.png
    # src-0.png
    run_vlm_object_detection(
        config_path="configs/qwen_3_vl_4b_instruct.yaml",
        image_path="../data/src-0.png",
    )
