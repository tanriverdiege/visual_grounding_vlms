"""End-to-end example: run a grounding prompt and draw the result.

    python run_object_detection.py --config-path CONFIG --dataset DATASET \\
        --num-images N [--target TARGET [TARGET ...]] [--verbose] [--save] \\
        [--save-dir DIR]

e.g., 2 images randomly sampled from the CSXA test split, one shared target:

    python run_object_detection.py --config-path configs/qwen_3_vl_4b_instruct.yaml \\
        --dataset csxa --num-images 2 --target "C3" --save

With --save, everything from the run (the resolved config, a copy of each
original image, its annotated version, and predictions.jsonl) is written to
one new timestamped directory under --save-dir (default "runs"), e.g.
"runs/20260918_143012_482913/".
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from datetime import datetime
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
        save: bool = False,
        save_dir: str | Path = "runs"
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
        save: If True, write everything from this run -- the resolved
            config, a copy of each original image, its annotated version, and
            predictions.jsonl -- into one new timestamped directory under
            `save_dir`.
        save_dir: Base directory the timestamped run directory is created
            under. Only used when `save` is True.
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

    run_dir = None
    if save:
        # Microsecond precision, not just seconds: this function can run more
        # than once per second (e.g. a script looping over configs), and a
        # colliding folder name would silently merge/overwrite that run's
        # output into an earlier one's.
        run_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        # The fully resolved config (not just a copy of config_path), so the
        # run directory alone documents every setting -- including defaults
        # that config_path itself left unset.
        config.to_yaml(run_dir / "config.yaml")
        print("saving run to", run_dir)

    vlm = HuggingFaceVLM(config)

    originals = [vlm.load_image(str(p)) for p in image_paths]
    sizes = [image.size for image in originals]
    prompts = [get_prompt(prompt_template, target=t) for t in targets]

    outputs = vlm.batch(
        images_paths=[[str(p)] for p in image_paths],
        prompts=[[p] for p in prompts],
    )

    # Zero-padded per-image prefix, not just path.name/path.stem: several
    # source datasets (CSXA included) reuse the same filename, e.g.
    # "<image_id>/src/0/src-0.png", for every example -- saving by bare name
    # would silently overwrite each earlier image with the next one sharing
    # it.
    index_width = len(str(len(image_paths) - 1))

    records = []
    for i, (path, original, (width, height), prompt, output) in enumerate(zip(
        image_paths, originals, sizes, prompts, outputs
    )):
        if verbose:
            print("model reply:", output.text)

        boxes = parse_bounding_boxes(output.text, width, height)

        if run_dir is not None:
            prefix = f"{i:0{index_width}d}_"
            # Copied from disk rather than re-saved from `original` (already
            # decoded to RGB by vlm.load_image), so this is a byte-identical
            # copy of the actual input, not a re-encoded one.
            shutil.copy2(path, run_dir / f"{prefix}{path.name}")

            annotated_path = run_dir / f"{prefix}{path.stem}_annotated.png"
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

    if run_dir is not None and records:
        save_predictions(run_dir / "predictions.jsonl", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM grounding on one image, or a batch of images "
        "in one forward pass."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the VLM's YAML config, e.g. "
        "configs/qwen_3_vl_4b_instruct.yaml",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["csxa"],
        help="The dataset to select the images from.",
    )
    parser.add_argument(
        "--num-images",
        required=True,
        type=int,
        help="Number of images to be randomly selected from the dataset.",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=None,
        help="What to locate. One value is reused for every image; give one "
        "value per image_path to target each image separately. Omit for "
        "prompt styles that don't need a target (e.g. spine/vertebra "
        "detection).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each image's model reply.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save this run's config, original and annotated images, and "
        "predictions.jsonl into a new timestamped directory under "
        "--save-dir.",
    )
    parser.add_argument(
        "--save-dir",
        default="runs",
        help="Base directory the timestamped run directory is created "
        "under. Only used with --save. Default: runs",
    )
    args = parser.parse_args()

    # A single --target value broadcasts to every image (see
    # run_vlm_object_detection's own target handling); more than one must
    # match image_path one-to-one, which the function itself validates.
    target = args.target[0] if args.target is not None and len(args.target) == 1 else args.target

    if args.dataset == "csxa":
        import sys

        # data/ is a sibling of this file's directory (vision_language_models/),
        # not a package inside it -- Python only puts the running script's own
        # directory on sys.path, so the repo root has to be added explicitly
        # before "data.scripts.utils" is importable, regardless of cwd.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.scripts.utils import sample_csxa_image_paths

        image_paths = sample_csxa_image_paths(
            baseline_directory="/data/datasets/csxa/spinetk_baseline",
            num_examples=args.num_images,
        )

    run_vlm_object_detection(
        config_path=args.config_path,
        image_path=image_paths,
        target=target,
        verbose=args.verbose,
        save=args.save,
        save_dir=args.save_dir,
    )
