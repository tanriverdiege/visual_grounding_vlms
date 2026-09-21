from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from drawing import (
    _CORNER_RADIUS,
    _GT_COLOR,
    KeyPoint,
    VertebraCorners,
    draw_all_vertebra_corners,
    draw_bounding_boxes,
    draw_vertebra_corners,
    parse_bounding_boxes,
    parse_keypoints,
    parse_vertebra_corners,
    save_predictions,
)
from prompts import get_prompt
from vlm import HuggingFaceVLM
from vlm_config import VLMConfig

# Dots on a cropped (and, per --crop's default, heavily upscaled)
# single-vertebra image need to be bigger than _CORNER_RADIUS to stay
# proportionally visible -- a dot sized for a normal-resolution image looks
# tiny next to a 10x-upscaled crop. Used for the prediction in that case.
_CROP_RADIUS = _CORNER_RADIUS * 3

# Ground truth's hollow circle on a CROPPED image (see
# _draw_ground_truth_corners) gets its own, bigger radius than the filled
# predicted dot uses -- sized independently from _CORNER_RADIUS/_CROP_RADIUS
# so it can be enlarged without also enlarging the prediction. line_width is
# kept small relative to that radius on purpose: making the ring thicker to
# be more visible (an earlier revision tried line_width=radius) just fills
# the circle solid, defeating the point of "hollow" -- a bigger radius with
# a modest ring width is what actually reads as a bigger hollow circle.
#
# The gap (GT_CROP_RADIUS - GT_CROP_LINE_WIDTH vs. _CROP_RADIUS) is kept
# small but positive, so a predicted dot right on top of its ground truth
# visibly nests inside the hollow ring -- like a target -- instead of
# looking disproportionately small next to it or poking outside it.
_GT_CROP_LINE_WIDTH = 6
_GT_CROP_RADIUS = _CROP_RADIUS + _GT_CROP_LINE_WIDTH + 2  # inner opening = _CROP_RADIUS + 2

# On an UNCROPPED (whole-image, several-vertebrae) reply, several ground
# truth points sit much closer together than on a single zoomed-in crop, so
# a big hollow ring (sized to nest a predicted dot, like the crop case above)
# would overlap its neighbors -- a small filled dot reads more clearly there
# than a hollow one at the same small size would.
_GT_RADIUS = 4


def _draw_ground_truth_corners(image, ground_truth, color):
    """Draws `ground_truth`'s corner points only (no quadrilateral/box).

    On an uncropped, whole-image reply (`ground_truth` is a
    `list[VertebraCorners]`), drawn as small filled dots (`_GT_RADIUS`). On a
    cropped, single-vertebra reply (`ground_truth` is a flat `list[KeyPoint]`
    instead), drawn as a bigger hollow circle (`_GT_CROP_RADIUS`/
    `_GT_CROP_LINE_WIDTH`) so it stays visually distinct from a filled
    predicted dot even in the same color -- dispatches on which shape
    `ground_truth` is, so the caller doesn't need to know or care which
    prediction style produced the image it's layering onto.
    """
    if not ground_truth:
        return image
    if isinstance(ground_truth[0], VertebraCorners):
        return draw_all_vertebra_corners(
            image, ground_truth, color=color, draw_outline=False, radius=_GT_RADIUS
        )
    return draw_vertebra_corners(
        image, ground_truth, color=color, draw_outline=False,
        radius=_GT_CROP_RADIUS, filled=False, line_width=_GT_CROP_LINE_WIDTH,
    )


def run_vlm_object_detection(
        config_path: str,
        image_path: str | list[str],
        verbose: bool = False,
        save: bool = False,
        save_dir: str | Path = "runs",
        ground_truth_corners: list[list[KeyPoint] | list[VertebraCorners]] | None = None,
    ) -> None:
    """Run grounding on one image, or a batch of images.

    Args:
        config_path: Path to the VLM's YAML config.
        image_path: A single image path, or a list of image paths to run as
            one batch.
        verbose: Print each model reply.
        save: If True, write everything from this run -- the resolved
            config, a copy of each original image, its annotated version, and
            predictions.jsonl -- into one new timestamped directory under
            `save_dir`.
        save_dir: Base directory the timestamped run directory is created
            under. Only used when `save` is True.
        ground_truth_corners: Only used with `save`. One entry per
            `image_path`, in the same coordinate frame as that image (a crop
            or the original, whichever was actually passed in) -- a flat
            `list[KeyPoint]` for one vertebra's corners, or a
            `list[VertebraCorners]` for several. Drawn on the annotated
            image as corner points only (never a box or connecting
            quadrilateral -- that's a derived shape, not the raw
            annotation) in `_GT_COLOR`, layered on top of the prediction
            regardless of whether that prediction is corners or a box.
    """
    batched = isinstance(image_path, list)
    image_paths = [Path(p) for p in image_path] if batched else [Path(image_path)]
    if not image_paths:
        raise ValueError("image_path is empty; pass at least one image")

    config = VLMConfig.from_yaml(config_path)
    if config.prompt_template is None:
        raise ValueError(
            f"{config_path} sets no prompt_template; add one before running "
            "(this is checked before the model loads on purpose)"
        )

    run_dir = None
    if save:
        run_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        config.to_yaml(run_dir / "run_model_config.yaml")
        print("saving run to", run_dir)

    vlm = HuggingFaceVLM(config)

    originals = [vlm.load_image(str(p)) for p in image_paths]
    sizes = [image.size for image in originals]
    prompts = [get_prompt(config.prompt_template)] * len(image_paths)

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

    # "qwen_crop_vert_corners" (single, already-isolated vertebra) and
    # "qwen_all_vert_corners" (every vertebra in an uncropped image) both ask
    # for corner points instead of a box, so they need their own parse/draw
    # functions rather than parse_bounding_boxes/draw_bounding_boxes; each
    # maps to its own (parse_fn, draw_fn, is_crop) triple -- is_crop picks
    # _CROP_RADIUS over the default dot size.
    corner_styles = {
        "qwen_crop_vert_corners": (parse_keypoints, draw_vertebra_corners, True),
        "qwen_all_vert_corners": (parse_vertebra_corners, draw_all_vertebra_corners, False),
    }
    corner_style = corner_styles.get(config.prompt_template)

    records = []
    for i, (path, original, (width, height), prompt, output) in enumerate(zip(
        image_paths, originals, sizes, prompts, outputs
    )):
        if verbose:
            print("model reply:", output.text)

        if corner_style is not None:
            parse_fn, draw_fn, is_crop = corner_style
            detections = parse_fn(output.text, width, height)
        else:
            detections = parse_bounding_boxes(output.text, width, height)

        if run_dir is not None:
            prefix = f"{i:0{index_width}d}_"
            # Copied from disk rather than re-saved from `original` (already
            # decoded to RGB by vlm.load_image), so this is a byte-identical
            # copy of the actual input, not a re-encoded one.
            shutil.copy2(path, run_dir / f"{prefix}{path.name}")

            annotated_path = run_dir / f"{prefix}{path.stem}_annotated.png"
            if corner_style is not None:
                # No connecting quadrilateral for a keypoint-detection
                # prediction either -- see _draw_ground_truth_corners, same
                # reasoning: the lines are a derived shape, not what the
                # model actually predicted (4 independent points).
                radius = _CROP_RADIUS if is_crop else _CORNER_RADIUS
                annotated = draw_fn(original, detections, draw_outline=False, radius=radius)
            else:
                annotated = draw_bounding_boxes(original, detections)

            gt = ground_truth_corners[i] if ground_truth_corners is not None else None
            annotated = _draw_ground_truth_corners(annotated, gt, color=_GT_COLOR)
            annotated.save(annotated_path)

            record = {
                "image_path": str(path),
                "prompt": prompt,
                "raw_text": output.text,
                "image_size": [width, height],
            }
            if corner_style is not None:
                record["keypoints"] = [asdict(d) for d in detections]
            else:
                record["boxes"] = [asdict(b) for b in detections]
            records.append(record)

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

    parser.add_argument(
            "--crop",
            action="store_true",
            help="Cropped inference on bounding box.",
        )
    parser.add_argument(
        "--vertebra-label",
        default=None,
        help="Vertebra to crop each image to, e.g. 'C3'. Required with --crop.",
    )
    args = parser.parse_args()

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

        # Ground-truth corner points for the annotated-image overlay (see
        # run_vlm_object_detection's `ground_truth_corners` arg) -- drawn as
        # points regardless of whether the prediction being compared is
        # corners or a box, so it's built the same way either way. Coordinate
        # frame follows whichever image actually gets shown to the model:
        # cropped-and-upscaled if --crop, the original CSXA image otherwise.
        ground_truth_corners = []

        if args.crop:
            from data.scripts.utils import CsxaXrayImage

            if args.vertebra_label is None:
                parser.error("--crop requires --vertebra-label (e.g. C3)")

            crop_dir = Path("examples/cropped_images")
            crop_dir.mkdir(parents=True, exist_ok=True)

            cropped_paths = []
            for path in image_paths:
                xray_image = CsxaXrayImage(path)
                cropped_image = xray_image.crop_to_bounding_box(
                    vertebra_label=args.vertebra_label,
                    upscale = 10,
                )
                # Every CSXA image is named "src-0.png"; the image_id (the
                # grandparent-of-parent directory, see
                # "<baseline_directory>/<image_id>/src/0/src-0.png") is what's
                # actually unique per sampled image, so it's used here instead
                # of the original filename to avoid collisions.
                image_id = Path(path).parent.parent.parent.name
                cropped_path = crop_dir / f"{image_id}_{args.vertebra_label}.png"
                cropped_image.save(cropped_path)
                cropped_paths.append(str(cropped_path))

                # to_crop_coordinates uses the crop just made above (it reads
                # back xray_image.last_crop_box/last_crop_upscale), so ground
                # truth lines up with the actual saved crop, not the original.
                corners = xray_image.get_corners(args.vertebra_label)
                ground_truth_corners.append([
                    KeyPoint(*xray_image.to_crop_coordinates(x, y), label=name)
                    for name, (x, y) in corners.items()
                ])

            image_paths = cropped_paths
        else:
            from data.scripts.utils import CsxaXrayImage

            for path in image_paths:
                xray_image = CsxaXrayImage(path)
                ground_truth_corners.append([
                    VertebraCorners(
                        label=vertebra_label,
                        **{
                            name: KeyPoint(x, y, label=name)
                            for name, (x, y) in xray_image.get_corners(vertebra_label).items()
                        },
                    )
                    for vertebra_label in xray_image.vertebra_labels
                ])

    run_vlm_object_detection(
        config_path=args.config_path,
        image_path=image_paths,
        verbose=args.verbose,
        save=args.save,
        save_dir=args.save_dir,
        ground_truth_corners=ground_truth_corners,
    )
