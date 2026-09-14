"""Runs inference on a SpineTK split and plots ground-truth vs. predicted
keypoints on each image, for visual inspection.

Ground truth is drawn as hollow circles, predictions as filled circles - both
using the same per-keypoint-index color (inference.KEYPOINT_COLORS), so a
hollow and filled circle of the same color mark the same keypoint, and the
gap between them is that keypoint's visual error. No IoU matching is needed
here (unlike compute_keypoint_error.py) since ground truth and predictions
are just drawn independently on top of each other.

Usage:
  python plot_predictions.py <config.yaml> [--split test] [--limit 10] [--output-dir plots] [--seed N]

Example:
  python plot_predictions.py configs/csxa.yaml --split test --limit 20
  python plot_predictions.py configs/csxa.yaml --limit 5 --seed 42   # reproducible random sample
"""

import argparse
import os
import random

import cv2
from detectron2.data import DatasetCatalog

from inference import KEYPOINT_COLORS, inference
from setup_catalogs import setup_catalogs
from spinetk_config import SpineTKConfig
from split import make_split


def draw_ground_truth(im, keypoints_flat: list[float]) -> None:
  """Draws one vertebra's ground-truth keypoints as hollow circles, in place.

  Args:
    im: BGR image array, modified in place.
    keypoints_flat: [x, y, v, x, y, v, ...] in config.kpnames order.
  """
  for i in range(len(keypoints_flat) // 3):
    x, y = keypoints_flat[i * 3], keypoints_flat[i * 3 + 1]
    color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
    cv2.circle(im, (round(x), round(y)), radius=7, color=color, thickness=2)


def draw_predictions(im, keypoints_nested: list[list[float]]) -> None:
  """Draws one instance's predicted keypoints as filled circles, in place.

  Args:
    im: BGR image array, modified in place.
    keypoints_nested: [[x, y, score], ...] in config.kpnames order (e.g.
      `instances.pred_keypoints[i].tolist()`).
  """
  for i, (x, y, _score) in enumerate(keypoints_nested):
    color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
    cv2.circle(im, (round(x), round(y)), radius=4, color=color, thickness=-1)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("config", help="Path to a SpineTKConfig YAML file.")
  parser.add_argument(
      "--split", choices=["train", "val", "test"], default="test",
      help="Which registered split to plot images from.",
  )
  parser.add_argument(
      "--limit", type=int, default=10, help="Number of images to plot.",
  )
  parser.add_argument(
      "--output-dir", default="prediction_plots", help="Directory to write annotated images to.",
  )
  parser.add_argument(
      "--seed", type=int, default=None,
      help="Random seed for --limit's sample. Omit for a different random "
      "sample each run; set it for a reproducible one.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = SpineTKConfig.from_yaml(args.config)

  x_train, x_val, x_test = make_split(config.baseline_directory)
  setup_catalogs(config, x_train, x_val, x_test)
  dataset_dicts = DatasetCatalog.get(f"vert{config.version}_{args.split}")

  rng = random.Random(args.seed)
  sample_size = min(args.limit, len(dataset_dicts))
  dataset_dicts = rng.sample(dataset_dicts, sample_size)
  print(f"Plotting a random sample of {sample_size} image(s) from the '{args.split}' split"
        f" (seed={args.seed}).")

  predictor = inference(config)
  os.makedirs(args.output_dir, exist_ok=True)

  for record in dataset_dicts:
    im = cv2.imread(record["file_name"])
    if im is None:
      print(f"WARNING: could not read {record['file_name']}, skipping")
      continue

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    for ann in record["annotations"]:
      draw_ground_truth(im, ann["keypoints"])
    for keypoints in instances.pred_keypoints:
      draw_predictions(im, keypoints.tolist())

    out_path = os.path.join(args.output_dir, f"{record['image_id']}.png")
    cv2.imwrite(out_path, im)
    print(f"  wrote {out_path} ({len(record['annotations'])} ground truth, {len(instances)} predicted)")

  print(f"\nDone. Hollow circles = ground truth, filled circles = predictions;"
        " matching colors mark the same keypoint.")


if __name__ == "__main__":
  main()
