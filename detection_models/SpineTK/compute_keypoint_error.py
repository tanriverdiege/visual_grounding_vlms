"""Runs inference on a SpineTK split and reports per-keypoint pixel error
against ground truth.

For each image, predicted vertebrae are matched to ground-truth vertebrae by
box IoU (see keypoint_matching.py), then each matched pair's keypoints are
compared index-for-index (both sides use config.kpnames order) and the
straight-line pixel distance is recorded as that keypoint's error. Missed
detections and false positives are counted separately, since a low mean
error is meaningless if half the vertebrae were never detected at all.

Each row also gets a normalized error: the x and y components are each
divided by that image's width/height before taking the distance, so it's not
dominated by whichever axis the image happens to be larger along. That makes
error comparable across images of different resolution -- a raw pixel_error
of 20px means very different things on a 500px-wide crop and a 4000px-wide
scan, but the normalized error doesn't.

Usage:
  python compute_keypoint_error.py <config.yaml> [--split test] [--csv errors.csv]

Example:
  python compute_keypoint_error.py configs/csxa.yaml --split test
"""

import argparse
import csv
import math

import cv2
from detectron2.data import DatasetCatalog

from inference import inference
from keypoint_matching import match_predictions_to_ground_truth
from setup_catalogs import setup_catalogs
from spinetk_config import SpineTKConfig
from split import make_split

CSV_FIELDS = [
    "image_id", "vertebra", "keypoint",
    "gt_x", "gt_y", "pred_x", "pred_y", "pred_score",
    "pixel_error", "normalized_error", "match_iou",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("config", help="Path to a SpineTKConfig YAML file.")
  parser.add_argument(
      "--split", choices=["train", "val", "test"], default="test",
      help="Which registered split to evaluate on.",
  )
  parser.add_argument(
      "--csv", default="keypoint_errors.csv",
      help="Where to write the per-keypoint error breakdown.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = SpineTKConfig.from_yaml(args.config)

  x_train, x_val, x_test = make_split(config.baseline_directory)
  setup_catalogs(config, x_train, x_val, x_test)
  dataset_dicts = DatasetCatalog.get(f"vert{config.version}_{args.split}")
  print(f"Evaluating on {len(dataset_dicts)} images from the '{args.split}' split.")

  predictor = inference(config)

  rows = []
  per_keypoint_errors = {name: [] for name in config.kpnames}
  per_keypoint_normalized_errors = {name: [] for name in config.kpnames}
  num_unmatched_gt = 0
  num_unmatched_pred = 0

  for record in dataset_dicts:
    im = cv2.imread(record["file_name"])
    if im is None:
      print(f"WARNING: could not read {record['file_name']}, skipping")
      continue
    height, width = im.shape[:2]

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    gt_annotations = record["annotations"]

    matches, unmatched_gt, unmatched_pred = match_predictions_to_ground_truth(
        gt_annotations, instances.pred_boxes.tensor
    )
    num_unmatched_gt += len(unmatched_gt)
    num_unmatched_pred += len(unmatched_pred)

    for gt_idx, pred_idx, iou in matches:
      gt_ann = gt_annotations[gt_idx]
      gt_kps = gt_ann["keypoints"]  # flat [x, y, v, x, y, v, ...], config.kpnames order
      pred_kps = instances.pred_keypoints[pred_idx].tolist()  # [[x, y, score], ...]
      vertebra_label = gt_ann.get("vertebra_label", f"gt{gt_idx}")

      for i, name in enumerate(config.kpnames):
        gx, gy = gt_kps[i * 3], gt_kps[i * 3 + 1]
        px, py, pscore = pred_kps[i]
        error = math.hypot(px - gx, py - gy)
        normalized_error = math.hypot((px - gx) / width, (py - gy) / height)
        per_keypoint_errors[name].append(error)
        per_keypoint_normalized_errors[name].append(normalized_error)
        rows.append({
            "image_id": record["image_id"], "vertebra": vertebra_label, "keypoint": name,
            "gt_x": gx, "gt_y": gy, "pred_x": px, "pred_y": py, "pred_score": pscore,
            "pixel_error": error, "normalized_error": normalized_error, "match_iou": iou,
        })

  with open(args.csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)

  print(f"\nWrote {len(rows)} per-keypoint error rows to {args.csv}")
  print(f"Unmatched ground-truth vertebrae (missed detections): {num_unmatched_gt}")
  print(f"Unmatched predictions (false positives): {num_unmatched_pred}")

  print("\nMean pixel error / normalized error per keypoint:")
  all_errors = []
  all_normalized_errors = []
  for name, errors in per_keypoint_errors.items():
    normalized_errors = per_keypoint_normalized_errors[name]
    if errors:
      mean_error = sum(errors) / len(errors)
      mean_normalized_error = sum(normalized_errors) / len(normalized_errors)
      print(f"  {name}: {mean_error:.2f}px / {mean_normalized_error:.4f} (n={len(errors)})")
      all_errors.extend(errors)
      all_normalized_errors.extend(normalized_errors)
    else:
      print(f"  {name}: no matched instances")

  if all_errors:
    print(
        f"\nOverall mean pixel error: {sum(all_errors) / len(all_errors):.2f}px "
        f"(n={len(all_errors)})"
    )
    print(
        f"Overall mean normalized error: "
        f"{sum(all_normalized_errors) / len(all_normalized_errors):.4f} "
        f"(n={len(all_normalized_errors)})"
    )
  else:
    print("\nNo matched instances at all - nothing to report.")


if __name__ == "__main__":
  main()
