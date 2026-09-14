"""Runs a trained SpineTK keypoint R-CNN model on a single image and saves
a visualization of the predicted corner keypoints (as colored dots, no boxes
or labels). If a sibling src-N.json annotation file is found next to the
image, also saves a matching ground-truth visualization for comparison.

Example:
  python visualize_prediction.py \
      --image /data/datasets/csxa/spinetk_baseline/1096029/src/0/src-0.png \
      --model-weights output/model_final.pth \
      --output example_prediction.png
"""

import argparse
import json
import os

import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

from inference import KEYPOINT_COLORS, eval_img, inference

# Must match the kpnames/order used in run.py and get_dicts.py's expected
# keypoint count, and the order z_temp_organize_csxa_data.py wrote keypoints
# in (KEYPOINT_ORDER there).
KEYPOINT_NAMES = ["bottom_left", "bottom_right", "top_right", "top_left"]


def draw_ground_truth(annotation_path: str, output_path: str) -> None:
  """Draws ground-truth corner keypoints (dots only) from a src-N.json record.

  Args:
    annotation_path: Path to the src-N.json / aug-N.json record produced by
      z_temp_organize_csxa_data.py, containing "file_name" and "annotations".
    output_path: Where to write the annotated image.
  """
  with open(annotation_path) as f:
    record = json.load(f)

  im = cv2.imread(record["file_name"])
  for annotation in record["annotations"]:
    keypoints = annotation["keypoints"]  # flat [x0, y0, v0, x1, y1, v1, ...]
    for i in range(0, len(keypoints), 3):
      x, y = keypoints[i], keypoints[i + 1]
      color = KEYPOINT_COLORS[(i // 3) % len(KEYPOINT_COLORS)]
      cv2.circle(im, (round(x), round(y)), radius=5, color=color, thickness=-1)
  cv2.imwrite(output_path, im)


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--image", required=True, help="Path to the image to run inference on.")
  parser.add_argument(
      "--model-weights", default="output/model_final.pth", help="Path to trained model weights."
  )
  parser.add_argument("--score-thresh", type=float, default=0.8, help="Inference score threshold.")
  parser.add_argument(
      "--output", default="example_prediction.png", help="Where to write the annotated image."
  )
  parser.add_argument(
      "--annotation-json",
      default=None,
      help="Path to the ground-truth src-N.json/aug-N.json for --image. Defaults to "
      "src-0.json in the same directory as --image, if present. Pass --no-ground-truth "
      "to skip this lookup entirely.",
  )
  parser.add_argument(
      "--no-ground-truth",
      action="store_true",
      help="Skip writing a ground-truth visualization even if an annotation file is found.",
  )
  parser.add_argument(
      "--ground-truth-output",
      default=None,
      help="Where to write the ground-truth visualization. Defaults to "
      "'<output>_gt.<ext>' next to --output.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  setup_logger()

  # eval_img only needs a metadata object for its call signature -- drawing no
  # longer uses class/keypoint names from it, so this is just a placeholder.
  metadata = MetadataCatalog.get("vert_predict_only")
  metadata.set(thing_classes=["vertebrae"], keypoint_names=KEYPOINT_NAMES, keypoint_flip_map=[])

  predictor = inference("predict_only", args.model_weights, args.score_thresh)
  eval_img(metadata, predictor, args.image, args.output)
  print(f"Wrote predicted-keypoints visualization to {args.output}")

  if args.no_ground_truth:
    return

  annotation_path = args.annotation_json
  if annotation_path is None:
    candidate = os.path.join(os.path.dirname(args.image), "src-0.json")
    annotation_path = candidate if os.path.exists(candidate) else None

  if annotation_path is None:
    print("No ground-truth annotation found next to --image; skipping.")
    return

  if args.ground_truth_output is not None:
    gt_output = args.ground_truth_output
  else:
    stem, ext = os.path.splitext(args.output)
    gt_output = f"{stem}_gt{ext}"

  draw_ground_truth(annotation_path, gt_output)
  print(f"Wrote ground-truth visualization to {gt_output}")


if __name__ == "__main__":
  main()
