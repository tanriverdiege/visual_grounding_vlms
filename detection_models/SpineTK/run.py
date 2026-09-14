"""Trains and evaluates a Detectron2 keypoint R-CNN model on SpineTK vertebrae.

Splits the baseline dataset into train/val/test, registers the splits as
Detectron2 dataset catalogs, trains a keypoint R-CNN model, and (optionally)
runs inference on a single image as a smoke test.

Example:
  python run.py --baseline-directory /data/datasets/csxa/spinetk_baseline \
      --version v1 --iters 1300 --eval-image path/to/image.png
"""

import argparse
import logging

from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger
from inference import eval_img, inference
from setup_catalogs import setup_catalogs
from split import make_split
from train import train
from visualize import visualize_sample

# KEYPOINT_ORDER. Keep these in sync if the annotation schema ever changes.
KEYPOINT_NAMES = ["bottom_left", "bottom_right", "top_right", "top_left"]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "--baseline-directory",
      default="/data/datasets/csxa/spinetk_baseline",
      help="Directory of per-image folders with src/aug annotations.",
  )
  parser.add_argument(
      "--version", default="v1", help="Dataset/catalog version tag, e.g. 'v1'."
  )
  parser.add_argument(
      "--iters", type=int, default=1300, help="Number of training iterations."
  )
  parser.add_argument(
      "--resume",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Resume training from an existing checkpoint if one is found.",
  )
  parser.add_argument(
      "--visualize",
      action="store_true",
      help="Dump a couple of annotated training samples before training.",
  )
  parser.add_argument(
      "--model-weights",
      default="output/model_final.pth",
      help="Path to trained model weights, used for the eval-image step.",
  )
  parser.add_argument(
      "--score-thresh", type=float, default=0.8, help="Inference score threshold."
  )
  parser.add_argument(
      "--eval-image",
      default=None,
      help="Path to an image to run a final inference smoke test on. The "
      "eval step is skipped if this is omitted.",
  )
  parser.add_argument(
      "--eval-output",
      default="out.png",
      help="Where to write the annotated eval-image output.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  setup_logger()

  x_train, x_val, x_test = make_split(args.baseline_directory)

  vert_metadata = setup_catalogs(
      args.version,
      x_train,
      x_val,
      x_test,
      kpnames=KEYPOINT_NAMES,
      baseline_directory=args.baseline_directory,
  )

  if args.visualize:
    visualize_sample(DatasetCatalog.get(f"vert{args.version}_train"), vert_metadata)

  train(args.version, args.iters, args.resume)

  if not args.eval_image:
    logger.info("No --eval-image given, skipping the inference smoke test.")
    return

  predictor = inference(args.version, args.model_weights, args.score_thresh)
  eval_img(vert_metadata, predictor, args.eval_image, args.eval_output)


if __name__ == "__main__":
  main()
