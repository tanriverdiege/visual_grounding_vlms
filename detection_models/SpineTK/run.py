"""Trains and evaluates a Detectron2 keypoint R-CNN model on SpineTK vertebrae.

Splits the baseline dataset into train/val/test, registers the splits as
Detectron2 dataset catalogs, trains a keypoint R-CNN model, and (optionally)
runs inference on a single image as a smoke test. Every setting lives in a
YAML config file (see configs/csxa.yaml) rather than being passed as flags -
same convention as vision_language_models/run_grounding.py.

Usage:
  python run.py <config.yaml>

Example:
  python run.py configs/csxa.yaml
"""

import logging
import sys

from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger
from inference import eval_img, inference
from setup_catalogs import setup_catalogs
from spinetk_config import SpineTKConfig
from split import make_split
from train import train
from visualize import visualize_sample

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    config = SpineTKConfig.from_yaml(sys.argv[1])
    setup_logger()

    x_train, x_val, x_test = make_split(config.baseline_directory)

    vert_metadata = setup_catalogs(config, x_train, x_val, x_test)

    if config.visualize:
        visualize_sample(
            DatasetCatalog.get(f"vert{config.version}_train"), vert_metadata
        )

    train(config)

    if not config.eval_image:
        logger.info("No eval_image given, skipping the inference smoke test.")
        return

    predictor = inference(config)
    eval_img(predictor, config.eval_image, config.eval_output)


if __name__ == "__main__":
    main()
