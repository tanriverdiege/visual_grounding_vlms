from detectron2.data import MetadataCatalog, DatasetCatalog
from return_dicts import return_dicts
import pandas as pd

from spinetk_config import SpineTKConfig

def keypoint_flip_map(kpnames):
  # Pairs every "*left*" keypoint with its "*right*" partner, e.g.
  # bottom_left <-> bottom_right. Detectron2 flips half the training images
  # horizontally; without this map the label stored as "left" would end up on
  # the right side of the vertebra in those images.
  pairs = []
  for name in kpnames:
    partner = name.replace("left", "right")
    if partner != name and partner in kpnames:
      pairs.append((name, partner))
  return pairs


def setup_catalogs(
      config: SpineTKConfig,
      train: pd.DataFrame,
      val: pd.DataFrame,
      test: pd.DataFrame,
    ):

  for d in ["train", "val", "test"]:
      DatasetCatalog.register(f"vert{config.version}_" + d,
        lambda d=d: return_dicts(d, train, val, test, config.baseline_directory, config.num_keypoints)
      )
      MetadataCatalog.get(f"vert{config.version}_" + d).set(thing_classes=["vertebrae"],
                                          keypoint_names=config.kpnames,
                                          keypoint_flip_map=keypoint_flip_map(config.kpnames))
  vert_train_metadata = MetadataCatalog.get(f"vert{config.version}_train")
  return vert_train_metadata
