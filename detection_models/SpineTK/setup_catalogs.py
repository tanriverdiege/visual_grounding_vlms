from detectron2.data import MetadataCatalog, DatasetCatalog
from return_dicts import return_dicts
import pandas as pd

def setup_catalogs(
      version: str,
      train: pd.DataFrame,
      val: pd.DataFrame,
      test: pd.DataFrame,
      kpnames: list[str],
      baseline_directory: str
    ):
  
  num_keypoints = len(kpnames)
  for d in ["train", "val", "test"]:
      DatasetCatalog.register(f"vert{version}_" + d,
        lambda d=d: return_dicts(d, train, val, test, baseline_directory, num_keypoints)
      )
      MetadataCatalog.get(f"vert{version}_" + d).set(thing_classes=["vertebrae"],
                                          keypoint_names=kpnames,
                                          keypoint_flip_map=[])
  vert_train_metadata = MetadataCatalog.get(f"vert{version}_train")
  return vert_train_metadata