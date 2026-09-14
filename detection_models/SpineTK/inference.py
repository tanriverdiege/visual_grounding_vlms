import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Per-corner colors (BGR, for cv2), indexed to match the KEYPOINT_NAMES order
# used throughout the project: bottom_left, bottom_right, top_right, top_left.
# Shared between predicted and ground-truth drawing so the same corner always
# gets the same color and the two are directly comparable.
KEYPOINT_COLORS = [
    (0, 0, 255),    # bottom_left  - red
    (0, 165, 255),  # bottom_right - orange
    (0, 255, 0),    # top_right    - green
    (255, 0, 0),    # top_left     - blue
]

def inference(version, filepath, thresh):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TEST = (f"vert{version}_val",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vertebrae)
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4

  cfg.MODEL.WEIGHTS = os.path.join(filepath)  # path to the model we just trained
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh   # set a custom testing threshold
  cfg.TEST.KEYPOINT_OKS_SIGMAS = [1,1,1,1]
  predictor = DefaultPredictor(cfg)
  return predictor

def eval_img(metadata, predictor, img_path, output_path):
  # metadata is unused now that we draw keypoints directly instead of going
  # through Visualizer.draw_instance_predictions (which also drew boxes and
  # "vertebrae NN%" labels); kept in the signature for call-site compatibility.
  del metadata
  im = cv2.imread(img_path)
  outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
  instances = outputs['instances'].to('cpu')
  for keypoints in instances.pred_keypoints:
    for i, (x, y, _score) in enumerate(keypoints.tolist()):
      color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
      cv2.circle(im, (round(x), round(y)), radius=5, color=color, thickness=-1)
  cv2.imwrite(output_path, im)
