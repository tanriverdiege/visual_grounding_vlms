import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from spinetk_config import SpineTKConfig

# Per-corner colors (BGR, for cv2), indexed to match config.kpnames order.
# Shared between predicted and ground-truth drawing so the same corner always
# gets the same color and the two are directly comparable. Cycles via
# `i % len(KEYPOINT_COLORS)` if kpnames has more than 4 entries.
KEYPOINT_COLORS = [
    (0, 0, 255),  # red
    (0, 165, 255),  # orange
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
]


def inference(config: SpineTKConfig) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.model_zoo_config))
    cfg.DATASETS.TEST = (f"vert{config.version}_val",)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = config.num_keypoints

    cfg.MODEL.WEIGHTS = config.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.score_thresh
    cfg.TEST.KEYPOINT_OKS_SIGMAS = (
        config.oks_sigmas
    )
    predictor = DefaultPredictor(cfg)
    return predictor


def eval_img(predictor, img_path, output_path):
    im = cv2.imread(img_path)
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    instances = outputs["instances"].to("cpu")
    for keypoints in instances.pred_keypoints:
        for i, (x, y, _score) in enumerate(keypoints.tolist()):
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(im, (round(x), round(y)), radius=5, color=color, thickness=-1)
    cv2.imwrite(output_path, im)
