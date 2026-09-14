import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from spinetk_config import SpineTKConfig


def train(config: SpineTKConfig) -> None:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.model_zoo_config))
    cfg.DATASETS.TRAIN = (f"vert{config.version}_train",)
    cfg.DATASETS.TEST = (f"vert{config.version}_val",)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.model_zoo_config)
    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.MAX_ITER = config.iters
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = config.num_keypoints
    cfg.OUTPUT_DIR = config.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=config.resume)
    trainer.train()
