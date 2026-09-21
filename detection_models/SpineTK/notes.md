SpineTK trains a keypoint-detection model (built on Facebook's Detectron2 library) to find vertebrae in MR/CT/X-ray spine images and mark 6 landmark points per vertebra (the four corners + midpoints of top/bottom edges)

Under the hood it's just Detectron2's standard Keypoint R-CNN
— a Mask R-CNN-style architecture repurposed for keypoints instead of masks — fine-tuned on a custom vertebrae dataset.

They fine-tune — not train from scratch (train.py line 12)
That line pulls down Detectron2's official pretrained COCO keypoint-detection checkpoint (a ResNet-50-FPN backbone trained on the full COCO keypoints dataset — humans, not spines) 
from Facebook's model zoo, and uses it as the starting point for training.

