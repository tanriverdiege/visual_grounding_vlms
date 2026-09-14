"""Shared prediction<->ground-truth matching for SpineTK eval scripts.

Detectron2 gives back an unordered set of detected instances per image, which
doesn't line up 1:1 with the ground-truth annotation list (different order,
possibly missed or extra detections) - both eval scripts need to figure out
which predicted vertebra corresponds to which labeled one before per-keypoint
error is meaningful at all.
"""

from __future__ import annotations

import torch
from detectron2.structures import Boxes, pairwise_iou


def match_predictions_to_ground_truth(
    gt_annotations: list[dict],
    pred_boxes: torch.Tensor,
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily matches predicted boxes to ground-truth annotations by IoU.

    Repeatedly takes the highest-IoU (gt, pred) pair not yet used, until no
    pair reaches iou_threshold - this is the same greedy strategy detection
    evaluation commonly uses (a real Hungarian assignment is overkill here:
    vertebra boxes within one image rarely overlap with each other).

    Args:
        gt_annotations: This image's annotation dicts, each with a 'bbox' in
            XYXY_ABS (get_dicts.py's format).
        pred_boxes: (N, 4) tensor of predicted boxes in XYXY_ABS, e.g.
            `outputs["instances"].pred_boxes.tensor`.
        iou_threshold: Minimum IoU for a pair to count as matched. Pairs
            below this are left unmatched rather than force-paired.

    Returns:
        A tuple of:
          - matches: (gt_index, pred_index, iou) for each matched pair.
          - unmatched_gt: gt_index values with no matching prediction
            (missed detections).
          - unmatched_pred: pred_index values with no matching ground truth
            (false positives).
    """
    num_gt = len(gt_annotations)
    num_pred = pred_boxes.shape[0]
    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))

    gt_boxes = Boxes(torch.tensor([a["bbox"] for a in gt_annotations], dtype=torch.float32))
    pred_boxes = Boxes(torch.as_tensor(pred_boxes, dtype=torch.float32))
    iou = pairwise_iou(gt_boxes, pred_boxes)  # (num_gt, num_pred)

    candidates = sorted(
        ((iou[g, p].item(), g, p) for g in range(num_gt) for p in range(num_pred)),
        key=lambda t: -t[0],
    )

    matches = []
    used_gt, used_pred = set(), set()
    for score, g, p in candidates:
        if score < iou_threshold:
            break
        if g in used_gt or p in used_pred:
            continue
        matches.append((g, p, score))
        used_gt.add(g)
        used_pred.add(p)

    unmatched_gt = [g for g in range(num_gt) if g not in used_gt]
    unmatched_pred = [p for p in range(num_pred) if p not in used_pred]
    return matches, unmatched_gt, unmatched_pred
