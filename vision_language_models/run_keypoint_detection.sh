#!/bin/bash
set -e

# Run from vision_language_models/ (relative paths below assume that cwd).

# Single image, single target, save the annotated output + predictions.jsonl.
python run_object_detection.py \
    --config-path configs/qwen_3_vl_4b_instruct_lora_vert_corners.yaml \
    --dataset csxa \
    --num-images 10 \
    --save \
    --verbose \
    --save-dir "/home/ege/visual_grounding_vlms/vision_language_models/runs" \
    
