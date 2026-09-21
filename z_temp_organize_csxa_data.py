import json
import os
import shutil

# Source: LabelMe-format point annotations, one JSON per PNG, both keyed by
# the same numeric id (e.g. 4177161.png / 4177161.json).
SOURCE_JSON_DIR = "/data/datasets/csxa/datasets-JSON"
SOURCE_PNG_DIR = "/data/datasets/csxa/datasets-PNG"

# The SpineTK-format baseline directory we are building.
OUTPUT_BASELINE_DIRECTORY = "/data/datasets/csxa/spinetk_baseline"

# Vertebrae with full 4-corner annotations. C2 is deliberately excluded -
# it only has bottom_left/bottom_right/centroid in this dataset, not enough
# to build a real 6-point box without fabricating the top corners.
VERTEBRAE = ["C3", "C4", "C5", "C6", "C7"]

# 4 directly-labeled corners only - no interpolated midpoints. Kept in the
# same corner ordering run.py's 6-point kpnames used, minus 'bottom_middle'/
# 'top_middle', so this stays a strict subset rather than a different order.
KEYPOINT_ORDER = ["bottom_left", "bottom_right", "top_right", "top_left"]


def round_point(p):
    return [float(round(p[0])), float(round(p[1]))]


metadata = {}
skipped = []

json_files = sorted(f for f in os.listdir(SOURCE_JSON_DIR) if f.endswith(".json"))

for fname in json_files:
    image_id = os.path.splitext(fname)[0]  # already numeric, e.g. "4177161"

    with open(os.path.join(SOURCE_JSON_DIR, fname)) as f:
        data = json.load(f)

    # Build label -> point lookup. Keep the first occurrence of a label if
    # duplicated (one file in this dataset has a duplicate-click point).
    points_by_label = {}
    for shape in data["shapes"]:
        label = shape["label"]
        if label in points_by_label:
            continue
        points_by_label[label] = shape["points"][0]  # shape_type == "point" -> single [x, y]

    annotations = []
    incomplete_vertebrae = []
    for v in VERTEBRAE:
        needed = [f"{v} top left", f"{v} top right", f"{v} bottom left", f"{v} bottom right"]
        if not all(k in points_by_label for k in needed):
            incomplete_vertebrae.append(v)
            continue

        top_left = points_by_label[f"{v} top left"]
        top_right = points_by_label[f"{v} top right"]
        bottom_left = points_by_label[f"{v} bottom left"]
        bottom_right = points_by_label[f"{v} bottom right"]

        kp_by_name = {
            "top_left": top_left, "top_right": top_right,
            "bottom_left": bottom_left, "bottom_right": bottom_right,
        }

        keypoints_flat = []
        xs, ys = [], []
        for name in KEYPOINT_ORDER:
            x, y = round_point(kp_by_name[name])
            keypoints_flat.extend([x, y, 2])  # visibility=2: labeled and visible
            xs.append(x)
            ys.append(y)

        bbox_xyxy = [min(xs), min(ys), max(xs), max(ys)]

        annotations.append({
            "bbox": bbox_xyxy,
            "bbox_mode": 0,  # BoxMode.XYXY_ABS
            "category_id": 0,  # single class ("vertebrae")
            "keypoints": keypoints_flat,
            "vertebra_label": v,  # extra field for traceability; harmless for get_dicts.py
        })

    if incomplete_vertebrae:
        skipped.append((image_id, incomplete_vertebrae))

    src_png = os.path.join(SOURCE_PNG_DIR, f"{image_id}.png")
    image_dir = os.path.join(OUTPUT_BASELINE_DIRECTORY, image_id, "src", "0")
    os.makedirs(image_dir, exist_ok=True)

    dst_png = os.path.join(image_dir, "src-0.png")
    shutil.copy2(src_png, dst_png)

    record = {
        "file_name": dst_png,
        "height": data["imageHeight"],
        "width": data["imageWidth"],
        "image_id": image_id,
        "annotations": annotations,
    }

    with open(os.path.join(image_dir, "src-0.json"), "w") as out:
        json.dump(record, out)

    metadata[image_id] = {"hardware": False}  # placeholder stratifying var, no such info available

with open(os.path.join(OUTPUT_BASELINE_DIRECTORY, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"Wrote {len(json_files)} image folders to {OUTPUT_BASELINE_DIRECTORY}")
if skipped:
    print(f"{len(skipped)} image(s) had at least one vertebra missing full corner data (skipped that vertebra only):")
    for image_id, verts in skipped[:10]:
        print(f"  {image_id}: missing {verts}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")
