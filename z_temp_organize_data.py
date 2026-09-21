import json
import os
import shutil

# The flat Roboflow export (raw images + a single COCO annotations file).
# NOTE: this is NOT yet in SpineTK's expected "baseline-directory" layout -
# it's just the source we convert from.
SOURCE_DIRECTORY = "/data/datasets/xray_sample/train"
ANNOTATIONS_FILE = os.path.join(SOURCE_DIRECTORY, "_annotations.coco.json")

# The SpineTK-format baseline directory we are building.
OUTPUT_BASELINE_DIRECTORY = "/data/datasets/xray_sample/spinetk_baseline"

with open(ANNOTATIONS_FILE) as f:
    coco = json.load(f)

images_by_id = {im["id"]: im for im in coco["images"]}

anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

metadata = {}

for image_id, image_info in images_by_id.items():
    src_file = os.path.join(SOURCE_DIRECTORY, image_info["file_name"])
    ext = os.path.splitext(image_info["file_name"])[1]

    image_dir = os.path.join(OUTPUT_BASELINE_DIRECTORY, str(image_id), "src", "0")
    os.makedirs(image_dir, exist_ok=True)

    dst_file = os.path.join(image_dir, f"src-0{ext}")
    shutil.copy2(src_file, dst_file)

    annotations = []
    for ann in anns_by_image.get(image_id, []):
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x + w, y + h]  # COCO stores [x,y,w,h]; SpineTK expects XYXY_ABS
        annotations.append({
            "bbox": bbox_xyxy,
            "bbox_mode": 0,  # BoxMode.XYXY_ABS
            "category_id": 0,  # single class ("vertebrae"), 0-indexed per setup_catalogs.py
            # NOTE: this dataset only has 2 keypoints (start/end) per vertebra, not the
            # 6 keypoints (corners + edge midpoints) SpineTK's train.py/get_dicts.py expect.
            # Passed through as-is for now - training config will need reconciling later.
            "keypoints": ann["keypoints"],
        })

    record = {
        "file_name": dst_file,
        "height": image_info["height"],
        "width": image_info["width"],
        "image_id": str(image_id),
        "annotations": annotations,
    }

    with open(os.path.join(image_dir, "src-0.json"), "w") as out:
        json.dump(record, out)

    # Placeholder stratifying variable - no hardware/other metadata available from Roboflow export.
    metadata[str(image_id)] = {"hardware": False}

with open(os.path.join(OUTPUT_BASELINE_DIRECTORY, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"Wrote {len(images_by_id)} image folders to {OUTPUT_BASELINE_DIRECTORY}")
