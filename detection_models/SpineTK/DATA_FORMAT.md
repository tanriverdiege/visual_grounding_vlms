# Preparing a dataset for SpineTK

How the SpineTK keypoint R-CNN pipeline finds images and annotations on disk, the exact contents of every JSON file it reads, and the rules a new dataset has to follow to load correctly.

Written from the code as of 1 Oct 2026. The working example throughout is the converted CSXA dataset at `/data/datasets/csxa/spinetk_baseline`.

## What to produce

A new dataset needs four things before `python run.py <config>.yaml` can train on it:

1. **One folder per image**, named with a number (for example `0001035`), all inside a single *baseline directory*.
2. **An image and a JSON record** in each folder, at the fixed paths `src/0/src-0.png` and `src/0/src-0.json`.
3. **A `metadata.json`** at the root of the baseline directory, with one key per image folder.
4. **A config YAML** (a copy of `configs/csxa.yaml`) whose `baseline_directory` and `kpnames` match the data.

Existing converters to copy from, both in the repo root: `z_temp_organize_csxa_data.py` (LabelMe point annotations) and `z_temp_organize_data.py` (a Roboflow COCO export).

## How the code finds data

The only path the code is given is `baseline_directory` from the config. Everything else is discovered from it:

1. **`run.py` → `SpineTKConfig.from_yaml()`**: loads the YAML. Unknown keys are an error. `num_keypoints` is always `len(kpnames)`.
2. **`split.make_split(baseline_directory)`**: reads `metadata.json` into a pandas DataFrame and splits its *keys* into train / val / test.
3. **`setup_catalogs.setup_catalogs()`**: registers three Detectron2 datasets, `vert{version}_train`, `_val` and `_test`, with `keypoint_names = kpnames`.
4. **`get_dicts.get_dicts()`** (called lazily when Detectron2 first asks for a split): walks the folders in `baseline_directory`, keeps those whose ID is in the split, loads `src-0.json` (and `augs/` for train/val), and returns the records as-is.
5. **`train.py` → `DefaultTrainer`**: Detectron2 opens each record's `file_name` and trains on its `annotations`.

`compute_keypoint_error.py` and `plot_predictions.py` repeat the same split and registration steps, then read a split back with `DatasetCatalog.get()`. So the data format below also applies to evaluation.

## Directory layout

```
spinetk_baseline/                 # baseline_directory in the config
├── metadata.json                 # required
├── 0001035/                      # one folder per image; name must be all digits
│   ├── src/
│   │   └── 0/                    # always "0"
│   │       ├── src-0.json        # required, exact name
│   │       └── src-0.png         # any name/format; the JSON's file_name points at it
│   └── augs/                     # optional: pre-made augmented copies
│       ├── 1/
│       │   ├── aug-1.json        # name must match its folder: augs/<n>/aug-<n>.json
│       │   └── aug-1.png
│       └── 2/ …
├── 0002035/
└── …
```

- **The JSON paths are hard-coded.** `get_dicts.py` opens exactly `<id>/src/0/src-0.json` and `<id>/augs/<n>/aug-<n>.json`. Any other name is ignored. If an image folder has no `src/0/`, the code prints `ERROR on <id>` and skips it.
- **The image path is not hard-coded.** The pixels are loaded from the `file_name` field inside the JSON, not from where the JSON sits. Keeping the image next to the JSON is a convention only.
- **Non-folders at the root are ignored**, so `metadata.json` and any notes or scripts there are harmless.
- **Folders not listed in `metadata.json` are skipped silently.** They never reach any split.
- Our datasets so far (CSXA, the Roboflow sample) have no `augs/` folders. Training uses the `src/0` images only.

## metadata.json

A JSON object keyed by image-folder name. CSXA's version:

```json
{
  "0001035": {"hardware": false},
  "0002035": {"hardware": false},
  …
}
```

**Only the keys matter.** They are the list of images that exist and get split. The values are loaded into DataFrame columns but never used. The upstream README calls them "stratifying variables", but `split.py` does not stratify, so `{"hardware": false}` is just a placeholder. Any object works as the value.

> [!WARNING]
> **Image IDs: read this before naming folders.**
> `split.py` reads the file with `pd.read_json(…, orient="index")`, and pandas converts the keys. `get_dicts.py` then matches folders with `int(folder_name) in split_ids`. If the converted keys aren't plain integers, **nothing matches and every split is empty, with no error**. Tested with pandas 2.3.3, the version in the `spinetk` env:
>
> | Keys in metadata.json | pandas turns them into | Result |
> |---|---|---|
> | `"0001035"`, `"4177161"` | ints `1035`, `4177161` | Works. Leading zeros are fine because `int("0001035") == 1035`. |
> | `"1"`, `"2"`, … | ints | Works. |
> | `"1700000000"`, … | Timestamps (2023-11-14 …) | **Empty splits.** Numeric keys ≥ 31,536,000 are read as Unix times. |
> | `"img_1"`, `"patient-A"` | strings | **Breaks.** `int("img_1")` raises a `ValueError`. |
>
> Safest choice: number the images `1 … N` (zero-padding is optional) and keep a separate mapping file from these IDs back to the source filenames. Also avoid two folders that are the same number, like `7` and `007`. Both match the same ID.

## Per-image JSON (`src-0.json`, `aug-n.json`)

Each file is one [Detectron2 standard dataset dict](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts). `get_dicts.py` passes it to Detectron2 almost untouched, adding only a top-level `bbox_mode`. A real CSXA record, trimmed to one of its five vertebrae:

```jsonc
{
  "file_name": "/data/datasets/csxa/spinetk_baseline/0001035/src/0/src-0.png",
  "height": 1014,
  "width": 802,
  "image_id": "0001035",
  "annotations": [
    {
      "bbox": [260.0, 337.0, 344.0, 444.0],
      "bbox_mode": 0,
      "category_id": 0,
      "keypoints": [260.0, 444.0, 2,  344.0, 420.0, 2,  329.0, 337.0, 2,  266.0, 364.0, 2],
      "vertebra_label": "C3"
    }
    // … C4, C5, C6, C7
  ]
}
```

### Image-level fields

| Field | Contents |
|---|---|
| `file_name` | Path to the image, read by Detectron2 and `cv2.imread`. Use an **absolute path**, because relative paths resolve against whatever directory you run the script from. If you move the dataset, rewrite this field. |
| `height`, `width` | Image size in pixels. Detectron2 checks them against the loaded image and raises an error on a mismatch. |
| `image_id` | A unique string per record, augmented copies included. It shows up in the `image_id` column of `compute_keypoint_error.py`'s CSV. |
| `annotations` | One entry per vertebra (per object instance). An empty list is allowed, but Detectron2 drops images with no annotations from training (`FILTER_EMPTY_ANNOTATIONS` is on by default). |

### Annotation fields

| Field | Contents |
|---|---|
| `bbox` | `[x_min, y_min, x_max, y_max]` in absolute pixels. The converters use the tight box around the instance's keypoints. Note that COCO exports store `[x, y, w, h]`, so convert them. |
| `bbox_mode` | `0`, which is `BoxMode.XYXY_ABS`. It is **required on every annotation**. The top-level value `get_dicts.py` adds is not the one Detectron2 reads. |
| `category_id` | `0`. There's a single class, `"vertebrae"` (`num_classes: 1`). |
| `keypoints` | A flat list of `[x, y, v]` triples, exactly `3 × len(kpnames)` numbers long, in `kpnames` order. `v` follows the COCO convention: `2` = labeled and visible, `1` = labeled but occluded, `0` = not labeled (set x and y to 0; the keypoint loss ignores it). |
| `vertebra_label` | Optional and ignored by training. `compute_keypoint_error.py` uses it for the CSV's `vertebra` column, falling back to `gt0`, `gt1`, … when it's missing. |

How the example's flat `keypoints` list maps to `kpnames` (image coordinates, y grows downward):

| Index | `kpnames` entry | x, y, v |
|---|---|---|
| 0 | `bottom_left` | 260, 444, 2 |
| 1 | `bottom_right` | 344, 420, 2 |
| 2 | `top_right` | 329, 337, 2 |
| 3 | `top_left` | 266, 364, 2 |

Every image must use the same keypoint order. The order you pick is defined entirely by your converter plus the `kpnames` list in the config, and nothing else in the code knows what the names mean. One exception: the prediction-plotting scripts color points by index (red, orange, green, blue).

## Loading rules in get_dicts.py

- **One wrong keypoint count drops the whole image, silently.** If any annotation's `keypoints` isn't exactly `3 × len(kpnames)` long, the entire record is discarded, including its valid vertebrae, and nothing is logged. If a vertebra is missing corners, either leave it out (as the CSXA converter does) or keep it with `v = 0` for the missing points.
- **Augmented copies are loaded for train and val only.** Test always uses `src/0` alone. Note that val gets augmented copies too, which makes it bigger and easier than test.
- **Split membership is by folder.** Every augmented copy of an image goes to the same split as its source, so augmentations can't leak into test.
- Every folder name is printed while loading. The long list of IDs in the log is normal.

## Train / val / test split

`split.py` takes 20% of the metadata keys as test, then 10/70 of the remainder as val. That's roughly 70 / 10 / 20 overall, using `random_state=100`. For CSXA's 4,963 images: **train 3,402 · val 568 · test 993**.

- **The split is fixed but depends on the key order in `metadata.json`.** Re-running gives the same split. Reordering or adding keys reshuffles it, so regenerating the file changes your test set.
- **Splitting is per folder, not per patient.** If your dataset has several images of the same patient, give them separate folders and the split can put one patient in both train and test. Avoiding that means changing `split.py` to group by patient.
- **Val isn't evaluated during training.** `train.py` sets `DATASETS.TEST` to val but registers no evaluator. To score val, run `compute_keypoint_error.py --split val`.

## Config for a new dataset

Copy `configs/csxa.yaml` and change at least these:

```yaml
baseline_directory: /data/datasets/<new>/spinetk_baseline
kpnames:            # must match the order your converter writes keypoints in
  - bottom_left
  - bottom_right
  - top_right
  - top_left
version: <new>        # registers vert<new>_train/_val/_test, distinct from CSXA's vertv1_*
output_dir: ./output_<new>
model_weights: output_<new>/model_final.pth   # used by eval / plot scripts
oks_sigmas: null      # or one value per kpnames entry
```

> [!WARNING]
> **Use a new `output_dir`.** `resume: true` is the default. Pointed at an `output_dir` that already holds a checkpoint, training continues from that checkpoint, at its iteration count, instead of starting from the COCO weights. With the CSXA directory, you'd be fine-tuning the CSXA model, and if it already reached `iters` it would stop immediately.

If your dataset has a different number of keypoints, only `kpnames` changes. `NUM_KEYPOINTS`, the length checks in `get_dicts.py` and the default `oks_sigmas` all follow from it.

## Known issues to decide on

These aren't data-format problems, but they affect anything trained through this pipeline:

- **Horizontal flips swap left and right labels.** Detectron2 flips half of the training images by default (`INPUT.RANDOM_FLIP = "horizontal"`). `setup_catalogs.py` registers `keypoint_flip_map=[]`, so after a flip the point stored as `bottom_left` is on the right side of the vertebra. The model therefore sees inconsistent left/right labels for half its training data. Two possible fixes: register `keypoint_flip_map=[("bottom_left", "bottom_right"), ("top_left", "top_right")]`, or set `cfg.INPUT.RANDOM_FLIP = "none"` in `train.py`. The existing CSXA model was trained with this behavior.
- **`visualize: true` crashes.** `visualize.py` calls `cv2.imwrite(image, filename)` with its arguments reversed. Keep it `false`, or use `plot_predictions.py` to look at data.
- **Images are resized on input.** Detectron2's defaults resize the shorter side to 800 px (capped at 1333 px on the longer side) for both training and inference. Predictions come back in original-image coordinates, so the JSON stays in original pixels.

## Check before training

Because the two worst failures (empty splits, dropped images) produce no error, it's worth running a check after converting. This script makes the same `metadata.json` call as `split.py` and validates every record. It passes on CSXA (4,963 records, 0 problems) and flags every record when given the wrong keypoint count.

```python
"""Checks a SpineTK baseline directory before training.

Usage: python check_baseline.py /data/my_dataset/spinetk_baseline 4
(the second argument is len(kpnames) from your config)
"""
import io, json, os, sys

import pandas as pd

root, num_kp = sys.argv[1], int(sys.argv[2])
meta = json.load(open(f"{root}/metadata.json"))

# Same call split.py makes; anything but an int index means empty splits.
index = pd.read_json(io.StringIO(json.dumps(meta)), orient="index").index
if index.dtype.kind != "i":
    sys.exit(f"metadata keys parse as {index.dtype}, not int: every split would be empty")

folders = [x for x in os.listdir(root) if os.path.isdir(f"{root}/{x}")]
print(f"{len(meta)} metadata keys, {len(folders)} folders")
print("non-numeric folder names:", [x for x in folders if not x.isdigit()][:5])
print("folders missing from metadata:", sorted(set(folders) - set(meta))[:5])
print("metadata keys with no folder:", sorted(set(meta) - set(folders))[:5])

ids, bad = set(), 0
for x in folders:
    jsons = [f"{root}/{x}/src/0/src-0.json"]
    if os.path.isdir(f"{root}/{x}/augs"):
        jsons += [f"{root}/{x}/augs/{a}/aug-{a}.json" for a in os.listdir(f"{root}/{x}/augs")]
    for path in jsons:
        if not os.path.exists(path):
            print("missing", path)
            bad += 1
            continue
        d = json.load(open(path))
        problems = []
        if not os.path.exists(d["file_name"]):
            problems.append("file_name not found")
        if d["image_id"] in ids:
            problems.append("duplicate image_id")
        ids.add(d["image_id"])
        for a in d["annotations"]:
            if len(a["keypoints"]) != num_kp * 3:
                problems.append(f"{len(a['keypoints'])} keypoint values")
            if a.get("bbox_mode") != 0:
                problems.append("bbox_mode != 0")
            x0, y0, x1, y1 = a["bbox"]
            if not (x0 < x1 and y0 < y1):
                problems.append("bbox not x0<x1, y0<y1")
        if problems:
            bad += 1
            if bad <= 10:
                print(path, sorted(set(problems)))
print(f"{len(ids)} records checked, {bad} with problems")
```

Run it with the `spinetk` env's Python. Then run `python split.py` (after editing its `BASELINE_DIRECTORY`) to confirm the three split sizes aren't zero.

---

Source files: `split.py`, `setup_catalogs.py`, `return_dicts.py`, `get_dicts.py`, `train.py`, `spinetk_config.py`, `configs/csxa.yaml`, and the converter `z_temp_organize_csxa_data.py`.
