# SpineTK [WIP]

![](https://i.imgur.com/3nQo0jB.png)

> **This is a modified copy of [abhisuri97/SpineTK](https://github.com/abhisuri97/SpineTK) by Abhinav Suri, cloned into this project.** All original code and rights belong to the original author; this copy is not affiliated with, endorsed by, or reviewed by them. The modifications described below were made for our own project's purposes and convenience.

This is a repository that contains the code to train a network for doing MR, CT, and X-ray image annotation (landmark annotation of six keypoints on individual vertebral bodies for vertebral height measurement). It is recommended that you run this on Google Colab. 

## Datasets used

The model in `configs/csxa.yaml` is trained on **csxa**, an open-source, publicly available cervical spine X-ray dataset with per-vertebra corner-point annotations (C2-C7) and clinical measurements (disc heights, Cobb angles, cervical slopes, etc.):

- **Dataset**: [csxa on Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=8e3b3d5e60a348ba961e19d48b881c90)

We are not the creators of this dataset - all credit and rights for the underlying images and annotations belong to its original authors. `z_temp_organize_csxa_data.py` (in the repo root, outside this folder) converts its native LabelMe-style export into this project's expected baseline-directory format; see that script for exactly how the raw dataset maps onto `configs/csxa.yaml`'s `kpnames`.

## What's different in this copy

The original repo is a research script collection, largely written for interactive/Colab use rather than being run end-to-end as-is (several of its functions had `import module` vs. `from module import function` mistakes, missing arguments, and a hardcoded 6-keypoint assumption that doesn't match every dataset). This copy keeps the original architecture and data format, but reworks the pipeline into something runnable from the command line on a real dataset:

- **Config-driven pipeline** (`spinetk_config.py`): every training/inference setting - dataset path, keypoint names, learning rate, iterations, score threshold, etc. - lives in one validated `SpineTKConfig`, loaded from a YAML file (see `configs/csxa.yaml`). `num_keypoints` is *derived* from `kpnames` everywhere rather than being a separate hardcoded number, which was a repeated source of bugs in the original code (a keypoint-count mismatch scattered across `get_dicts.py`, `train.py`, and `inference.py`).
- **`run.py <config.yaml>`** trains end-to-end from a single config file (`python run.py configs/csxa.yaml`), replacing the original's cell-by-cell Colab workflow.
- **`compute_keypoint_error.py <config.yaml>`** runs inference over a full split, IoU-matches predictions to ground-truth vertebrae (`keypoint_matching.py`), and reports per-keypoint pixel error plus missed-detection/false-positive counts, with a full per-vertebra CSV breakdown.
- **`plot_predictions.py <config.yaml>`** plots ground truth (hollow circles) against predicted keypoints (filled circles, matching colors) on a random - optionally seeded, for reproducibility - sample of images from any split.
- **`configs/detectron.yaml`** is a sanity-check config that loads the *original*, never-fine-tuned COCO-pretrained checkpoint instead of a trained one, to verify that fine-tuning is actually responsible for a model's predictions rather than the pretrained backbone alone.
- See `commands.sh` for the exact commands used for this project's own training/eval runs, and the "Environment setup" section below for reproducing the pinned dependency stack this still requires.

### Environment setup (local / conda)

This repo pins old versions of PyTorch/CUDA/Detectron2 (torch 1.8.0+cu101, detectron2 built for torch1.8/cu101) because Detectron2 never released official wheels for newer PyTorch at the time this was written. These wheels are still hosted and installable, but only for **Python 3.6–3.9** (no wheel exists for cp310+), and a couple of dependency pins are needed to keep newer resolvers from pulling in incompatible package versions.

Verified working with: Python 3.9, NVIDIA driver 550.x (CUDA 12.4 runtime), Turing-class GPUs (e.g. Quadro RTX 5000, compute capability 7.5). A CUDA 10.1-compiled wheel needs a GPU with a compute capability that existed by CUDA 10.1 (Turing/Volta/Pascal etc.) — very new GPU generations (e.g. Ada/Hopper/Blackwell) will not have kernels for their architecture in this build.

To (re)create the environment:

```bash
conda create -n spinetk python=3.9 -y
conda activate spinetk
cd detection_models/SpineTK
./install.sh
```

`install.sh` installs, in order:
1. `pyyaml==5.1` (required by an older detectron2/yacs version)
2. `torch==1.8.0+cu101` / `torchvision==0.9.0+cu101` from `download.pytorch.org`
3. `detectron2` from the prebuilt `cu101`/`torch1.8` wheel index
4. `numpy<2` — torch 1.8's compiled extensions predate NumPy 2.0's ABI break (`AttributeError: _ARRAY_API not found` otherwise)
5. `pillow<10` — detectron2 0.6 uses legacy `PIL.Image.LINEAR`/`BILINEAR` constants that Pillow 10 removed
6. `opencv-python-headless<4.10`, `pandas`, `scikit-learn` — the newest `opencv-python-headless` releases require `numpy>=2`, which would silently re-break torch, so it's pinned below that

Verify the install:

```bash
python -c "import torch, detectron2; print(torch.__version__, torch.cuda.is_available(), detectron2.__version__)"
# expected: 1.8.0+cu101 True 0.6+cu101
```

### Set up images/annotations

*This section is the original project's documentation of the on-disk data format, which this copy's code still uses unchanged. The DICOM-preprocessing and `imgaug`-based augmentation-generation example code below is the original author's own approach; for our own datasets we instead wrote one-off conversion scripts per source dataset (outside this folder, in the repo root - e.g. converting a Roboflow COCO export and a LabelMe-annotated dataset into this exact folder structure) and have not yet generated `augs/` folders, so our own baseline directories currently only have the `src/0/` half of this structure.*

Images must be contained in a single folder with the following structure:

```
.
└── baseline-directory/
    ├── image-id-1/
    │   ├── augs/
    │   │   ├── 1/
    │   │   │   ├── aug-1.json
    │   │   │   └── aug-1.png
    │   │   ├── 2/
    │   │   │   ├── aug-2.json
    │   │   │   └── aug-2.png
    │   │   └── ...
    │   └── src/
    │       └── 0/
    │           ├── src-0.json
    │           └── src-0.png
    ├── image-id-2/
    │   ├── augs/
    │   │   ├── 1/
    │   │   │   ├── aug-1.json
    │   │   │   └── aug-1.png
    │   │   ├── 2/
    │   │   │   ├── aug-2.json
    │   │   │   └── aug-2.png
    │   │   └── ...
    │   └── src/
    │       └── 0/
    │           ├── src-0.json
    │           └── src-0.png
    ├── ...
    └── metadata.json
```

Each case gets its own folder. Under each folder, a case has image augmentations and a source image. The source image is a .png image with values in the range of 0-255. It is up to you how to get DICOM image values into this range; however, it has been useful for me to follow this rough algorithm: 

1. Pick only the most central slice of your image series (or just the single image if a radiograph)
2. Find the mean and standard deviation of all non-zero pixel values.
3. Use np.clip to "clip" values between -2.5 SD and +2.5 SD of the mean. 
4. Rescale the min/max values to 0 to 255. 
5. Save the image

This is some example code that implements that: 

```py
def read_img_and_kps(img_entry):
    ds = pydicom.dcmread(img_entry['dicom_path'])
    img = np.array(Image.fromarray(ds.pixel_array))
    mean, std = img.mean(), img.std()
    cl1 = np.clip((img-mean)/std, -2.5, 2.5)
    cl1 = ((cl1 - cl1.min()) * (1/(cl1.max() - cl1.min()) * 255)).astype(np.uint8)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    return cl1
```

Augmentations can be generated using the `imgaug` python library: I use the following augmentation formula (has evolved since the original paper).

```py
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),
        sometimes(iaa.Affine(
            rotate=(-5, 5),
        )),
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Add((-10, 10)),
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0)),
            ],
            random_order=True
        )
    ],
    random_order=True
)
```

From there, you will need to generate a COCO compliant annotation file. The following method was used to do so in our paper. Note this isn't meant to be reproduced, but just to provide a guideline for how to generate augmented images with keypoints.

```py
# params: `img_entry` which is a dict with a filename  
# and whether the image contains hardware (we will 
# selectively augment these to a higher extent than other images). 
# `fname` is the filename, `aug_seq` is the augmentation sequence above
def read_img_and_plot_kps(img_entry, fname, aug_seq):
    try:
        #gets image (and stores in cl1) and keypoint entries [[X,Y], [X,Y], ...]
        cl1, kps = read_img_and_kps(img_entry) 
        # resize image and keypoints to 800px wide
        scale_pct = 800/cl1.shape[1]
        cl1 = cv2.resize(cl1,
            (int(cl1.shape[1]*scale_pct), int(cl1.shape[0]*scale_pct))
        )
        kps = kps*scale_pct
        
        # put image keypoints into separate array for img aug to work on them
        img_aug_kps = []
        for pt in kps:
            img_aug_kps.append(Keypoint(x=int(round(pt[0])), y=int(round(pt[1]))))
        kpsoi = KeypointsOnImage(img_aug_kps, shape=cl1.shape)
        
        # Here we selectively augment any images that have hardware
        tot_augs = 10 if img_entry['hardware'] == False else 30
        aug_images = [cl1 for _ in range(tot_augs)]
        aug_kps = [kpsoi for _ in range(tot_augs)]
        seq = aug_seq
        image_augs, kpsoi_augs = seq(images=aug_images, keypoints=aug_kps)
        
        # Save to folder
        baseline_dir = f"./baseline-directory/{fname}"
        if (os.path.isdir(baseline_dir)):
            shutil.rmtree(baseline_dir)
        os.makedirs(baseline_dir)
        if (os.path.isdir(f"{baseline_dir}/augs") == False):
            os.mkdir(f'{baseline_dir}/augs')
        if (os.path.isdir(f"{baseline_dir}/src") == False):
            os.mkdir(f'{baseline_dir}/src')
        image_augs.insert(0, cl1)
        kpsoi_augs.insert(0, kpsoi)
        for ida, i in enumerate(image_augs):
            # Now generate the COCO dict
            if ida > 0:
                os.mkdir(f'{baseline_dir}/augs/{ida}')
                cv2.imwrite(f'{baseline_dir}/augs/{ida}/aug-{ida}.png', i)
            else:
                os.mkdir(f'{baseline_dir}/src/{ida}')
                cv2.imwrite(f'{baseline_dir}/src/{ida}/src-{ida}.png', i)
            
            polygons = [
                list(
                    map(lambda p: [float(round(p.x)), float(round(p.y)), 2],
                        kpsoi_augs[ida][i: i + 4])
                )
                for i in range(0, len(kpsoi_augs[ida]), 4)
            ]
            
            bboxes = [get_bounding_box(p) for p in polygons]
            boxmode = 0 # BoxMode.XYXY_ABS from Detectron2 Docs
            
            writefile = f'{baseline_dir}/augs/{ida}/aug-{ida}' if ida > 0 else\
                        f'{baseline_dir}/src/{ida}/src-{ida}'
            with open(f"{writefile}.json", 'w') as outfile:
                obj = {
                    'file_name': f'{writefile}.png',
                    'height': int(i.shape[0]),
                    'width': int(i.shape[1]),
                    'image_id': f"{fname}-{'aug' if ida > 0 else 'src'}-{ida}",
                    'annotations': [
                        {'bbox': bboxes[idx], 
                         'bbox_mode': boxmode,
                         'category_id': 0,
                         'keypoints': list(itertools.chain(*p))
                        } for idx,p in enumerate(polygons)
                    ]
                }
                json.dump(obj, outfile)
        
        print(f"DONE WITH {fname}")
    except:
        print(f"COULD NOT DO {fname}")
        traceback.print_exc()
read_img_and_plot_kps(test_img, 'test', seq)

```


The .json files must have the following general format: 

```js
{
    'file_name': path to file,
    'height': image height in px,
    'width': image width in px,
    'image_id': unique name for the image (filepath will do),
    'annotations': [
        // for each of the vertebrae
        { 
            'bbox': bounding box array specified as 
                    [topleftX, topleftY, bottomRightX, bottomRightY],
            'bbox_mode': 0,
            'category_id': 0,
            'keypoints': [X1, Y1, 2.0, X2, Y2, 2.0, X3, Y3, 2.0, ...]
        }, 
    ]
}
```

You'll also need to generate a `metadata.json` file. The metadata file should contain the following: 

```js
{
    "image_id_that_correlated_to_folder_names": {
        "stratifying variable": True or False // (e.g. "Hardware": True)
    }, 
    ...
}
```

### Running

This copy replaces the original's Colab-notebook, cell-by-cell workflow with plain command-line scripts, each driven by a `SpineTKConfig` YAML file. `configs/csxa.yaml` is a fully worked example already pointing at a real converted dataset.

Train:

```bash
conda activate spinetk
python run.py configs/csxa.yaml
```

This splits the baseline dataset, registers it with Detectron2, trains for `iters` iterations (resuming from an existing checkpoint in `output_dir` if `resume: true`), and writes `model_final.pth` to `output_dir`.

Evaluate per-keypoint pixel error against ground truth, over a full split:

```bash
python compute_keypoint_error.py configs/csxa.yaml --split test --csv keypoint_errors.csv
```

Plot a random sample of predictions against ground truth, for visual inspection:

```bash
python plot_predictions.py configs/csxa.yaml --split test --limit 5 --seed 42 --output-dir prediction_plots
```

Sanity-check that training actually mattered, by running the same evaluation against the original, never-fine-tuned checkpoint instead:

```bash
python plot_predictions.py configs/detectron.yaml --split test --limit 5 --seed 42 --output-dir plots_untrained
```

See `commands.sh` for the exact commands used for this project's own runs, and each script's own `--help` / module docstring for the full option list.
