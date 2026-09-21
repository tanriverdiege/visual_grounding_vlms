import json
import os
import random

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def make_split(
    baseline_directory: str,
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits the baseline dataset(in the expected format specified in the README) 
    into train/val/test splits.
    Args:
        baseline_directory (str): Path to the baseline dataset directory.
    Returns:
        tuple[list[int], list[int], list[int]]: Lists of train, val, and test IDs.
    """

    with open(f"{baseline_directory}/metadata.json", encoding="utf-8") as meta_file:
        df = pd.read_json(meta_file, orient="index")
        x_train_p, x_test = train_test_split(
            df, test_size=0.2, random_state=100, shuffle=True
        )
        x_train, x_val = train_test_split(
            x_train_p, test_size=(10 / 70), random_state=100, shuffle=True
        )

    train_ids = x_train.index.to_list()
    val_ids = x_val.index.to_list()
    test_ids = x_test.index.to_list()

    return train_ids, val_ids, test_ids


def sample_csxa_image_paths(
    baseline_directory: str,
    num_examples: int,
    seed: int = 42,
) -> list[str]:
    """
    Randomly samples image paths from the CSXA baseline dataset's TEST split,
    so images used to evaluate a VLM never overlap with SpineTK's train/val
    images from the same baseline directory.
    Args:
        baseline_directory (str): Path to the CSXA baseline dataset directory,
            e.g. "/data/datasets/csxa/spinetk_baseline". Passed straight to
            make_split, so it must satisfy that function's requirements (a
            top-level metadata.json), and each image_id's PNG must be at
            "<baseline_directory>/<image_id>/src/0/src-0.png".
        num_examples (int): Number of example image paths to select.
        seed (int): Random seed, for a reproducible sample across runs.
            Independent of make_split's own random_state=100, which controls
            which ids land in the test split in the first place, not which
            of them get selected here.
    Returns:
        list[str]: Paths to the selected images' src-0.png files.
    Raises:
        ValueError: If num_examples exceeds the number of test-split images.
    """
    _, _, test_ids = make_split(baseline_directory)
    # make_split's ids come from pd.read_json(..., orient="index"), which
    # infers an int64 index for metadata.json's purely-numeric string keys --
    # silently dropping CSXA's leading zeros (e.g. "0212127" -> 212127). The
    # on-disk <image_id> directories are always 7-digit zero-padded strings,
    # so zfill(7) restores the id this function actually needs to look up.
    image_ids = [str(image_id).zfill(7) for image_id in test_ids]

    if num_examples > len(image_ids):
        raise ValueError(
            f"requested {num_examples} examples but only {len(image_ids)} "
            f"test-split images are available in {baseline_directory}"
        )

    selected_ids = random.Random(seed).sample(image_ids, num_examples)

    return [
        os.path.join(baseline_directory, image_id, "src", "0", "src-0.png")
        for image_id in selected_ids
    ]


def crop_bounding_box(image, bbox):
    """
    Crops the image to the specified bounding box.
    Args:
        image (PIL.Image.Image): The input image.
        bbox (tuple[int, int, int, int]): Bounding box in (x_min, y_min, x_max, y_max) format.
    Returns:
        PIL.Image.Image: The cropped image.
    """
    return image.crop(bbox)

def get_image_res_dims(image_path: str) -> tuple[int, int]:
    """
    Given an image return its resolution dimensions.
    Args:
        image_path (str): Path to the image file.
    Returns:
        tuple[int, int]: (width, height) in pixels.
    """
    with Image.open(image_path) as image:
        return image.size

class XrayImage:
    """
    A class to handle X-ray images.
    """

    def __init__(self, image_path: str):
        """
        Initializes the XrayImage object.
        Args:
            image_path (str): Path to the X-ray image file.
        """
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.resolution = self.image.size

    def get_coordinate_data(self):
        """
        Returns the coordinate data for this X-ray image (e.g. vertebrae
        locations, bounding boxes, etc.). Specific per dataset and must be
        implemented in a subclass.
        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def crop_to_bounding_box(self):
        """
        Crops the image to its bounding box, derived from this image's
        coordinate data. Specific per dataset and must be implemented in a
        subclass.
        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

class CsxaXrayImage(XrayImage):
    """
    A class to handle X-ray images from the CSXA dataset.
    """

    def __init__(self, image_path: str):
        """
        Initializes the CsxaXrayImage object.
        Args:
            image_path (str): Path to the CSXA X-ray image file, e.g.
                "<baseline_directory>/<image_id>/src/0/src-0.png".
        """
        super().__init__(image_path)
        self.metadata_path = os.path.splitext(image_path)[0] + ".json"
        self.get_metadata()

    def get_metadata(self):
        """
        Loads this image's CSXA annotation JSON and assigns its per-vertebra
        data as attributes.
        Returns:
            list[dict]: The raw per-vertebra annotation records.
        """
        with open(self.metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        self.annotations = metadata["annotations"]
        self.bboxes = [annotation["bbox"] for annotation in self.annotations]
        self.keypoints = [annotation["keypoints"] for annotation in self.annotations]
        self.vertebra_labels = [annotation["vertebra_label"] for annotation in self.annotations]

        return self.annotations

    def crop_to_bounding_box(
        self,
        vertebra_label: str,
        save_base_path: str | None = None,
        padding: float = 0.7,
        upscale: float = 5.0,
    ) -> Image.Image:
        """
        Crops the image to the bounding box of the given vertebra, expanded
        by `padding` (e.g. 0.1 -> the crop's width and height are 10% larger
        than the raw bounding box, centered on it).
        Args:
            vertebra_label (str): Vertebra to crop to, e.g. "C3".
            save_base_path (str, optional): If given, the folder to save the
                cropped image into, named
                "<original_basename>-cropped-<vertebra_label>-pad<padding>-up<upscale><ext>".
                If None, the image is not saved.
            padding (float): Fraction to grow the bounding box by. 0 means
                no padding.
            upscale (float): Factor to resize the cropped image by after
                cropping, e.g. 2.0 doubles both dimensions. 1.0 means no
                upscaling.
        Returns:
            PIL.Image.Image: The cropped (and optionally upscaled) image.
        """
        if vertebra_label not in self.vertebra_labels:
            raise ValueError(
                f"{vertebra_label!r} not found in {self.image_path}; "
                f"available labels: {self.vertebra_labels}"
            )
        index = self.vertebra_labels.index(vertebra_label)
        x_min, y_min, x_max, y_max = self.bboxes[index]

        pad_x = (x_max - x_min) * padding / 2
        pad_y = (y_max - y_min) * padding / 2
        image_width, image_height = self.resolution
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(image_width, x_max + pad_x)
        y_max = min(image_height, y_max + pad_y)

        cropped_image = self.image.crop((x_min, y_min, x_max, y_max))

        if upscale != 1.0:
            new_size = (
                round(cropped_image.width * upscale),
                round(cropped_image.height * upscale),
            )
            cropped_image = cropped_image.resize(new_size, Image.LANCZOS)

        if save_base_path is not None:
            base_name, ext = os.path.splitext(os.path.basename(self.image_path))
            file_name = f"{base_name}-cropped-{vertebra_label}-pad{padding}-up{upscale}{ext}"
            cropped_image.save(os.path.join(save_base_path, file_name))

        return cropped_image
