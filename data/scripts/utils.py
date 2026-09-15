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

