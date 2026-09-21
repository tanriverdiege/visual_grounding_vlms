from utils import CsxaXrayImage, sample_csxa_image_paths, make_split

image_paths = sample_csxa_image_paths(
    baseline_directory="/data/datasets/csxa/spinetk_baseline",
    num_examples=5,
)

print(image_paths)