"""Contrast/visibility enhancement for low-contrast images (X-rays, scans, ...).

`percentile_stretch` and `clahe` are pure intensity remapping -- they never
crop or resize, so the output is pixel-for-pixel the same size as the input.
That matters for grounding: a box parsed from a reply against the enhanced
image rescales identically to one against the original, no extra bookkeeping
needed.

`pad_to_square` is the exception: it changes the canvas size (by design), so
callers doing grounding against a padded image need the returned offsets to
map boxes back to the original, unpadded image.

These help two separate things, and it's worth being clear about which is
which:
- A human looking at the saved output -- reclaiming dynamic range a low
  contrast scan wastes on a narrow gray band is a real, direct win for eyes.
- The VLM's own detection accuracy -- these models are trained overwhelmingly
  on natural photos, so there's no guarantee better human-visible contrast
  helps the vision encoder; it can help, hurt, or do nothing depending on the
  model. Treat it as one more thing to A/B against the original, not an
  assumed fix.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def percentile_stretch(image: Image.Image, low: float = 1.0, high: float = 99.0) -> Image.Image:
    """Rescale intensities so the [low, high] percentile range fills [0, 255].

    Cheap, global, robust to outlier pixels (hence percentiles rather than
    min/max). The common fix for a scan that looks uniformly gray because a
    scanner or export step compressed its real dynamic range into a narrow
    band well inside 0-255.

    Args:
        image: Source image, any mode.
        low: Lower percentile clipped to black.
        high: Upper percentile clipped to white.

    Returns:
        A new image, same size and mode as the input. Unchanged (a copy) if
        the image has no usable range at these percentiles (e.g. blank).
    """
    arr = np.asarray(image).astype(np.float32)
    lo, hi = np.percentile(arr, low), np.percentile(arr, high)
    if hi <= lo:
        return image.copy()
    stretched = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode=image.mode)


def clahe(
    image: Image.Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> Image.Image:
    """Apply CLAHE: local, noise-limited contrast enhancement.

    Unlike a global stretch, CLAHE equalizes contrast within local tiles, so
    it can reveal detail in both an over-dark and an over-bright region of
    the SAME image at once -- e.g. a radiograph where soft tissue and bone
    sit in very different intensity bands, which a single global curve can't
    serve well simultaneously.

    Args:
        image: Source image.
        clip_limit: Caps how much any one tile's histogram can be amplified,
            keeping noise from being amplified into visible artifacts. Higher
            = more contrast, more noise.
        tile_grid_size: (rows, cols) of tiles the image is divided into for
            local equalization. Smaller tiles = more local, more noise-prone.

    Returns:
        A new image, same size and mode as the input.
    """
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.mode == "L":
        return Image.fromarray(clahe_op.apply(np.asarray(image)), mode="L")
    # Color: equalize luminance only (LAB), so hue/saturation aren't distorted.
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), mode="RGB")


def pad_to_square(image: Image.Image, fill: int | tuple[int, ...] = 0) -> tuple[Image.Image, int, int]:
    """Center `image` on a square canvas sized to its longer side.

    Some models' processors force every input to a square (e.g. by resizing
    to a fixed square input size for the vision encoder), which silently
    stretches a non-square image's aspect ratio and distorts the spatial
    layout the model reasons about -- a real problem for grounding, where the
    model is asked to report *where* something is. Padding to square first,
    the way MedGemma's own localization notebook does, keeps the aspect ratio
    intact; the model instead sees uniform, uninformative padding on two
    sides.

    Args:
        image: Source image, any mode.
        fill: Pixel value for the padding -- an int for single-channel modes,
            an RGB/RGBA tuple otherwise. Defaults to black.

    Returns:
        (padded, x_offset, y_offset): the square image, plus the offset at
        which the original image was placed. Subtract these from coordinates
        read off the padded image (after unnormalizing) to map back to the
        original image's pixel coordinates.
    """
    width, height = image.size
    side = max(width, height)
    x_offset, y_offset = (side - width) // 2, (side - height) // 2
    padded = Image.new(image.mode, (side, side), fill)
    padded.paste(image, (x_offset, y_offset))
    return padded, x_offset, y_offset


if __name__ == "__main__":
    # Quick visual A/B: python image_prep.py <image_path>
    # Writes <stem>_stretch.png and <stem>_clahe.png next to the input.
    import sys
    from pathlib import Path

    src = Path(sys.argv[1])
    img = Image.open(src)
    percentile_stretch(img).save(src.with_name(f"{src.stem}_stretch.png"))
    clahe(img).save(src.with_name(f"{src.stem}_clahe.png"))
    print("wrote", src.with_name(f"{src.stem}_stretch.png"))
    print("wrote", src.with_name(f"{src.stem}_clahe.png"))
