"""Creates a contact-sheet PNG from a run directory's "*_annotated.png" images.

Usage:
  python make_grid.py DIRECTORY [--output PATH] [--rows N] [--cols N]
      [--cell-width W] [--cell-height H]

e.g., the default 2x5 grid, one cell per image in a 10-image run:
  python make_grid.py runs/crop_vert_corner_up10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

_ANNOTATED_GLOB = "*_annotated.png"
_DEFAULT_CELL_SIZE = (320, 320)
_PADDING = 8  # empty margin, in pixels, kept around each thumbnail within its cell
_BACKGROUND = "white"

# Saved PNG DPI metadata (a viewer/printer uses this to map pixels to a
# physical size; it doesn't change the pixel dimensions above -- that's
# controlled by --cell-width/--cell-height, e.g. use a bigger cell size for
# more actual pixels as well as a higher DPI tag).
_DEFAULT_DPI = 300

# run_object_detection.py names each annotated file "<i>_<...>_annotated.png",
# a zero-padded-in-width but not zero-padded-in-value run index (e.g. "0_",
# "9_", "10_") -- sorting on the filename directly would put "10_" before
# "2_" lexicographically, so the leading index is parsed out and sorted on
# numerically instead.
_LEADING_INDEX_RE = re.compile(r"^(\d+)_")


def _sort_key(path: Path) -> tuple[float, str]:
    match = _LEADING_INDEX_RE.match(path.name)
    return (float(match.group(1)) if match else float("inf")), path.name


def make_grid(
    directory: str | Path,
    output_path: str | Path | None = None,
    rows: int = 2,
    cols: int = 5,
    cell_size: tuple[int, int] = _DEFAULT_CELL_SIZE,
    dpi: int = _DEFAULT_DPI,
) -> Path:
    """Tiles every "*_annotated.png" in `directory` into one `rows`x`cols` grid.

    Every image is resized to the exact same pixel dimensions (`cell_size`
    minus `_PADDING`) regardless of its own original size or aspect ratio --
    a run's annotated images aren't all the same size (CSXA source images
    vary per patient, and a crop's own dimensions depend on that vertebra's
    bounding box), so without this a differently-shaped source would end up
    displayed smaller than the rest after an aspect-ratio-preserving fit.
    This can visibly stretch/squash a source that isn't already close to
    `cell_size`'s aspect ratio.

    Args:
        directory: Run directory to read "*_annotated.png" files from (not
            recursive).
        output_path: Where to save the grid PNG. Defaults to
            "<directory>/grid.png".
        rows: Grid rows.
        cols: Grid columns.
        cell_size: (width, height) of each grid cell, in pixels.
        dpi: DPI metadata to embed in the saved PNG. This is a tag a
            viewer/printer reads to map pixels to a physical size -- it
            doesn't add any actual pixels, so pair a higher value with a
            bigger `cell_size` for a sharper image, not just this alone.

    Returns:
        Path: `output_path` (or its default), where the grid was saved.

    Raises:
        FileNotFoundError: If `directory` has no "*_annotated.png" files.
        ValueError: If there are more images than `rows * cols` cells.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(_ANNOTATED_GLOB), key=_sort_key)
    if not paths:
        raise FileNotFoundError(f"no {_ANNOTATED_GLOB!r} files found in {directory}")

    capacity = rows * cols
    if len(paths) > capacity:
        raise ValueError(
            f"found {len(paths)} {_ANNOTATED_GLOB!r} files in {directory}, but "
            f"a {rows}x{cols} grid only holds {capacity}; pass --rows/--cols "
            "to fit them all"
        )

    cell_w, cell_h = cell_size
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), _BACKGROUND)

    target_size = (cell_w - _PADDING * 2, cell_h - _PADDING * 2)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            thumb = image.convert("RGB").resize(target_size, Image.LANCZOS)

        row, col = divmod(i, cols)
        x = col * cell_w + _PADDING
        y = row * cell_h + _PADDING
        grid.paste(thumb, (x, y))

    output_path = Path(output_path) if output_path is not None else directory / "grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, dpi=(dpi, dpi))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--directory", help="Run directory containing *_annotated.png files")
    parser.add_argument(
        "--output", default=None, help="Output PNG path (default: <directory>/grid.png)"
    )
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=_DEFAULT_CELL_SIZE[0])
    parser.add_argument("--cell-height", type=int, default=_DEFAULT_CELL_SIZE[1])
    parser.add_argument(
        "--dpi", type=int, default=_DEFAULT_DPI,
        help="DPI metadata for the saved PNG; pair with --cell-width/--cell-height "
        "for a sharper (more actual pixels) image, not just this alone.",
    )
    args = parser.parse_args()

    saved_to = make_grid(
        args.directory,
        output_path=args.output,
        rows=args.rows,
        cols=args.cols,
        dpi=args.dpi,
        cell_size=(args.cell_width, args.cell_height),
    )
    print(f"saved grid to {saved_to}")
