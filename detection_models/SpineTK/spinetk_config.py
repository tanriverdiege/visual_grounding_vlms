"""File-backed configuration for SpineTK training/inference.

Every knob `train()`/`inference()`/`setup_catalogs()` need lives in
`SpineTKConfig`, and a YAML file is the canonical way to fill it in - same
convention as `vision_language_models/vlm_config.py`. Holding the knobs in one
serialisable object means a run can be reproduced from the file that produced
it, and `num_keypoints` is *derived* from `kpnames` rather than being a
separate field - this project has repeatedly broken by hand-syncing a
keypoint count (get_dicts.py's 18/12 check, NUM_KEYPOINTS, KEYPOINT_OKS_SIGMAS
length) across several files; deriving it removes that whole bug class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SpineTKConfig:
    """Every setting SpineTK's train/inference/catalog-setup steps need.

    Frozen on purpose: a run should be describable by the exact file that
    produced it, not by a config that got mutated partway through.

    Attributes:
        baseline_directory: Directory of per-image folders with src/aug
            annotations (see get_dicts.py).
        kpnames: Ordered keypoint names, e.g. ['bottom_left', 'bottom_right',
            'top_right', 'top_left']. num_keypoints is always len(kpnames) -
            it is not a separate field, so it can never drift out of sync.
        version: Dataset/catalog version tag Detectron2's DatasetCatalog
            registers names under, e.g. "v1" -> "vertv1_train".
        model_zoo_config: Detectron2 model zoo config name to start from.
        num_classes: Number of object classes (SpineTK has one: "vertebrae").
        iters: Number of training iterations.
        resume: Resume training from an existing checkpoint if one is found.
        ims_per_batch: Images per training batch.
        base_lr: Base learning rate.
        batch_size_per_image: ROI head batch size per image.
        num_workers: Data loader worker count.
        output_dir: Where checkpoints get written.
        model_weights: Path to trained weights, used for inference.
        score_thresh: Inference score threshold.
        oks_sigmas: Per-keypoint OKS sigmas for keypoint-AP evaluation, one
            per entry in kpnames. None fills in a uniform 1.0 per keypoint -
            an unmeasured placeholder, not calibrated against real annotation
            variance; only the length is guaranteed correct.
        visualize: Dump a couple of annotated training samples before
            training.
        eval_image: Path to an image to run a final inference smoke test on.
            None skips that step.
        eval_output: Where to write the annotated eval-image output.
    """

    baseline_directory: str
    kpnames: list[str]
    version: str = "v1"

    model_zoo_config: str = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    num_classes: int = 1

    iters: int = 1300
    resume: bool = True
    ims_per_batch: int = 2
    base_lr: float = 0.00025
    batch_size_per_image: int = 512
    num_workers: int = 2
    output_dir: str = "./output"

    model_weights: str = "output/model_final.pth"
    score_thresh: float = 0.8
    oks_sigmas: list[float] | None = None

    visualize: bool = False
    eval_image: str | None = None
    eval_output: str = "out.png"

    @property
    def num_keypoints(self) -> int:
        """Number of keypoints per instance - always len(kpnames)."""
        return len(self.kpnames)

    def __post_init__(self) -> None:
        """Validates fields and fills in the derived oks_sigmas default.

        Raises:
            TypeError: If a field holds a value of the wrong type.
            ValueError: If a field's value is out of the valid range, or
                kpnames/oks_sigmas are malformed.
        """
        if (
            not isinstance(self.baseline_directory, str)
            or not self.baseline_directory.strip()
        ):
            raise ValueError(
                f"baseline_directory must be a non-empty path, got {self.baseline_directory!r}"
            )

        if not isinstance(self.kpnames, list) or not self.kpnames:
            raise ValueError(f"kpnames must be a non-empty list, got {self.kpnames!r}")
        if not all(isinstance(name, str) and name.strip() for name in self.kpnames):
            raise ValueError(
                f"kpnames must all be non-empty strings, got {self.kpnames!r}"
            )
        if len(set(self.kpnames)) != len(self.kpnames):
            raise ValueError(
                f"kpnames must not contain duplicates, got {self.kpnames!r}"
            )

        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(
                f"version must be a non-empty string, got {self.version!r}"
            )

        for name in ("iters", "batch_size_per_image", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if (
            isinstance(self.ims_per_batch, bool)
            or not isinstance(self.ims_per_batch, int)
            or self.ims_per_batch < 1
        ):
            raise ValueError(
                f"ims_per_batch must be a positive int, got {self.ims_per_batch!r}"
            )

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError(
                f"num_workers must be a non-negative int, got {self.num_workers!r}"
            )

        if (
            not isinstance(self.base_lr, (int, float))
            or isinstance(self.base_lr, bool)
            or self.base_lr <= 0
        ):
            raise ValueError(f"base_lr must be a positive number, got {self.base_lr!r}")

        if (
            not isinstance(self.score_thresh, (int, float))
            or isinstance(self.score_thresh, bool)
            or not (0.0 <= self.score_thresh <= 1.0)
        ):
            raise ValueError(
                f"score_thresh must be between 0 and 1, got {self.score_thresh!r}"
            )

        if not isinstance(self.resume, bool):
            raise TypeError(f"resume must be a bool, got {type(self.resume).__name__}")
        if not isinstance(self.visualize, bool):
            raise TypeError(
                f"visualize must be a bool, got {type(self.visualize).__name__}"
            )

        if self.oks_sigmas is None:
            object.__setattr__(self, "oks_sigmas", [1.0] * self.num_keypoints)
        else:
            if len(self.oks_sigmas) != self.num_keypoints:
                raise ValueError(
                    f"oks_sigmas has {len(self.oks_sigmas)} entries but there are "
                    f"{self.num_keypoints} kpnames; they must match 1:1"
                )
            if not all(
                isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0
                for s in self.oks_sigmas
            ):
                raise ValueError(
                    f"oks_sigmas must all be positive numbers, got {self.oks_sigmas!r}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpineTKConfig:
        """Builds a config from a YAML file.

        Unknown keys are rejected rather than ignored: in a config file a
        typo'd key is a setting that silently did not apply.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.

        Raises:
            TypeError: If the file's top level is not a mapping.
            ValueError: If the file is empty or holds keys that are not
                fields of SpineTKConfig.
        """
        path = Path(path)
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raise ValueError(
                f"{path} is empty; it must define at least baseline_directory and kpnames"
            )
        if not isinstance(raw, dict):
            raise TypeError(
                f"{path} must hold a YAML mapping of settings, got {type(raw).__name__}"
            )

        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: {', '.join(sorted(str(key) for key in unknown))}; "
                f"expected any of {', '.join(sorted(known))}"
            )
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Writes the config back out, so a run can record what produced it.

        Args:
            path: Destination file; overwritten if it exists.
        """
        data = {k: v for k, v in asdict(self).items()}
        Path(path).write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
