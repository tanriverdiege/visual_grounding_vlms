from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Fields that must be positive ints / positive numbers / bools. bool is a
# subclass of int, so it is rejected explicitly for the int fields.
_POSITIVE_INT_FIELDS = (
    "min_pixels",
    "max_pixels",
    "batch_size",
    "grad_accum",
    "lora_r",
    "lora_alpha",
    "eval_steps",
    "save_total_limit",
    "logging_steps",
)
_POSITIVE_FLOAT_FIELDS = ("epochs", "lr")
_OPTIONAL_POSITIVE_INT_FIELDS = ("max_train_samples", "max_val_samples")


@dataclass(frozen=True)
class LoraTrainingConfig:
    """Every setting train_lora.py takes.

    Attributes:
        model_name: A HuggingFace model id for the base model.
        data_path: The JSONL for data (promopts, answers, image_paths, image_ids).
        baseline_dir: Raw image data directory.
        output_dir: Where checkpoints, the final adapter and a copy of this
            config are written.
        min_pixels, max_pixels: Qwen-VL image processor resolution bounds.
            Keep them identical to the inference config the adapter will be
            evaluated with, so training and eval see the same resolution.
        epochs: Number of training epochs (fractional allowed).
        lr: Peak learning rate.
        lr_scheduler_type: Any transformers scheduler name, e.g. "cosine".
        warmup_ratio: Fraction of total steps spent warming up the lr.
        weight_decay: AdamW weight decay.
        batch_size: Per-GPU batch size (train and eval).
        grad_accum: Gradient accumulation steps. Effective batch size is
            n_gpus * batch_size * grad_accum.
        lora_r, lora_alpha, lora_dropout: LoRA rank, scaling and dropout.
        lora_target_modules: Module names LoRA adapters are attached to.
        eval_steps: Evaluate (val loss) and save a checkpoint every this many
            optimizer steps.
        save_total_limit: Checkpoints kept on disk; the best one by val loss
            is always kept.
        logging_steps: Log training loss every this many steps.
        dataloader_num_workers: Worker processes per GPU for image loading.
        max_train_samples, max_val_samples: Truncate the splits, for
            testing. None uses the whole split.
        seed: Random seed for Trainer (shuffling, dropout, LoRA init).
    """

    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    data_path: str = "/data/datasets/csxa/lora/lora_vert_corners.jsonl"
    baseline_dir: str = "/data/datasets/csxa/spinetk_baseline"
    output_dir: str = "lora/runs/qwen3_vl_4b_vert_corners"

    min_pixels: int = 802816
    max_pixels: int = 2744000

    epochs: float = 3
    lr: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    batch_size: int = 1
    grad_accum: int = 8

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # LLM-side projections only. Qwen3-VL's vision tower names its layers
    # differently (attn.qkv, attn.proj, mlp.linear_fc1/2), so these names
    # leave the vision encoder frozen!
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
    )

    eval_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10
    dataloader_num_workers: int = 4

    max_train_samples: int | None = None
    max_val_samples: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate field types and ranges.

        Raises:
            TypeError: If a field holds a value of the wrong type.
            ValueError: If a field holds a value out of range.
        """
        for name in ("model_name", "data_path", "baseline_dir", "output_dir", "lr_scheduler_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        for name in _POSITIVE_INT_FIELDS:
            _check_positive_int(name, getattr(self, name))
        for name in _OPTIONAL_POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _check_positive_int(name, value)

        for name in _POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name in ("warmup_ratio", "weight_decay", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        if self.min_pixels > self.max_pixels:
            raise ValueError(
                f"min_pixels ({self.min_pixels}) must not exceed max_pixels ({self.max_pixels})"
            )

        if (
            not isinstance(self.lora_target_modules, list)
            or not self.lora_target_modules
            or not all(isinstance(m, str) for m in self.lora_target_modules)
        ):
            raise TypeError(
                f"lora_target_modules must be a non-empty list of strings, got "
                f"{self.lora_target_modules!r}"
            )

        if isinstance(self.dataloader_num_workers, bool) or not isinstance(
            self.dataloader_num_workers, int
        ) or self.dataloader_num_workers < 0:
            raise ValueError(
                f"dataloader_num_workers must be a non-negative int, got "
                f"{self.dataloader_num_workers!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoraTrainingConfig:
        """Build a config from a YAML file.

        Keys left out of the file keep their defaults above. Unknown keys are
        rejected rather than ignored: in a config file a typo'd key is a
        setting that silently did not apply.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.

        Raises:
            TypeError: If the file's top level is not a mapping.
            ValueError: If the file holds keys that are not fields of
                `LoraTrainingConfig`.
        """
        path = Path(path)
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"{path} must hold a YAML mapping of settings, got {type(raw).__name__}"
            )

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: "
                f"{', '.join(sorted(str(key) for key in unknown))}; "
                f"expected any of {', '.join(sorted(known))}"
            )
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write the config out, so a run records exactly what produced it.

        Args:
            path: Destination file; overwritten if it exists.
        """
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )


def _check_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
