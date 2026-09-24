from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from lora_config import LoraTrainingConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.scripts.utils import make_split  # noqa: E402


def load_samples(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def split_samples(
    samples: list[dict], baseline_dir: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition samples by make_split's train/val/test image ids."""
    train_ids, val_ids, test_ids = make_split(baseline_dir)
    # make_split's ids come back as ints (pd.read_json drops the leading
    # zeros); on-disk ids are 7-digit zero-padded strings.
    train_ids = {str(i).zfill(7) for i in train_ids}
    val_ids = {str(i).zfill(7) for i in val_ids}
    test_ids = {str(i).zfill(7) for i in test_ids}

    train = [s for s in samples if s["image_id"] in train_ids]
    val = [s for s in samples if s["image_id"] in val_ids]
    test = [s for s in samples if s["image_id"] in test_ids]
    return train, val, test


class VertebraDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class QwenVLCollator:
    """Turns a list of samples into a padded, label-masked training batch."""

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # Everything up to and including this marker is prompt; everything
        # after it (the answer and its closing <|im_end|>) is trained on.
        self.assistant_marker = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def _answer_start(self, ids: list[int]) -> int:
        """Index of the first answer token: just past the LAST assistant marker."""
        marker = self.assistant_marker
        for start in range(len(ids) - len(marker), -1, -1):
            if ids[start : start + len(marker)] == marker:
                return start + len(marker)
        raise ValueError("assistant marker not found -- chat template changed?")

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts, images = [], []
        for sample in batch:
            # Same user-turn layout as vlm.py's format_prompt (image first,
            # then text), so training and inference prompts match exactly.
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            images.append(Image.open(sample["image_dir"]).convert("RGB"))

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for row, ids in enumerate(inputs["input_ids"].tolist()):
            labels[row, : self._answer_start(ids)] = -100
        inputs["labels"] = labels
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of Qwen3-VL on CSXA.")
    parser.add_argument(
        "--config-path",
        default="lora/configs/qwen_3_vl_4b_vert_corners.yaml",
        help="LoraTrainingConfig YAML; see lora/configs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LoraTrainingConfig.from_yaml(args.config_path)

    # A plain `python` launch with several visible GPUs makes Trainer fall
    # back to nn.DataParallel, which splits every batch tensor along dim 0.
    # Qwen-VL packs all images' patches into one flat pixel_values tensor, so
    # that split cuts images apart and the vision tower crashes with a
    # patch/position-embedding size mismatch. Use DDP (torchrun) or one GPU.
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            f"{torch.cuda.device_count()} GPUs visible but not launched with torchrun.\n"
            "  one GPU:   CUDA_VISIBLE_DEVICES=0 python lora/train_lora.py --config-path <yaml>\n"
            "  multi-GPU: torchrun --nproc_per_node=<N> lora/train_lora.py --config-path <yaml>"
        )

    train, val, test = split_samples(load_samples(config.data_path), config.baseline_dir)
    if config.max_train_samples is not None:
        train = train[: config.max_train_samples]
    if config.max_val_samples is not None:
        val = val[: config.max_val_samples]
    print(f"train={len(train)}  val={len(val)}  test={len(test)} (held out, unused)")

    processor = AutoProcessor.from_pretrained(
        config.model_name, min_pixels=config.min_pixels, max_pixels=config.max_pixels
    )
    # Right padding for training: the answer-start search and the loss mask
    # both assume real tokens come first in each row.
    processor.tokenizer.padding_side = "right"

    # fp16, not bf16: the cluster's Quadro RTX 5000s (Turing) have no bf16
    # support. Base weights stay fp16 and frozen; peft keeps the LoRA weights
    # in fp32, which is what fp16 mixed precision + the grad scaler need.
    model = AutoModelForImageTextToText.from_pretrained(config.model_name, dtype=torch.float16)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        gradient_checkpointing=True,
        # Non-reentrant checkpointing works with frozen inputs, so no
        # enable_input_require_grads() hack is needed for peft.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=config.logging_steps,
        # The dataset yields raw dicts (paths, strings) the collator turns
        # into tensors; Trainer would otherwise drop them as "unused".
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=VertebraDataset(train),
        eval_dataset=VertebraDataset(val),
        data_collator=QwenVLCollator(processor),
    )
    # Record the exact config next to the checkpoints (rank 0 only, so DDP
    # processes don't race on the same file).
    if trainer.is_world_process_zero():
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        config.to_yaml(Path(config.output_dir) / "lora_config.yaml")
    trainer.train()

    # Saves only the LoRA adapter (tens of MB), not the 4B base model.
    final_dir = Path(config.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
