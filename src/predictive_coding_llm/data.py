from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase


def get_dataset_cache_dir(config: dict[str, Any]) -> Path:
    cache_dir = config.get("dataset", {}).get("cache_dir")
    if cache_dir:
        return Path(cache_dir)
    return Path(config["run"]["output_dir"]) / "dataset-cache"


def ensure_free_disk_space(path: str | Path, min_free_gb: float = 5.0) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < min_free_gb:
        raise OSError(
            f"Only {free_gb:.2f} GB free at {target}. "
            f"Need at least {min_free_gb:.2f} GB before preparing a dataset."
        )


def load_text_datasets(dataset_config: dict[str, Any]) -> DatasetDict:
    kwargs: dict[str, Any] = {"path": dataset_config["name"]}
    if dataset_config.get("config_name"):
        kwargs["name"] = dataset_config["config_name"]

    train = load_dataset(**kwargs, split=dataset_config["split"])
    validation_split = dataset_config.get("validation_split")
    if validation_split:
        validation = load_dataset(**kwargs, split=validation_split)
    else:
        split = train.train_test_split(test_size=0.002, seed=42)
        train = split["train"]
        validation = split["test"]

    return DatasetDict(train=train, validation=validation)


def dataset_cache_exists(path: str | Path) -> bool:
    cache_path = Path(path)
    return (cache_path / "dataset_dict.json").exists()


def load_tokenized_cache(path: str | Path) -> DatasetDict:
    return load_from_disk(str(path))


def tokenize_for_causal_lm(
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    dataset_config: dict[str, Any],
) -> DatasetDict:
    text_field = dataset_config.get("text_field", "text")
    chat_template = dataset_config.get("chat_template", False)
    max_length = int(dataset_config["max_length"])
    workers = resolve_num_workers(dataset_config.get("preprocessing_num_workers"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, Any]:
        texts = examples[text_field]
        if chat_template:
            texts = [format_messages(messages) for messages in texts]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    remove_columns = datasets["train"].column_names
    return datasets.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        num_proc=workers,
        desc="Tokenizing dataset",
    )


def resolve_num_workers(configured_workers: Any) -> int:
    if configured_workers not in (None, "auto"):
        return max(1, int(configured_workers))

    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count - 1, 8))


def format_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)

    parts = []
    for message in messages:
        if not isinstance(message, dict):
            parts.append(str(message))
            continue
        role = str(message.get("role", "message")).strip().lower()
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)
