from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import AutoTokenizer

from predictive_coding_llm.config import load_config
from predictive_coding_llm.data import (
    dataset_cache_exists,
    ensure_free_disk_space,
    get_dataset_cache_dir,
    load_text_datasets,
    tokenize_for_causal_lm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--progress-file", help="Path to a JSON progress file.")
    return parser.parse_args()


def write_progress(path: str | None, **values) -> None:
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.time(), **values}
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        write_progress(args.progress_file, stage="loading config", percent=2, status="running")
        config = load_config(args.config)
        output_dir = get_dataset_cache_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_free_disk_space(output_dir, float(config["dataset"].get("min_free_disk_gb", 5.0)))

        if dataset_cache_exists(output_dir):
            write_progress(
                args.progress_file,
                stage="using cached dataset",
                percent=100,
                status="cached",
                output_dir=str(output_dir),
            )
            print(f"Using cached tokenized dataset at {output_dir}")
            return

        write_progress(args.progress_file, stage="loading tokenizer", percent=12, status="running")
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        write_progress(args.progress_file, stage="loading dataset", percent=30, status="running")
        datasets = load_text_datasets(config["dataset"])
        train_rows = len(datasets["train"])
        validation_rows = len(datasets["validation"])

        write_progress(
            args.progress_file,
            stage="tokenizing dataset",
            percent=58,
            status="running",
            train_rows=train_rows,
            validation_rows=validation_rows,
        )
        tokenized = tokenize_for_causal_lm(datasets, tokenizer, config["dataset"])

        write_progress(
            args.progress_file,
            stage="saving dataset",
            percent=86,
            status="running",
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=str(output_dir),
        )
        tokenized.save_to_disk(output_dir)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        write_progress(
            args.progress_file,
            stage="complete",
            percent=100,
            status="complete",
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=str(output_dir),
        )
        print(f"Saved tokenized dataset to {output_dir}")
    except Exception as exc:
        write_progress(args.progress_file, stage="failed", percent=100, status="error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
