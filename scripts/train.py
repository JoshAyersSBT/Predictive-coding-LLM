from __future__ import annotations

import argparse
import inspect
import os
import signal
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments

from predictive_coding_llm import PredictiveCodingGPT2Config, PredictiveCodingGPT2LMHeadModel
from predictive_coding_llm.config import load_config
from predictive_coding_llm.data import (
    dataset_cache_exists,
    ensure_free_disk_space,
    get_dataset_cache_dir,
    load_text_datasets,
    load_tokenized_cache,
    tokenize_for_causal_lm,
)
from predictive_coding_llm.metrics import JsonlMetricsCallback
from predictive_coding_llm.trainer import PredictiveCodingTrainer


def build_model(config: dict) -> PredictiveCodingGPT2LMHeadModel:
    model_config = PredictiveCodingGPT2Config(**config["model"])
    return PredictiveCodingGPT2LMHeadModel(model_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    return parser.parse_args()


def build_training_args(config: dict, output_dir: Path) -> TrainingArguments:
    training_config = dict(config["training"])
    if not torch.cuda.is_available():
        training_config.setdefault("dataloader_pin_memory", False)

    kwargs = {
        "output_dir": str(output_dir),
        "seed": config["run"].get("seed", 42),
        "do_train": True,
        "do_eval": True,
        "report_to": "none",
        "remove_unused_columns": False,
        **training_config,
    }
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_settings = config["model"]
    model_settings["vocab_size"] = model_settings["vocab_size"] or len(tokenizer)
    model_settings["bos_token_id"] = tokenizer.bos_token_id
    model_settings["eos_token_id"] = tokenizer.eos_token_id
    model_settings["pad_token_id"] = tokenizer.pad_token_id

    output_dir = Path(config["run"]["output_dir"])
    dataset_cache = get_dataset_cache_dir(config)
    if dataset_cache_exists(dataset_cache):
        print(f"Using cached tokenized dataset at {dataset_cache}", flush=True)
        tokenized = load_tokenized_cache(dataset_cache)
    else:
        ensure_free_disk_space(dataset_cache, float(config["dataset"].get("min_free_disk_gb", 5.0)))
        datasets = load_text_datasets(config["dataset"])
        tokenized = tokenize_for_causal_lm(datasets, tokenizer, config["dataset"])

    model = build_model(config)

    if config["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    training_args = build_training_args(config, output_dir)
    metrics_file = output_dir / "metrics.jsonl"

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = PredictiveCodingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[JsonlMetricsCallback(metrics_file)],
    )
    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; saving checkpoint-latest before exit.", flush=True)
    finally:
        latest_dir = output_dir / "checkpoint-latest"
        trainer.save_model(latest_dir)
        tokenizer.save_pretrained(latest_dir)

    if interrupted:
        raise SystemExit(130)

    final_dir = output_dir / "checkpoint-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
