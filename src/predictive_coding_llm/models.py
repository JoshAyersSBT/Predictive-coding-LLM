from __future__ import annotations

import json
from pathlib import Path

from .modeling_predictive_coding_gpt2 import PredictiveCodingGPT2Config, PredictiveCodingGPT2LMHeadModel
from .modeling_predictive_coding_ssm import PredictiveCodingSSMConfig, PredictiveCodingSSMLMHeadModel


def architecture_name(model_config: dict) -> str:
    value = str(model_config.get("architecture") or model_config.get("model_type") or "gpt").lower()
    if value in {"predictive-coding-ssm", "ssm", "state-space", "state_space"}:
        return "ssm"
    return "gpt"


def build_model_from_config(model_config: dict):
    model_config = dict(model_config)
    architecture = architecture_name(model_config)
    model_config.pop("architecture", None)
    if architecture == "ssm":
        return PredictiveCodingSSMLMHeadModel(PredictiveCodingSSMConfig(**model_config))
    return PredictiveCodingGPT2LMHeadModel(PredictiveCodingGPT2Config(**model_config))


def load_model_from_checkpoint(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            model_type = json.load(handle).get("model_type")
        if model_type == PredictiveCodingSSMConfig.model_type:
            return PredictiveCodingSSMLMHeadModel.from_pretrained(checkpoint)
    return PredictiveCodingGPT2LMHeadModel.from_pretrained(checkpoint)
