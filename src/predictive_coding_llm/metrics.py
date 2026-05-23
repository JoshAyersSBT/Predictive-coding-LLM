from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_file: str | Path):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file.write_text("", encoding="utf-8")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        event = {
            "time": time.time(),
            "step": state.global_step,
            "epoch": state.epoch,
            **_json_safe(logs),
        }
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe

