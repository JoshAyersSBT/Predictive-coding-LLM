from __future__ import annotations

from typing import Any

from transformers import Trainer


class PredictiveCodingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        self._latest_predictive_coding_logs = {}
        for metric_name in ("lm_loss", "predictive_coding_loss"):
            value = getattr(outputs, metric_name, None)
            if value is not None:
                self._latest_predictive_coding_logs[metric_name] = float(value.detach().cpu())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        logs = {
            **getattr(self, "_latest_predictive_coding_logs", {}),
            **logs,
        }
        super().log(logs, *args, **kwargs)

