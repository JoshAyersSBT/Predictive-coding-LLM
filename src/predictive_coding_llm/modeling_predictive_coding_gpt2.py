from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class PredictiveCodingGPT2Config(GPT2Config):
    model_type = "predictive-coding-gpt2"

    def __init__(self, predictive_coding_weight: float = 0.05, **kwargs):
        kwargs.setdefault("loss_type", "ForCausalLMLoss")
        super().__init__(**kwargs)
        self.loss_type = self.loss_type or "ForCausalLMLoss"
        self.predictive_coding_weight = predictive_coding_weight


@dataclass
class PredictiveCodingCausalLMOutput(CausalLMOutputWithCrossAttentions):
    predictive_coding_loss: torch.FloatTensor | None = None
    lm_loss: torch.FloatTensor | None = None


class PredictiveCodingGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = PredictiveCodingGPT2Config

    def __init__(self, config: PredictiveCodingGPT2Config):
        super().__init__(config)
        self.pc_predictors = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.n_embd),
                nn.Linear(config.n_embd, config.n_embd, bias=False),
            )
            for _ in range(max(config.n_layer - 1, 0))
        )
        self.post_init()

    def forward(self, *args, **kwargs) -> PredictiveCodingCausalLMOutput:
        kwargs["output_hidden_states"] = True
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        kwargs["return_dict"] = True

        outputs = super().forward(*args, **kwargs)
        pc_loss = self._predictive_coding_loss(outputs.hidden_states)

        loss = outputs.loss
        lm_loss = loss
        if loss is not None and pc_loss is not None:
            loss = loss + self.config.predictive_coding_weight * pc_loss
        elif pc_loss is not None:
            loss = self.config.predictive_coding_weight * pc_loss

        if not return_dict:
            as_tuple = outputs.to_tuple()
            return ((loss,) + as_tuple[1:]) if loss is not None else as_tuple

        return PredictiveCodingCausalLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            predictive_coding_loss=pc_loss,
            lm_loss=lm_loss,
        )

    def _predictive_coding_loss(
        self,
        hidden_states: tuple[torch.FloatTensor, ...] | None,
    ) -> torch.FloatTensor | None:
        if hidden_states is None or len(hidden_states) < 3:
            return None

        losses = []
        block_hidden_states = hidden_states[1:-1]
        for predictor, current_hidden, next_hidden in zip(
            self.pc_predictors,
            block_hidden_states[:-1],
            block_hidden_states[1:],
            strict=False,
        ):
            predicted_next = predictor(current_hidden)
            losses.append(F.mse_loss(predicted_next, next_hidden.detach()))

        if not losses:
            return None
        return torch.stack(losses).mean()
