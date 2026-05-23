from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class PredictiveCodingSSMConfig(PretrainedConfig):
    model_type = "predictive-coding-ssm"

    def __init__(
        self,
        vocab_size: int | None = 50257,
        n_positions: int = 256,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        ssm_kernel_size: int = 4,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        predictive_coding_weight: float = 0.05,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size or 50257
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.ssm_kernel_size = ssm_kernel_size
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.predictive_coding_weight = predictive_coding_weight


@dataclass
class PredictiveCodingSSMOutput(CausalLMOutputWithCrossAttentions):
    predictive_coding_loss: torch.FloatTensor | None = None
    lm_loss: torch.FloatTensor | None = None


class GatedSSMBlock(nn.Module):
    def __init__(self, config: PredictiveCodingSSMConfig):
        super().__init__()
        hidden = config.n_embd
        kernel_size = max(2, int(config.ssm_kernel_size))
        self.in_norm = nn.LayerNorm(hidden)
        self.in_proj = nn.Linear(hidden, 2 * hidden)
        self.depthwise_conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden,
        )
        self.decay_logit = nn.Parameter(torch.zeros(hidden))
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.in_norm(hidden_states)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = F.silu(value)
        conv = self.depthwise_conv(value.transpose(1, 2))
        conv = conv[:, :, : value.shape[1]].transpose(1, 2)
        state = self._scan(conv)
        mixed = torch.sigmoid(gate) * state
        hidden_states = residual + self.dropout(self.out_proj(mixed))
        return hidden_states + self.ff(self.ff_norm(hidden_states))

    def _scan(self, values: torch.Tensor) -> torch.Tensor:
        decay = torch.sigmoid(self.decay_logit).view(1, 1, -1)
        state = torch.zeros(values.shape[0], 1, values.shape[2], device=values.device, dtype=values.dtype)
        outputs = []
        for index in range(values.shape[1]):
            state = (decay * state) + ((1.0 - decay) * values[:, index : index + 1, :])
            outputs.append(state)
        return torch.cat(outputs, dim=1)


class PredictiveCodingSSMLMHeadModel(PreTrainedModel):
    config_class = PredictiveCodingSSMConfig
    base_model_prefix = "pc_ssm"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PredictiveCodingSSMConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList(GatedSSMBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.pc_predictors = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.n_embd),
                nn.Linear(config.n_embd, config.n_embd, bias=False),
            )
            for _ in range(max(config.n_layer - 1, 0))
        )
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.wte = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> PredictiveCodingSSMOutput | tuple:
        del attention_mask, kwargs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            raise ValueError("input_ids are required for PredictiveCodingSSMLMHeadModel.")
        if input_ids.shape[1] > self.config.n_positions:
            input_ids = input_ids[:, -self.config.n_positions :]
            if labels is not None:
                labels = labels[:, -self.config.n_positions :]

        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        all_hidden_states = [hidden_states]

        for block in self.blocks:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        lm_loss = self._causal_lm_loss(logits, labels) if labels is not None else None
        pc_loss = self._predictive_coding_loss(tuple(all_hidden_states))

        loss = lm_loss
        if loss is not None and pc_loss is not None:
            loss = loss + self.config.predictive_coding_weight * pc_loss
        elif pc_loss is not None:
            loss = self.config.predictive_coding_weight * pc_loss

        hidden_tuple = tuple(all_hidden_states) if output_hidden_states else None
        if not return_dict:
            values = (logits, hidden_tuple)
            return ((loss,) + values) if loss is not None else values

        return PredictiveCodingSSMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_tuple,
            predictive_coding_loss=pc_loss,
            lm_loss=lm_loss,
        )

    def _causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
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
