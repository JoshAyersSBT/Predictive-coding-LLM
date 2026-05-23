# Predictive Coding LLM

A compact research scaffold for training a causal language model with a
predictive-coding-inspired auxiliary objective.

The first runnable target is intentionally small so the pipeline can be tested on
one machine. The included `4b-int8` config describes the intended larger model
shape and post-training int8 export path.

## What This Builds

- Hugging Face dataset loading and tokenization.
- A GPT-style causal language model.
- Predictive-coding auxiliary losses between neighboring hidden layers.
- Training with Hugging Face `Trainer`.
- Optional post-training dynamic int8 export for CPU inference.
- Separate configs for a smoke-test model and a 4B-class target.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

For GPU training, install the PyTorch build that matches your accelerator before
installing this package. NVIDIA uses a CUDA build. AMD GPUs use a ROCm build.
PyTorch exposes ROCm devices through the normal `torch.cuda` API, so the training
scripts will still report and use a `cuda` device when ROCm is installed.

On Linux with a supported AMD GPU, install a ROCm PyTorch wheel from the official
PyTorch selector for your ROCm version. Then verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

The configs use `training.precision: auto`. On ROCm/CUDA this selects bf16 when
PyTorch reports bf16 support, otherwise fp16. On CPU it leaves precision alone.

## Smoke Test Training

Start the dashboard in one terminal:

```powershell
python scripts/dashboard.py --metrics-file outputs/pc-llm-smoke/metrics.jsonl
```

Open `http://127.0.0.1:7860`. From the dashboard you can:

- collect/tokenize the configured Hugging Face dataset;
- start and stop training;
- watch train, eval, LM, and predictive-coding losses;
- test a saved checkpoint with a prompt.

You can also run training directly from a terminal:

```powershell
python scripts/train.py --config configs/smoke.yaml
```

Additional dataset configs are available:

```powershell
python scripts/prepare_dataset.py --config configs/kimi-k25-smoke.yaml
python scripts/train.py --config configs/kimi-k25-smoke.yaml
```

`configs/kimi-k25-smoke.yaml` uses the `General-Distillation` subset of
`ianncity/KIMI-K2.5-1000000x` with `train[:50%]` by default. The source dataset is
chat-formatted, so the pipeline flattens its `messages` column into role-tagged
training text.

The smoke config uses `roneneldan/TinyStories` and a tiny model so you can verify
that dataset loading, tokenization, training, and checkpoint writing work.
The dashboard is available at `http://127.0.0.1:7860` and refreshes every two
seconds as `metrics.jsonl` changes.

Dataset tokenization runs in parallel by default. Set
`dataset.preprocessing_num_workers` to `auto` or an integer in the config to
control how many Hugging Face `datasets.map` workers are used.
If `outputs/<run>/dataset-cache/dataset_dict.json` already exists, collection and
training reuse that tokenized cache instead of downloading/tokenizing again.

## 4B-Class Target

Check the parameter count:

```powershell
python scripts/estimate_params.py --config configs/4b-int8.yaml
```

The current target is approximately 4.1B parameters, including the auxiliary
predictive-coding heads.

```powershell
python scripts/train.py --config configs/4b-int8.yaml
```

This config is a hardware-scale target, not a laptop default. It defines roughly
a 4B parameter transformer before quantization. Real training will require
multi-GPU distributed execution, mixed precision, gradient checkpointing, and a
larger dataset than the smoke test.

After training, export an int8 copy:

```powershell
python scripts/export_int8.py --checkpoint outputs/pc-llm-4b/checkpoint-final --output outputs/pc-llm-4b-int8
```

Dynamic int8 export is useful for CPU inference and artifact size. For GPU int8
inference, use a serving stack such as bitsandbytes, Quanto, AutoGPTQ, AWQ, or
the target deployment runtime's quantizer.

`scripts/export_int8.py` performs CPU dynamic quantization. That is not the right
path for ROCm GPU inference; use a ROCm-compatible inference or quantization
runtime for GPU-side quantized serving.

## Predictive Coding Objective

The model keeps the standard next-token cross entropy loss. In addition, every
transformer block has a small predictor that tries to predict the next block's
hidden state from the current block's hidden state. The auxiliary objective is:

```text
loss = lm_loss + predictive_coding_weight * mean(mse(predicted_next_hidden, actual_next_hidden))
```

The actual next hidden state is detached for this auxiliary term, so each layer is
encouraged to become locally predictive without turning the auxiliary loss into a
second global backpropagation path through all upper layers.
