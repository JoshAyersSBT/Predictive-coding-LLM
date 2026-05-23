# Predictive Coding LLM

A local research scaffold for training and testing causal language models with a predictive-coding auxiliary objective. The stack includes:

- Hugging Face dataset collection and tokenization.
- A browser dashboard at `http://127.0.0.1:7860`.
- GPT-style and SSM-style model variants with the same scaling controls.
- Live train/eval convergence charts.
- Checkpoint selection and prompt testing.
- Optional iterative reasoning/context-fuzzer generation experiments.

## Requirements

- Python 3.10 or newer.
- Enough disk space for Hugging Face datasets and tokenized caches.
- CPU works for smoke tests. Real training needs a CUDA or ROCm PyTorch build.

For AMD/ROCm systems, install the ROCm-specific PyTorch wheel from the official PyTorch selector before large runs. PyTorch exposes ROCm devices through `torch.cuda`, so a healthy ROCm install should report `torch.cuda.is_available() == True` and a non-empty `torch.version.hip`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Install the dependency stack:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

For ROCm on Linux, install PyTorch first with `--no-cache-dir`, then install the rest of the stack:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
python -m pip install -r requirements-rocm.txt
python -m pip install -e .
```

The `--no-cache-dir` flag matters because ROCm PyTorch wheels are very large and can trigger `ValueError: Memoryview is too large` when pip tries to cache/process them. `requirements-rocm.txt` intentionally contains only the non-PyTorch project dependencies.

The command above uses the PyTorch ROCm 6.4 wheel index. If your driver/runtime requires a different ROCm wheel family, replace `rocm6.4` with the matching PyTorch index from the official selector before installing. Verify the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## Launch Dashboard

Start the web UI:

```powershell
python scripts/dashboard.py --metrics-file outputs/pc-llm-smoke/metrics.jsonl
```

Open:

```text
http://127.0.0.1:7860
```

From the dashboard you can collect datasets, start/stop training, scale the model, choose GPT attention or SSM state-space architecture, inspect parameter counts/layer stacks, and test checkpoints.

## Common Workflow

Collect/tokenize the configured dataset:

```powershell
python scripts/prepare_dataset.py --config configs/smoke.yaml
```

Train from the terminal:

```powershell
python scripts/train.py --config configs/smoke.yaml
```

Generate from a checkpoint:

```powershell
python scripts/generate.py --checkpoint outputs/pc-llm-smoke-125m/checkpoint-final --prompt "Explain Python decorators" --max-new-tokens 160
```

Estimate model size:

```powershell
python scripts/estimate_params.py --config configs/4b-int8.yaml
```

Export a CPU dynamic-int8 checkpoint:

```powershell
python scripts/export_int8.py --checkpoint outputs/pc-llm-4b/checkpoint-final --output outputs/pc-llm-4b-int8
```

## Configs

- `configs/smoke.yaml`: TinyStories smoke test.
- `configs/kimi-k25-smoke.yaml`: KIMI K2.5 `General-Distillation`, `train[:50%]`.
- `configs/4b-int8.yaml`: larger 4B-class target shape.

Useful config fields:

- `model.architecture`: `gpt` or `ssm`.
- `model.n_layer`: layer count.
- `model.n_embd`: hidden width.
- `model.n_head`: attention heads for GPT, SSM groups for SSM.
- `model.n_positions`: context length.
- `training.precision`: `auto` selects bf16/fp16 on CUDA/ROCm when available.
- `dataset.preprocessing_num_workers`: `auto` or an integer for parallel tokenization.

Tokenized datasets are cached under the configured output/cache directory. If the cache exists, training reuses it instead of downloading and tokenizing again.

## Predictive Coding Objective

The model keeps the normal next-token language modeling loss and adds a local hidden-state prediction term:

```text
loss = lm_loss + predictive_coding_weight * mean(mse(predicted_next_hidden, stopgrad(actual_next_hidden)))
```

In the GPT variant, hidden states come from transformer blocks. In the SSM variant, hidden states come from gated state-space blocks. Both variants use the same dashboard scaling controls and checkpoint/test flow.
