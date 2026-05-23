from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from transformers import AutoTokenizer

from predictive_coding_llm import PredictiveCodingGPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to quantize.")
    parser.add_argument("--output", required=True, help="Directory for the int8 artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    print("Dynamic int8 export runs on CPU. Use ROCm/CUDA runtime quantizers for GPU int8 inference.", flush=True)
    model = PredictiveCodingGPT2LMHeadModel.from_pretrained(checkpoint, map_location="cpu")
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.save_pretrained(output)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
