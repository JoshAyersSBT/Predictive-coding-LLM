from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoTokenizer

from predictive_coding_llm import PredictiveCodingGPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--irm", action="store_true", help="Use iterative reasoning mode with rolling context.")
    parser.add_argument("--irm-passes", type=int, default=2, help="Number of hidden reasoning draft passes.")
    parser.add_argument("--chunk-tokens", type=int, default=64, help="Visible tokens to generate per rolling chunk.")
    return parser.parse_args()


def generate_once(
    model: PredictiveCodingGPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    requested_new_tokens: int,
    *,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    max_positions = int(getattr(model.config, "n_positions", 256))
    requested_new_tokens = max(1, int(requested_new_tokens))
    max_prompt_tokens = max(1, max_positions - requested_new_tokens)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    available_new_tokens = max(1, max_positions - prompt_tokens)
    max_new_tokens = min(requested_new_tokens, available_new_tokens)

    if max_new_tokens < requested_new_tokens:
        print(
            f"[generation note] Clamped new tokens from {requested_new_tokens} to {max_new_tokens} "
            f"for this checkpoint's {max_positions}-token context window.",
            file=sys.stderr,
        )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def rolling_tail(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def strip_prompt_prefix(prompt: str, generated: str) -> str:
    return generated[len(prompt) :].lstrip() if generated.startswith(prompt) else generated.strip()


def generate_with_irm(
    model: PredictiveCodingGPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    passes: int,
    chunk_tokens: int,
) -> str:
    max_positions = int(getattr(model.config, "n_positions", 256))
    context_budget = max(16, max_positions - max(8, chunk_tokens))
    scratch = ""

    for index in range(max(0, passes)):
        draft_prompt = (
            f"{prompt}\n\n"
            f"[IRM private draft {index + 1}] Build a compact internal plan, constraints, and likely answer. "
            "Do not write the final answer yet.\n"
            f"{scratch}"
        )
        draft = generate_once(model, tokenizer, draft_prompt, min(chunk_tokens, max_new_tokens), temperature=0.7, top_p=0.9)
        scratch = rolling_tail(tokenizer, strip_prompt_prefix(draft_prompt, draft), context_budget // 2)

    visible = ""
    remaining = max(1, max_new_tokens)
    while remaining > 0:
        next_tokens = min(chunk_tokens, remaining)
        context = rolling_tail(tokenizer, f"{prompt}\n\n{scratch}\n\n{visible}", context_budget)
        final_prompt = (
            f"{context}\n\n"
            "[IRM final] Continue the visible answer only. Do not include private draft notes.\n"
        )
        chunk = generate_once(model, tokenizer, final_prompt, next_tokens, temperature=0.8, top_p=0.95)
        piece = strip_prompt_prefix(final_prompt, chunk)
        visible = f"{visible} {piece}".strip()
        remaining -= next_tokens
        if piece.endswith((".", "!", "?", "\n")) and remaining <= chunk_tokens:
            break
    return visible


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PredictiveCodingGPT2LMHeadModel.from_pretrained(checkpoint)
    model.eval()

    if args.irm:
        text = generate_with_irm(model, tokenizer, args.prompt, args.max_new_tokens, args.irm_passes, args.chunk_tokens)
    else:
        text = generate_once(model, tokenizer, args.prompt, args.max_new_tokens)

    sys.stdout.write(text)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
