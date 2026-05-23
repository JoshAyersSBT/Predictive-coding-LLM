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
from predictive_coding_llm.hardware import default_device, print_accelerator_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--irm", action="store_true", help="Use iterative reasoning mode with rolling context.")
    parser.add_argument("--context-fuzzer", action="store_true", help="Use fitted token-trend context instead of rolling text.")
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
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
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
    completion_ids = output_ids[0][prompt_tokens:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def rolling_tail(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def token_ids(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def token_trend_summary(tokenizer: AutoTokenizer, text: str) -> str:
    ids = token_ids(tokenizer, text)
    if len(ids) < 2:
        return "not enough prior generated tokens for a trend"

    count = len(ids)
    x_mean = (count - 1) / 2
    y_mean = sum(ids) / count
    denominator = sum((index - x_mean) ** 2 for index in range(count)) or 1
    slope = sum((index - x_mean) * (token_id - y_mean) for index, token_id in enumerate(ids)) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [token_id - (intercept + slope * index) for index, token_id in enumerate(ids)]
    mae = sum(abs(value) for value in residuals) / count
    direction = "rising" if slope > 0.1 else "falling" if slope < -0.1 else "flat"
    low = min(ids)
    high = max(ids)
    return (
        f"prior token count={count}; least-squares token id fit: id ~= {intercept:.2f} + {slope:.4f}*position; "
        f"trend={direction}; mean_abs_residual={mae:.2f}; token_id_range=[{low}, {high}]"
    )


def generate_with_irm(
    model: PredictiveCodingGPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    passes: int,
    chunk_tokens: int,
    context_fuzzer: bool = False,
) -> str:
    max_positions = int(getattr(model.config, "n_positions", 256))
    context_budget = max(16, max_positions - max(8, chunk_tokens))
    scratch = ""

    for index in range(0 if context_fuzzer else max(0, passes)):
        draft_prompt = (
            "system: Private planning pass. Use the user request as the only task.\n"
            f"user: {prompt}\n"
            "assistant: Draft concise implementation notes. Do not answer a different task.\n"
            f"Prior notes: {scratch}\n"
            "Notes:"
        )
        draft = generate_once(model, tokenizer, draft_prompt, min(chunk_tokens, max_new_tokens), temperature=0.7, top_p=0.9)
        scratch = rolling_tail(tokenizer, draft, context_budget // 2)

    visible = ""
    last_batch = ""
    remaining = max(1, max_new_tokens)
    while remaining > 0:
        next_tokens = min(chunk_tokens, remaining)
        if context_fuzzer:
            final_prompt = fuzzer_prompt(tokenizer, prompt, scratch, visible, last_batch)
        else:
            visible_tail = rolling_tail(tokenizer, visible, max(8, context_budget // 2))
            notes_tail = rolling_tail(tokenizer, scratch, max(8, context_budget // 4))
            final_prompt = (
                "system: Answer the user request directly. Stay on task. Do not quote private notes.\n"
                f"user: {prompt}\n"
                f"assistant: {visible_tail}\n"
                f"Private notes summary, do not quote: {notes_tail}\n"
                "assistant:"
            )
        chunk = generate_once(model, tokenizer, final_prompt, next_tokens, temperature=0.8, top_p=0.95)
        piece = clean_irm_output(chunk)
        if not piece:
            break
        visible = f"{visible} {piece}".strip()
        last_batch = piece
        remaining -= next_tokens
        if piece.endswith((".", "!", "?", "\n")) and remaining <= chunk_tokens:
            break
    return visible


def fuzzer_prompt(tokenizer: AutoTokenizer, prompt: str, scratch: str, visible: str, last_batch: str) -> str:
    trend = token_trend_summary(tokenizer, visible)
    last_hint = rolling_tail(tokenizer, last_batch, 64)
    return (
        "system: You must answer only the original user request. Do not invent a new task. "
        "The prior answer is not provided as full text; it is represented by numeric continuity metadata "
        "plus the last generated batch.\n"
        f"user_original_prompt: {prompt}\n"
        f"math_approximation_of_prior_tokens: {trend}\n"
        f"last_visible_batch: {last_hint}\n"
        "instruction: Continue the answer to user_original_prompt. If the prompt asks for code, provide code.\n"
        "assistant:"
    )


def clean_irm_output(text: str) -> str:
    blocked_prefixes = (
        "private planning",
        "private notes",
        "notes:",
        "user request:",
        "user:",
        "system:",
        "assistant:",
        "<think>",
        "</think>",
        "let me think",
        "the user is asking",
        "visible answer",
        "continue the visible answer",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(stripped.lower().startswith(prefix) for prefix in blocked_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PredictiveCodingGPT2LMHeadModel.from_pretrained(checkpoint)
    model.to(default_device())
    model.eval()
    print_accelerator_summary()

    if args.irm:
        text = generate_with_irm(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.irm_passes,
            args.chunk_tokens,
            context_fuzzer=args.context_fuzzer,
        )
    else:
        text = generate_once(model, tokenizer, format_user_prompt(args.prompt), args.max_new_tokens)

    sys.stdout.write(text)
    sys.stdout.write("\n")


def format_user_prompt(prompt: str) -> str:
    stripped = prompt.strip()
    if stripped.lower().startswith(("user:", "system:", "assistant:")):
        return stripped
    return f"user: {stripped}\nassistant:"


if __name__ == "__main__":
    main()
