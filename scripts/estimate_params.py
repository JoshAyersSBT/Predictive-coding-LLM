from __future__ import annotations

import argparse

from predictive_coding_llm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML model config.")
    return parser.parse_args()


def estimate_gpt2_parameters(model_config: dict) -> int:
    vocab_size = int(model_config["vocab_size"] or 50257)
    n_positions = int(model_config["n_positions"])
    n_embd = int(model_config["n_embd"])
    n_layer = int(model_config["n_layer"])

    token_embeddings = vocab_size * n_embd
    position_embeddings = n_positions * n_embd

    attention = (n_embd * 3 * n_embd) + (3 * n_embd) + (n_embd * n_embd) + n_embd
    mlp = (n_embd * 4 * n_embd) + (4 * n_embd) + (4 * n_embd * n_embd) + n_embd
    layer_norms = 4 * n_embd
    transformer_blocks = n_layer * (attention + mlp + layer_norms)

    final_layer_norm = 2 * n_embd
    pc_predictors = max(n_layer - 1, 0) * ((n_embd * n_embd) + (2 * n_embd))

    return token_embeddings + position_embeddings + transformer_blocks + final_layer_norm + pc_predictors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = dict(config["model"])
    parameters = estimate_gpt2_parameters(model_config)
    print(f"parameters: {parameters:,}")
    print(f"trainable:   {parameters:,}")
    print(f"int8 size:   {parameters / 1_000_000_000:.2f} GB before metadata/scale overhead")


if __name__ == "__main__":
    main()
