from __future__ import annotations

import torch


def accelerator_summary() -> dict[str, str | bool | int | None]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "backend": "cpu",
            "device_count": 0,
            "device_name": None,
            "hip": getattr(torch.version, "hip", None),
            "cuda": torch.version.cuda,
        }

    return {
        "available": True,
        "backend": "rocm" if getattr(torch.version, "hip", None) else "cuda",
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
    }


def print_accelerator_summary() -> None:
    info = accelerator_summary()
    if not info["available"]:
        print("Accelerator: CPU only. Install a CUDA or ROCm PyTorch build for GPU training.", flush=True)
        return
    print(
        "Accelerator: "
        f"{info['backend']} device='{info['device_name']}' "
        f"count={info['device_count']} hip={info['hip']} cuda={info['cuda']}",
        flush=True,
    )


def resolve_auto_precision(training_config: dict) -> None:
    precision = training_config.pop("precision", None)
    if precision != "auto" or not torch.cuda.is_available():
        return

    training_config["fp16"] = False
    training_config["bf16"] = False
    if safe_bf16_supported():
        training_config["bf16"] = True
    else:
        training_config["fp16"] = True


def safe_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
