from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]

WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "outputs" / "dashboard"
DATASET_PROGRESS_FILE = LOG_DIR / "dataset_progress.json"
PROCESSES: dict[str, subprocess.Popen | None] = {"train": None, "dataset": None}


class DashboardHandler(SimpleHTTPRequestHandler):
    metrics_file: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        metrics_file = resolve_metrics_file(self.metrics_file)
        if parsed.path == "/api/metrics":
            self._write_json(read_metrics(metrics_file))
            return
        if parsed.path == "/api/status":
            self._write_json(
                {
                    "metrics_file": str(metrics_file),
                    "exists": metrics_file.exists(),
                    "bytes": metrics_file.stat().st_size if metrics_file.exists() else 0,
                    "processes": process_snapshot(),
                    "dataset_progress": read_dataset_progress(),
                    "checkpoints": checkpoint_snapshot(metrics_file.parent),
                }
            )
            return
        if parsed.path == "/api/logs":
            self._write_json(read_logs())
            return
        if parsed.path == "/api/checkpoints":
            self._write_json(list_checkpoints())
            return
        if parsed.path == "/api/dataset/cache":
            query = parse_qs(parsed.query)
            self._write_json(dataset_cache_snapshot(query.get("config", ["configs/smoke.yaml"])[0]))
            return
        if parsed.path == "/api/model/inspect":
            query = parse_qs(parsed.query)
            self._write_json(
                model_inspection(
                    query.get("config", ["configs/smoke.yaml"])[0],
                    query_overrides(query, ("n_layer", "n_embd", "n_head", "n_positions")),
                )
            )
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json_body()

        if parsed.path == "/api/train/start":
            self._write_json(start_train(body))
            return
        if parsed.path == "/api/train/stop":
            self._write_json(stop_process("train"))
            return
        if parsed.path == "/api/dataset/collect":
            self._write_json(start_dataset(body))
            return
        if parsed.path == "/api/generate":
            self._write_json(generate_text(body))
            return

        self._write_json({"ok": False, "error": "Unknown endpoint"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_json(self, payload, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def read_metrics(metrics_file: str | Path) -> list[dict]:
    metrics_file = Path(metrics_file)
    if not metrics_file.exists():
        return []

    events = []
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def resolve_metrics_file(metrics_file: str | Path) -> Path:
    metrics_file = Path(metrics_file)
    if metrics_file.exists():
        return metrics_file

    candidates = sorted(
        [path for path in (ROOT / "outputs").glob("*/metrics.jsonl") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else metrics_file


def process_snapshot() -> dict[str, dict]:
    return {name: process_state(process) for name, process in PROCESSES.items()}


def process_state(process: subprocess.Popen | None) -> dict:
    if process is None:
        return {"running": False, "returncode": None, "pid": None}
    return {"running": process.poll() is None, "returncode": process.poll(), "pid": process.pid}


def read_logs() -> dict[str, str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logs = {}
    for name in ("train", "dataset"):
        path = LOG_DIR / f"{name}.log"
        if path.exists():
            logs[name] = path.read_text(encoding="utf-8", errors="replace")[-6000:]
        else:
            logs[name] = ""
    return logs


def read_dataset_progress() -> dict:
    if not DATASET_PROGRESS_FILE.exists():
        return {"status": "idle", "stage": "not started", "percent": 0}
    try:
        return json.loads(DATASET_PROGRESS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unknown", "stage": "progress file unreadable", "percent": 0}


def start_train(body: dict) -> dict:
    config = resolve_config(body.get("config", "configs/smoke.yaml"))
    if config is None:
        return {"ok": False, "error": "Config was not found inside the project."}
    try:
        run_config = write_run_override_config(config, body.get("training", {}), body.get("model", {}))
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    DashboardHandler.metrics_file = (ROOT / load_dashboard_config(str(run_config))["run"]["output_dir"] / "metrics.jsonl").resolve()
    return start_process("train", [sys.executable, "scripts/train.py", "--config", str(run_config)])


def write_run_override_config(config: Path, training_overrides: dict, model_overrides: dict) -> Path:
    loaded = copy.deepcopy(load_dashboard_config(str(config)))
    if loaded is None:
        raise ValueError("Config was not found inside the project.")
    applied = {}
    applied.update(apply_training_overrides(loaded, training_overrides))
    applied.update(apply_model_overrides(loaded, model_overrides))

    if not applied:
        return config

    if any(key.startswith("model.") for key in applied):
        loaded["run"]["output_dir"] = scaled_output_dir(loaded)

    import yaml

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    override_path = LOG_DIR / "last_train_config.yaml"
    override_path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    return override_path


def apply_training_overrides(loaded: dict, overrides: dict) -> dict:
    if not isinstance(overrides, dict) or not overrides:
        return {}

    training = loaded.setdefault("training", {})
    applied = {}

    integer_fields = {
        "per_device_train_batch_size": "train batch size",
        "per_device_eval_batch_size": "eval batch size",
        "gradient_accumulation_steps": "gradient accumulation steps",
    }
    for field, label in integer_fields.items():
        value = overrides.get(field)
        if value in (None, ""):
            continue
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"{label} must be at least 1.")
        training[field] = parsed
        applied[f"training.{field}"] = parsed

    epochs = overrides.get("num_train_epochs")
    if epochs not in (None, ""):
        parsed_epochs = float(epochs)
        if parsed_epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        training["num_train_epochs"] = parsed_epochs
        training["max_steps"] = -1
        applied["training.num_train_epochs"] = parsed_epochs

    return applied


def apply_model_overrides(loaded: dict, overrides: dict) -> dict:
    if not isinstance(overrides, dict) or not overrides:
        return {}

    model = loaded.setdefault("model", {})
    applied = {}
    integer_fields = {
        "n_layer": "layers",
        "n_embd": "hidden size",
        "n_head": "attention heads",
        "n_positions": "context length",
    }
    for field, label in integer_fields.items():
        value = overrides.get(field)
        if value in (None, ""):
            continue
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"{label} must be at least 1.")
        model[field] = parsed
        applied[f"model.{field}"] = parsed

    if not applied:
        return {}

    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError("hidden size must be divisible by attention heads.")

    dataset = loaded.setdefault("dataset", {})
    if int(dataset.get("max_length", model["n_positions"])) > int(model["n_positions"]):
        dataset["max_length"] = int(model["n_positions"])

    return applied


def scaled_output_dir(loaded: dict) -> str:
    model = loaded["model"]
    vocab_size = int(model.get("vocab_size") or 50257)
    params_m = max(1, round(estimate_parameters(model, vocab_size) / 1_000_000))
    base = Path(loaded["run"]["output_dir"])
    stem = re.sub(r"-\d+m$", "", base.name)
    return str(Path("outputs") / f"{stem}-{params_m}m")


def start_dataset(body: dict) -> dict:
    config = resolve_config(body.get("config", "configs/smoke.yaml"))
    if config is None:
        return {"ok": False, "error": "Config was not found inside the project."}
    DATASET_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PROGRESS_FILE.write_text(
        json.dumps({"status": "queued", "stage": "queued", "percent": 0}, indent=2),
        encoding="utf-8",
    )
    return start_process(
        "dataset",
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--config",
            str(config),
            "--progress-file",
            str(DATASET_PROGRESS_FILE),
        ],
    )


def dataset_cache_snapshot(config_path: str) -> dict:
    loaded = load_dashboard_config(config_path)
    if loaded is None:
        return {"exists": False, "error": "Config was not found inside the project."}
    cache_dir = ROOT / dashboard_dataset_cache_dir(loaded)
    return {
        "exists": (cache_dir / "dataset_dict.json").exists(),
        "path": str(cache_dir),
    }


def dashboard_dataset_cache_dir(config: dict) -> Path:
    cache_dir = config.get("dataset", {}).get("cache_dir")
    if cache_dir:
        return Path(cache_dir)
    return Path(config["run"]["output_dir"]) / "dataset-cache"


def load_dashboard_config(config_path: str) -> dict | None:
    config = resolve_config(config_path)
    if config is None:
        return None
    if config.suffix.lower() == ".json":
        with config.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    import yaml

    with config.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def query_overrides(query: dict[str, list[str]], fields: tuple[str, ...]) -> dict:
    return {field: query[field][0] for field in fields if query.get(field) and query[field][0] not in ("", "config")}


def model_inspection(config_path: str, model_overrides: dict | None = None) -> dict:
    loaded = load_dashboard_config(config_path)
    if loaded is None:
        return {"ok": False, "error": "Config was not found inside the project."}
    try:
        apply_model_overrides(loaded, model_overrides or {})
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    model = loaded["model"]
    vocab_size = int(model.get("vocab_size") or 50257)
    n_positions = int(model["n_positions"])
    n_embd = int(model["n_embd"])
    n_layer = int(model["n_layer"])
    n_head = int(model["n_head"])
    pc_weight = float(model.get("predictive_coding_weight", 0.0))
    total_parameters = estimate_parameters(model, vocab_size)

    layers = [
        {
            "name": "token_embeddings",
            "type": "Embedding",
            "shape": f"{vocab_size} x {n_embd}",
            "parameters": vocab_size * n_embd,
        },
        {
            "name": "position_embeddings",
            "type": "Embedding",
            "shape": f"{n_positions} x {n_embd}",
            "parameters": n_positions * n_embd,
        },
    ]
    for index in range(n_layer):
        layers.extend(transformer_layer_summary(index, n_embd, n_head))
    layers.append({"name": "final_layer_norm", "type": "LayerNorm", "shape": str(n_embd), "parameters": 2 * n_embd})
    for index in range(max(n_layer - 1, 0)):
        layers.append(
            {
                "name": f"pc_predictor_{index}",
                "type": "LayerNorm + Linear",
                "shape": f"{n_embd} -> {n_embd}",
                "parameters": (n_embd * n_embd) + (2 * n_embd),
            }
        )
    layers.append({"name": "lm_head", "type": "Tied Linear", "shape": f"{n_embd} -> {vocab_size}", "parameters": 0})

    return {
        "ok": True,
        "parameters": total_parameters,
        "int8_size_gb": total_parameters / 1_000_000_000,
        "architecture": {
            "layers": n_layer,
            "hidden": n_embd,
            "heads": n_head,
            "context": n_positions,
            "vocab": vocab_size,
            "predictive_coding_weight": pc_weight,
        },
        "layer_stack": layers,
    }


def transformer_layer_summary(index: int, n_embd: int, n_head: int) -> list[dict]:
    return [
        {"name": f"block_{index}.ln_1", "type": "LayerNorm", "shape": str(n_embd), "parameters": 2 * n_embd},
        {
            "name": f"block_{index}.self_attention.qkv",
            "type": "Linear",
            "shape": f"{n_embd} -> {3 * n_embd}",
            "parameters": (n_embd * 3 * n_embd) + (3 * n_embd),
        },
        {
            "name": f"block_{index}.self_attention.out",
            "type": "Linear",
            "shape": f"{n_embd} -> {n_embd}",
            "parameters": (n_embd * n_embd) + n_embd,
        },
        {"name": f"block_{index}.ln_2", "type": "LayerNorm", "shape": str(n_embd), "parameters": 2 * n_embd},
        {
            "name": f"block_{index}.mlp.fc",
            "type": "Linear + GELU",
            "shape": f"{n_embd} -> {4 * n_embd}",
            "parameters": (n_embd * 4 * n_embd) + (4 * n_embd),
        },
        {
            "name": f"block_{index}.mlp.proj",
            "type": "Linear",
            "shape": f"{4 * n_embd} -> {n_embd}",
            "parameters": (4 * n_embd * n_embd) + n_embd,
        },
    ]


def estimate_parameters(model: dict, vocab_size: int) -> int:
    n_positions = int(model["n_positions"])
    n_embd = int(model["n_embd"])
    n_layer = int(model["n_layer"])
    token_embeddings = vocab_size * n_embd
    position_embeddings = n_positions * n_embd
    attention = (n_embd * 3 * n_embd) + (3 * n_embd) + (n_embd * n_embd) + n_embd
    mlp = (n_embd * 4 * n_embd) + (4 * n_embd) + (4 * n_embd * n_embd) + n_embd
    layer_norms = 4 * n_embd
    transformer_blocks = n_layer * (attention + mlp + layer_norms)
    final_layer_norm = 2 * n_embd
    pc_predictors = max(n_layer - 1, 0) * ((n_embd * n_embd) + (2 * n_embd))
    return token_embeddings + position_embeddings + transformer_blocks + final_layer_norm + pc_predictors


def stop_process(name: str) -> dict:
    process = PROCESSES.get(name)
    if process is None or process.poll() is not None:
        return {"ok": True, "message": f"{name} is not running", "processes": process_snapshot()}
    if os.name == "nt" and name == "train":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    return {"ok": True, "message": f"Stopping {name}", "processes": process_snapshot()}


def start_process(name: str, command: list[str]) -> dict:
    current = PROCESSES.get(name)
    if current is not None and current.poll() is None:
        return {"ok": False, "error": f"{name} is already running", "processes": process_snapshot()}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n--- starting {' '.join(command)} ---\n")
    log_file.flush()

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    PROCESSES[name] = process
    return {"ok": True, "message": f"Started {name}", "pid": process.pid, "log": str(log_path)}


def checkpoint_snapshot(output_dir: Path) -> dict:
    final_dir = output_dir / "checkpoint-final"
    latest_dir = output_dir / "checkpoint-latest"
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    newest = checkpoint_dirs[0] if checkpoint_dirs else None
    return {
        "output_dir": str(output_dir),
        "final_exists": checkpoint_ready(final_dir),
        "latest_exists": checkpoint_ready(latest_dir),
        "newest": str(newest) if newest else None,
        "newest_ready": checkpoint_ready(newest) if newest else False,
    }


def checkpoint_ready(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def list_checkpoints() -> list[dict]:
    outputs_dir = ROOT / "outputs"
    if not outputs_dir.exists():
        return []

    checkpoints = []
    for path in outputs_dir.glob("*/checkpoint-*"):
        if not path.is_dir() or not checkpoint_ready(path):
            continue
        relative = path.relative_to(ROOT)
        checkpoints.append(
            {
                "label": str(relative),
                "value": str(relative),
                "updated_at": path.stat().st_mtime,
            }
        )
    return sorted(checkpoints, key=lambda item: item["updated_at"], reverse=True)


def generate_text(body: dict) -> dict:
    checkpoint = resolve_project_path(body.get("checkpoint", "outputs/pc-llm-smoke/checkpoint-final"))
    prompt = str(body.get("prompt", "")).strip()
    if checkpoint is None:
        return {"ok": False, "error": "Checkpoint path must stay inside the project."}
    if not checkpoint.exists():
        return {"ok": False, "error": f"Checkpoint does not exist: {checkpoint}"}
    if not prompt:
        return {"ok": False, "error": "Prompt is empty."}

    max_new_tokens = str(int(body.get("max_new_tokens", 80)))
    use_irm = bool(body.get("irm", False))
    use_context_fuzzer = bool(body.get("context_fuzzer", False))
    irm_passes = str(max(0, int(body.get("irm_passes", 2))))
    chunk_tokens = str(max(1, int(body.get("chunk_tokens", 64))))
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "scripts/generate.py",
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-new-tokens",
        max_new_tokens,
    ]
    if use_irm:
        command.extend(["--irm", "--irm-passes", irm_passes, "--chunk-tokens", chunk_tokens])
        if use_context_fuzzer:
            command.append("--context-fuzzer")

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip()}
    return {"ok": True, "text": result.stdout.strip()}


def resolve_config(path: str) -> Path | None:
    resolved = resolve_project_path(path)
    if resolved is None or not resolved.exists() or resolved.suffix.lower() not in {".yaml", ".yml"}:
        return None
    return resolved


def resolve_project_path(path: str) -> Path | None:
    candidate = (ROOT / path).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return None
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-file",
        default="outputs/pc-llm-smoke/metrics.jsonl",
        help="Path to the JSONL metrics file written by training.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DashboardHandler.metrics_file = Path(args.metrics_file).resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Metrics:   {DashboardHandler.metrics_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
