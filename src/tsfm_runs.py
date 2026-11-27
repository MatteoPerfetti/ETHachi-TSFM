"""Local TSFM run records and artifact identities."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wind_path() -> Path:
    import pandas as pd
    start, end = pd.Timestamp(config.DATA_START), pd.Timestamp(config.DATA_END)
    return config.DATA_DIR / "wind" / f"{config.RESOLUTIONCODE}_{start:%Y%m%d}_{end:%Y%m%dT}.parquet"


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(record, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_record(run_dir: Path) -> dict:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record.get("status", "completed") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")
    if not record.get("base_checkpoint") or not record.get("base_revision"):
        raise ValueError(f"Run does not identify its base revision: {run_dir}")
    if not record.get("reloaded_validation"):
        raise ValueError(f"Run has no successful adapter reload verification: {run_dir}")
    return record


def training_records(root: Path | None = None) -> dict[str, dict]:
    """Read training records, including failed and interrupted runs."""
    root = root or config.BACKTEST_RESULTS_DIR / "finetuning"
    runs = {}
    for path in sorted(root.glob("*/*/run.json")):
        model, run_id = path.parent.parent.name, path.parent.name
        if model not in ("chronos2", "timestral_209m", "delta_y"):
            continue
        key = f"delta:{run_id}" if model == "delta_y" else f"lora:{model}:{run_id}"
        try:
            runs[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return runs


def available_adapters(root: Path | None = None) -> dict[str, dict]:
    """Find verified training runs; preserve their original manifests and paths."""
    root = root or config.BACKTEST_RESULTS_DIR / "finetuning"
    runs = {}
    for key in training_records(root):
        if not key.startswith("lora:"):
            continue
        _, model, run_id = key.split(":", 2)
        run_dir = root / model / run_id
        try:
            record = read_record(run_dir)
            if record.get("base_checkpoint") != config.MODELS_REGISTRY[model]["checkpoint"]:
                continue
            artifact = run_dir / record.get("adapter_path", "finetuned-ckpt" if record.get("backend") == "peft" else "adapter.safetensors")
            if not artifact.exists():
                continue
            runs[key] = record
        except (OSError, ValueError, KeyError):
            continue
    return runs


def available_delta_adapters(root: Path | None = None) -> dict[str, dict]:
    """Completed δY output adapters with a reload check, keyed delta:<run-id> (the backtest model key)."""
    root = root or config.BACKTEST_RESULTS_DIR / "finetuning"
    runs = {}
    for path in sorted((root / "delta_y").glob("*/run.json")):
        try:
            record = read_record(path.parent)
            if (path.parent / "adapter.safetensors").is_file() and record.get("base_model_key"):
                runs[f"delta:{path.parent.name}"] = record
        except (OSError, ValueError, KeyError):
            continue
    return runs


def start_training(run_dir: Path, model: str, args, checkpoint: str, revision: str, adapter: str,
                   covariates: list[str] | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(config.__file__).parent,
                                             text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(config.__file__).parent,
                                                 text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            commit = None
            dirty = None
        packages = {}
        for name in ("torch", "chronos-forecasting", "timestral", "peft", "transformers", "pandas"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        source = wind_path()
        record = {"schema_version": 1, "run_id": args.run_id, "model": model, "status": "running",
                  "started_at": now(), "git_commit": commit, "git_dirty": dirty, "packages": packages,
                  "base_checkpoint": checkpoint, "base_revision": revision, "adapter_path": adapter,
                  "parameters": vars(args).copy(), "dataset": {"path": str(source),
                  "sha256": fingerprint(source), "resolution": config.RESOLUTIONCODE},
                  "covariates": list(covariates or [])}
        if covariates:
            weather = Path(config.OPENMETEO_SOURCES["dwd"]["path"])
            record["dataset"]["weather"] = {"path": str(weather), "sha256": fingerprint(weather)}
        write_record(run_dir / "run.json", record)
        return record
    except BaseException:
        write_record(run_dir / "run.json", {"schema_version": 1, "run_id": args.run_id, "model": model,
                                            "status": "failed", "finished_at": now()})
        raise


def finish_training(run_dir: Path, record: dict, status: str, error: str | None = None) -> None:
    record.update(status=status, finished_at=now())
    if error is not None:
        record["error"] = error
    write_record(run_dir / "run.json", record)
    try:
        from src.experiment_tracking import track_training
        track_training(run_dir, record)
    except Exception as exc:
        from logger import logger
        logger.warning("Training saved locally, but MLflow tracking failed: %s", exc)
