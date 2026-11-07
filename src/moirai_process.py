import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def run_moirai_process(
    y_context: pd.Series | pd.DataFrame,
    horizon: int,
    device: str,
) -> pd.DataFrame:
    """Run the Moirai worker and return its validated forecast."""
    interpreter = os.environ.get("MOIRAI_PYTHON")
    if not interpreter:
        raise RuntimeError(
            "Moirai requires a separate Python environment. "
            "Set MOIRAI_PYTHON to that environment's Python executable."
        )

    python_executable = shutil.which(interpreter)
    if python_executable is None:
        interpreter_path = Path(interpreter).expanduser()
        if not interpreter_path.is_file():
            raise FileNotFoundError(
                f"MOIRAI_PYTHON does not identify a Python executable: {interpreter}"
            )
        python_executable = str(interpreter_path.resolve())

    if isinstance(y_context, pd.Series):
        context = y_context
    elif y_context.shape[1] > 0:
        context = y_context.iloc[:, 0]
    else:
        raise ValueError("Moirai context must contain a target column.")

    worker_path = Path(__file__).with_name("moirai_worker.py")
    with tempfile.TemporaryDirectory(prefix="ethachi-moirai-") as temp_dir:
        temp_path = Path(temp_dir)
        context_path = temp_path / "context.csv"
        forecast_path = temp_path / "forecast.csv"
        context.rename("target").rename_axis("timestamp").to_frame().to_csv(context_path)

        completed = subprocess.run(
            [
                python_executable,
                str(worker_path),
                "--context",
                str(context_path),
                "--output",
                str(forecast_path),
                "--horizon",
                str(horizon),
                "--device",
                device,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Moirai worker failed with exit code {completed.returncode}"
                + (f":\n{details}" if details else ".")
            )
        if not forecast_path.is_file():
            raise RuntimeError("Moirai worker completed without writing its forecast.")

        forecast = pd.read_csv(forecast_path)

    required_columns = {"median", "q10", "q90"}
    missing_columns = required_columns - set(forecast.columns)
    if missing_columns:
        raise ValueError(
            f"Moirai worker output is missing columns: {sorted(missing_columns)}"
        )
    if len(forecast) != horizon:
        raise ValueError(
            f"Moirai returned {len(forecast)} points; expected {horizon}."
        )
    for column in ("median", "q10", "q90"):
        forecast[column] = forecast[column].to_numpy(dtype=float)
        if not np.isfinite(forecast[column].to_numpy()).all():
            raise ValueError("Moirai worker output contains non-finite forecast values.")

    return forecast
