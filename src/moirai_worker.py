import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


MODEL_NAME = "Salesforce/moirai-1.1-R-large"
PATCH_SIZE = 128
NUM_SAMPLES = 100
RANDOM_SEED = 42


def forecast_moirai(
    context: pd.Series,
    horizon: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    dataset = PandasDataset({context.name: context}, freq="15min")
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(MODEL_NAME),
        prediction_length=horizon,
        context_length=len(context),
        patch_size=PATCH_SIZE,
        num_samples=NUM_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=1, device=device)
    forecast = next(predictor.predict(dataset))
    values = tuple(
        np.asarray(forecast.quantile(level), dtype=float).reshape(-1)
        for level in (0.5, 0.1, 0.9)
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Moirai in its isolated environment.")
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon must be a positive integer.")

    context_frame = pd.read_csv(args.context, parse_dates=["timestamp"])
    if list(context_frame.columns) != ["timestamp", "target"]:
        raise ValueError("Context CSV must contain timestamp and target columns.")
    context = pd.Series(
        context_frame["target"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(context_frame["timestamp"]),
        name="target",
    )
    if context.empty or not np.isfinite(context.to_numpy()).all():
        raise ValueError("Context must contain finite observations.")

    median, q10, q90 = forecast_moirai(context, args.horizon, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"median": median, "q10": q10, "q90": q90}).to_csv(
        args.output,
        index=False,
    )


if __name__ == "__main__":
    main()
