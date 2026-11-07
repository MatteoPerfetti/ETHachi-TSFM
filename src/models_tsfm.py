from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from timestral import TimestralForecaster
from timesfm import TimesFM3Forecaster

import config
from logger import logger


@dataclass
class TSFMResult:
    """Standard container for foundation model predictions."""
    median: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    model_name: str

    def to_dataframe(self, index: pd.Index | None = None) -> pd.DataFrame:
        """Return predictions as a pandas DataFrame."""
        df = pd.DataFrame(
            {
                f"{self.model_name}_median": self.median,
                f"{self.model_name}_q10": self.q10,
                f"{self.model_name}_q90": self.q90,
            },
            index=index,
        )
        return df


class TSFMRunner:
    """Unified adapter for loading and running Chronos, Timestral, and TimesFM."""

    @staticmethod
    def get_default_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def load_model(cls, model_key: str, device: str | None = None) -> Any:
        """Load and return a model from the checkpoint configured in MODELS_REGISTRY."""
        if device is None:
            device = cls.get_default_device()

        model_info = config.MODELS_REGISTRY.get(model_key)
        if not model_info or model_info.get("type") != "tsfm":
            raise ValueError(f"Model '{model_key}' was not found or is not of type 'tsfm' in MODELS_REGISTRY")

        checkpoint = model_info["checkpoint"]
        provider = model_info.get("provider", "")

        logger.info("Loading model weights for %s (%s) on %s...", model_key, checkpoint, device)

        if "chronos" in model_key or provider == "amazon":
            return BaseChronosPipeline.from_pretrained(checkpoint, device_map=device)
        elif "timestral" in model_key or provider == "mistral":
            return TimestralForecaster.from_pretrained(checkpoint, device=device)
        elif "timesfm" in model_key or provider == "google":
            return TimesFM3Forecaster.from_pretrained(checkpoint, device=device)
        else:
            raise ValueError(f"Unrecognized provider or architecture for {model_key}")

    @classmethod
    def run_chronos(
        cls,
        y_context: pd.Series | pd.DataFrame,
        horizon: int = config.HORIZON_STEPS,
        model: BaseChronosPipeline | None = None,
        weather_covariates: pd.DataFrame | None = None,
        device: str | None = None,
    ) -> TSFMResult:
        """Run inference with Chronos-2."""
        if model is None:
            model = cls.load_model("chronos2", device=device)

        # Format the context DataFrame for Chronos
        if isinstance(y_context, pd.Series):
            context_df = y_context.to_frame(name="target").reset_index()
        else:
            context_df = y_context.iloc[:, [0]].copy().reset_index()
            context_df.columns = ["timestamp", "target"]

        context_df.columns = ["timestamp", "target"]
        context_df["item_id"] = "wind_belgium"
        context_df = context_df[["item_id", "timestamp", "target"]]

        # Remove timezone information for Chronos compatibility
        context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
        if context_df["timestamp"].dt.tz is not None:
            context_df["timestamp"] = context_df["timestamp"].dt.tz_localize(None)

        pred_chronos = model.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        median = pred_chronos["0.5"].to_numpy().astype(float)
        q10 = pred_chronos["0.1"].to_numpy().astype(float)
        q90 = pred_chronos["0.9"].to_numpy().astype(float)

        return TSFMResult(median=median, q10=q10, q90=q90, model_name="Chronos-2")

    @classmethod
    def run_timestral(
        cls,
        y_context: pd.Series | pd.DataFrame,
        horizon: int = config.HORIZON_STEPS,
        model: TimestralForecaster | None = None,
        weather_covariates: pd.DataFrame | None = None,
        device: str | None = None,
    ) -> TSFMResult:
        """Run inference with Timestral-209M."""
        if model is None:
            model = cls.load_model("timestral_209m", device=device)

        # Input shape for Timestral: (series=1, variates=1, time)
        y_values = y_context.values.squeeze().reshape(1, 1, -1)

        forecast = model.forecast(
            y_values,
            prediction_length=horizon,
        )

        median = np.asarray(forecast.median.squeeze(), dtype=float)
        q10 = np.asarray(forecast.quantile(0.1).squeeze(), dtype=float)
        q90 = np.asarray(forecast.quantile(0.9).squeeze(), dtype=float)

        return TSFMResult(median=median, q10=q10, q90=q90, model_name="Timestral-209M")

    @classmethod
    def run_timesfm(
        cls,
        y_context: pd.Series | pd.DataFrame,
        horizon: int = config.HORIZON_STEPS,
        model: TimesFM3Forecaster | None = None,
        weather_covariates: pd.DataFrame | None = None,
        device: str | None = None,
    ) -> TSFMResult:
        """Run inference with TimesFM-3.0."""
        if model is None:
            model = cls.load_model("timesfm3", device=device)

        # Input: 1D float32 NumPy array for TimesFM
        context_timesfm = np.asarray(y_context.values.squeeze(), dtype=np.float32)

        output = model.predict(
            context_timesfm,
            horizon=horizon,
            return_quantiles=True,
        )

        median = np.asarray(output.forecast, dtype=float).squeeze()
        q10 = np.asarray(output.quantiles[:, 0], dtype=float).squeeze()
        q90 = np.asarray(output.quantiles[:, 8], dtype=float).squeeze()

        return TSFMResult(median=median, q10=q10, q90=q90, model_name="TimesFM-3")

    @classmethod
    def run_model(
        cls,
        model_key: str,
        y_context: pd.Series | pd.DataFrame,
        horizon: int = config.HORIZON_STEPS,
        model: Any | None = None,
        weather_covariates: pd.DataFrame | None = None,
        device: str | None = None,
    ) -> TSFMResult:
        """Dispatch inference to any supported registered TSFM."""
        if model_key == "chronos2":
            return cls.run_chronos(y_context, horizon, model=model, weather_covariates=weather_covariates, device=device)
        elif model_key == "timestral_209m":
            return cls.run_timestral(y_context, horizon, model=model, weather_covariates=weather_covariates, device=device)
        elif model_key == "timesfm3":
            return cls.run_timesfm(y_context, horizon, model=model, weather_covariates=weather_covariates, device=device)
        else:
            raise ValueError(f"Inference for model '{model_key}' is not implemented in TSFMRunner")

