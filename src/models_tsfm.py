from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

import config
from logger import logger
from src.forecast import QUANTILE_COLUMNS, Forecaster, ForecastRequest, ForecastResult
from src.pca import CovariatePCA, fit_covariate_pca

TSFM_BATCH_SIZE = 64  # requests per model call; lower it if the device runs out of memory


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_numpy(values: Any) -> np.ndarray:
    """Torch tensor (any device) or array-like -> float numpy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


class TSFMAdapter(Forecaster):
    """Common part of the TSFM adapters: loads the weights once, splits requests into batches, times inference.

    Subclasses implement `_load` and `_predict_batch`; every request in a batch must have the same
    model_horizon and either no covariates or the same covariate columns.
    """
    model_key: str
    uses_covariates = True

    def __init__(self, device: str | None = None, batch_size: int = TSFM_BATCH_SIZE,
                 checkpoint: str | None = None, model: Any = None, name: str | None = None,
                 pca_components: int | float | None = None):
        self.device = device or default_device()
        self.batch_size = batch_size
        self.pca_components = pca_components
        self.pca: CovariatePCA | None = None
        checkpoint = checkpoint or config.MODELS_REGISTRY[self.model_key]["checkpoint"]
        if name is not None:
            self.name = name
        start = time.perf_counter()
        self.model = self._load(checkpoint) if model is None else model
        if model is not None:
            self.max_context = (model.model_context_length if self.model_key == "chronos2"
                                else model.max_context if self.model_key == "timestral_209m"
                                else model.global_context)
        self.load_seconds = time.perf_counter() - start
        self.inference_seconds = 0.0
        logger.info("Loaded %s (%s) on %s in %.1fs, max context %s steps",
                    self.name, checkpoint, self.device, self.load_seconds, self.max_context)

    def fit(self, target: pd.Series, covariates: pd.DataFrame | None = None) -> None:
        """Zero-shot: only the covariate PCA is fitted (on the training period), the weights stay untouched.

        TODO fine-tuning: train the pretrained weights here on the history given by the backtest engine
        (target + covariates of the training period), then keep using `predict` unchanged.
        """
        self.pca = fit_covariate_pca(self.pca_components, covariates)

    def predict(self, requests: list[ForecastRequest]) -> list[ForecastResult]:
        if self.pca is not None:
            requests = [self.pca.transform_request(request) for request in requests]
        results = []
        for first in range(0, len(requests), self.batch_size):
            batch = requests[first:first + self.batch_size]
            self._check_batch(batch)
            start = time.perf_counter()
            outputs = self._predict_batch(batch)
            self.inference_seconds += time.perf_counter() - start
            results += [ForecastResult.from_model_output(self.name, request, values)
                        for request, values in zip(batch, outputs)]
            logger.info("%s: %d/%d requests forecast", self.name, len(results), len(requests))
        return results

    @staticmethod
    def _check_batch(batch: list[ForecastRequest]) -> None:
        if len({request.model_horizon for request in batch}) > 1:
            raise ValueError("all requests in a batch must have the same model_horizon")
        columns = {None if r.past_covariates is None else tuple(r.past_covariates.columns) for r in batch}
        if len(columns) > 1:
            raise ValueError("all requests in a batch must have the same covariate columns (or none)")

    def _load(self, checkpoint: str) -> Any:
        raise NotImplementedError

    def _predict_batch(self, batch: list[ForecastRequest]) -> list[dict[float, np.ndarray]]:
        """Raw output per request: quantile level -> array over request.model_index."""
        raise NotImplementedError


# ------------- Chronos-2 (amazon/chronos-2, chronos-forecasting) -------------
# - Input: one long DataFrame (item_id, timestamp, target, covariates...) for the whole batch, one item_id per request,
#   timestamps without timezone. One predict_df call per batch.
# - Covariates: columns in the context frame that also appear in future_df (over model_index) are known future
#   covariates; columns only in the context frame would be past-only.
# - Output: one column per requested quantile level ("0.1", "0.5", "0.9").
# - Max context 8192 steps (pipeline.model_context_length), longer contexts are cut to the most recent steps.
def _to_chronos_long(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    """Frame indexed by timestamp -> Chronos long format (item_id, timestamp, ...) with naive timestamps."""
    out = df.rename_axis("timestamp").reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    out.insert(0, "item_id", item_id)
    return out


class Chronos2Adapter(TSFMAdapter):
    name = "Chronos-2"
    model_key = "chronos2"

    def _load(self, checkpoint: str) -> BaseChronosPipeline:
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(checkpoint, device_map=self.device)
        self.max_context = pipeline.model_context_length
        return pipeline

    def _predict_batch(self, batch: list[ForecastRequest]) -> list[dict[float, np.ndarray]]:
        contexts, futures = [], []
        for i, request in enumerate(batch):
            context = request.context.rename("target").to_frame()
            if request.past_covariates is not None:
                context = context.join(request.past_covariates)
            contexts.append(_to_chronos_long(context, str(i)))
            if request.future_covariates is not None:
                futures.append(_to_chronos_long(request.future_covariates, str(i)))

        levels = list(QUANTILE_COLUMNS)
        pred = self.model.predict_df(
            pd.concat(contexts, ignore_index=True),
            future_df=pd.concat(futures, ignore_index=True) if futures else None,
            prediction_length=batch[0].model_horizon,
            quantile_levels=levels,
        )
        by_item = dict(list(pred.groupby("item_id", sort=False)))
        return [{level: by_item[str(i)][str(level)].to_numpy() for level in levels} for i in range(len(batch))]


# ------------- Timestral-209M (mistralai/Timestral-209M, timestral) -------------
# - Input: context tensor (B, V=1, T); every context in a batch must have the same length T.
# - Covariates: known_covariates (B, C, T + H) = past and future weather concatenated along time;
#   past_covariates (B, C, T) if a request has no future values.
# - Output: Forecast.quantiles (B, V, H, 21) on levels 0.01..0.99; Forecast.quantile(level) picks one level.
# - clip_nonnegative (library default) clamps at 0 the series whose context is non-negative.
# - Max context 32768 steps (forecaster.max_context), the oldest steps are dropped beyond it.
class TimestralAdapter(TSFMAdapter):
    name = "Timestral-209M"
    model_key = "timestral_209m"

    def _load(self, checkpoint: str) -> TimestralForecaster:
        from timestral import TimestralForecaster

        forecaster = TimestralForecaster.from_pretrained(checkpoint, device=self.device)
        self.max_context = forecaster.max_context
        return forecaster

    def _predict_batch(self, batch: list[ForecastRequest]) -> list[dict[float, np.ndarray]]:
        if len({len(request.context) for request in batch}) > 1:
            raise ValueError(f"{self.name}: all contexts in a batch must have the same length")
        context = np.stack([request.context.to_numpy(np.float32) for request in batch])[:, None, :]

        covariates = {}
        if batch[0].future_covariates is not None:
            covariates["known_covariates"] = np.stack([
                np.concatenate([r.past_covariates.to_numpy(np.float32), r.future_covariates.to_numpy(np.float32)]).T
                for r in batch])
        elif batch[0].past_covariates is not None:
            covariates["past_covariates"] = np.stack([r.past_covariates.to_numpy(np.float32).T for r in batch])

        forecast = self.model.forecast(context, prediction_length=batch[0].model_horizon, **covariates)
        per_level = {level: _to_numpy(forecast.quantile(level))[:, 0, :] for level in QUANTILE_COLUMNS}
        return [{level: values[i] for level, values in per_level.items()} for i in range(len(batch))]


# ------------- TimesFM-3 (google/timesfm-3.0-pytorch, timesfm) -------------
# - Input: list of 1D float32 contexts, one per request (lengths may differ); predict_batch yields one output each.
# - Covariates: past_future_covariates (C, T + H) = past and future weather concatenated along time;
#   past_only_covariates (C, T) if a request has no future values.
# - Output: ForecastOutput.quantiles (H, 9) on levels 0.1..0.9 (model.config.quantiles);
#   ForecastOutput.forecast is the 0.5 quantile.
# - Max context 15360 steps (forecaster.global_context).
class TimesFMAdapter(TSFMAdapter):
    name = "TimesFM-3"
    model_key = "timesfm3"

    def _load(self, checkpoint: str) -> TimesFM3Forecaster:
        from timesfm import TimesFM3Forecaster

        forecaster = TimesFM3Forecaster.from_pretrained(checkpoint, device=self.device)
        self.max_context = forecaster.global_context
        return forecaster

    def _predict_batch(self, batch: list[ForecastRequest]) -> list[dict[float, np.ndarray]]:
        contexts = [request.context.to_numpy(np.float32) for request in batch]
        covariates = {}
        if batch[0].future_covariates is not None:
            covariates["past_future_covariates"] = [
                np.concatenate([r.past_covariates.to_numpy(np.float32), r.future_covariates.to_numpy(np.float32)]).T
                for r in batch]
        elif batch[0].past_covariates is not None:
            covariates["past_only_covariates"] = [r.past_covariates.to_numpy(np.float32).T for r in batch]

        outputs = self.model.predict_batch(contexts, horizon=batch[0].model_horizon, return_quantiles=True,
                                           **covariates)
        index = {level: self.model.config.quantiles.index(level) for level in QUANTILE_COLUMNS}
        return [{level: _to_numpy(output.quantiles[:, i]) for level, i in index.items()} for output in outputs]


# Model key (config.MODELS_REGISTRY) -> adapter class
TSFM_ADAPTERS: dict[str, type[TSFMAdapter]] = {
    "chronos2": Chronos2Adapter,
    "timestral_209m": TimestralAdapter,
    "timesfm3": TimesFMAdapter,
}


def load_tsfm(model_key: str, device: str | None = None, pca_components: int | float | None = None) -> TSFMAdapter:
    """Create the adapter of a TSFM registered in config.MODELS_REGISTRY (loads the weights)."""
    if model_key not in TSFM_ADAPTERS:
        raise ValueError(f"No TSFM adapter for '{model_key}', available: {list(TSFM_ADAPTERS)}")
    return TSFM_ADAPTERS[model_key](device=device, pca_components=pca_components)
