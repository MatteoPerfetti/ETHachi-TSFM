"""Naive reference forecasts: a model is only useful if it beats these."""
import numpy as np

from src.forecast import Forecaster, ForecastRequest, ForecastResult


class Persistence(Forecaster):
    """Repeats the last `model_horizon` observed values: "the next 39 h look like the last 39 h".

    Point forecast only (q50); q10 and q90 stay NaN, so no interval metrics. No training, no covariates.
    min_context is set by the backtest engine to the model_horizon it needs.
    """
    name = "Persistence"

    def __init__(self, min_context: int):
        self.min_context = min_context

    def predict(self, requests: list[ForecastRequest]) -> list[ForecastResult]:
        results = []
        for request in requests:
            if len(request.context) < request.model_horizon:
                raise ValueError(f"{self.name}: context of {len(request.context)} steps, "
                                 f"needs at least model_horizon = {request.model_horizon}")
            values = request.context.to_numpy(dtype=float)[-request.model_horizon:]
            results.append(ForecastResult.from_model_output(self.name, request, {0.5: values}))
        return results
