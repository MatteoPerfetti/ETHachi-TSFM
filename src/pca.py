"""PCA condensation of the weather covariates, shared by the traditional and the TSFM adapters.

The PCA is fitted on the training covariates only and reused unchanged for every forecast, so no test-period
information leaks into the components. It always runs on a covariate frame that is already assembled for the model
(after build_origin_aligned_dataset for the traditional models), so whatever selection of the weather run happened
there is not affected.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from logger import logger
from src.forecast import ForecastRequest


class CovariatePCA:
    """Standardize `columns` and replace them by their principal components; all other columns pass through."""

    def __init__(self, n_components: int | float, columns: list[str]) -> None:
        self.columns = list(columns)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full")

    @property
    def component_columns(self) -> list[str]:
        return [f"{config.PCA_PREFIX}_{i}" for i in range(1, self.pca.n_components_ + 1)]

    def fit(self, frame: pd.DataFrame) -> "CovariatePCA":
        """Fit on the rows of `frame` that have every covariate."""
        rows = frame[self.columns].dropna()
        self.pca.fit(self.scaler.fit_transform(rows))
        logger.info("PCA: %d covariates -> %d components explaining %.1f%% of the variance (%d rows)",
                    len(self.columns), self.pca.n_components_, 100 * self.pca.explained_variance_ratio_.sum(),
                    len(rows))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """`frame` without the covariate columns, followed by the principal components."""
        components = pd.DataFrame(self.pca.transform(self.scaler.transform(frame[self.columns])),
                                  index=frame.index, columns=self.component_columns)
        return pd.concat([frame.drop(columns=self.columns), components], axis=1)

    def transform_request(self, request: ForecastRequest) -> ForecastRequest:
        """Request with past and future covariates replaced by their principal components."""
        if request.past_covariates is None:
            return request
        future = None if request.future_covariates is None else self.transform(request.future_covariates)
        return replace(request, past_covariates=self.transform(request.past_covariates), future_covariates=future)


def fit_covariate_pca(n_components: int | float | None, covariates: pd.DataFrame | None,
                      frame: pd.DataFrame | None = None) -> CovariatePCA | None:
    """PCA fitted on `frame` (default: `covariates`) for the columns of `covariates`, None if PCA is off."""
    if n_components is None or covariates is None:
        return None
    return CovariatePCA(n_components, list(covariates.columns)).fit(covariates if frame is None else frame)
