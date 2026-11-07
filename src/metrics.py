import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series | None = None,
    q90: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calculate point forecast metrics (RMSE, MAE, WAPE) and 80% prediction interval coverage.

    Parameters:
        y_true: observed values (MW)
        y_pred: model predictions (median / point forecast) (MW)
        q10: 10th percentile (lower bound of the 80% prediction interval) [optional]
        q90: 90th percentile (upper bound of the 80% prediction interval) [optional]

    Returns:
        Dictionary of calculated metrics:
        - "RMSE": Root Mean Squared Error (in MW)
        - "MAE": Mean Absolute Error (in MW)
        - "WAPE": Weighted Absolute Percentage Error (in %)
        - "Coverage_80CI": PICP - Prediction Interval Coverage Probability (in %, when q10 and q90 are provided)
    """
    y_t = np.asarray(y_true, dtype=np.float64).squeeze()
    y_p = np.asarray(y_pred, dtype=np.float64).squeeze()

    rmse = float(root_mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))

    # WAPE: sum of absolute errors divided by the sum of absolute observations (%)
    total_energy = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(np.abs(y_t - y_p)) / total_energy * 100.0) if total_energy > 0 else 0.0

    results = {
        "RMSE": rmse,
        "MAE": mae,
        "WAPE": wape,
    }

    # Prediction Interval Coverage Probability (PICP) for the 80% interval (q10 -> q90)
    if q10 is not None and q90 is not None:
        q_low = np.asarray(q10, dtype=np.float64).squeeze()
        q_high = np.asarray(q90, dtype=np.float64).squeeze()
        inside = (y_t >= q_low) & (y_t <= q_high)
        results["Coverage_80CI"] = float(np.mean(inside) * 100.0)

    return results


def metrics_summary_table(all_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert a {model_name: {metric: value}} dictionary into a summary DataFrame."""
    df = pd.DataFrame.from_dict(all_metrics, orient="index")
    # Order columns when present
    cols_order = [c for c in ["RMSE", "MAE", "WAPE", "Coverage_80CI"] if c in df.columns]
    return df[cols_order]

