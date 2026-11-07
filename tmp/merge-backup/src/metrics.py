"""Metrics of the backtest predictions table (see src/backtest.py): RMSE of q50 and Winkler score of q10-q90.

Daily scores follow Predico: one score per forecast (issue time), averaged over the days.
"""
import numpy as np
import pandas as pd

import config

WINKLER_ALPHA = 0.2           # q10-q90 is the 80% central interval
EVAL_WINDOWS = ("all", "d+1")  # all forecast steps, or only the next day (00:00-23:45 after the issue day)


<<<<<<< HEAD
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
=======
def rmse(actual: np.ndarray | pd.Series, point: np.ndarray | pd.Series) -> float:
    """Root mean squared error of the point forecast (q50), in MW."""
    error = np.asarray(actual, dtype=float) - np.asarray(point, dtype=float)
    return float(np.sqrt(np.mean(error ** 2)))


def winkler(actual: np.ndarray | pd.Series, lower: np.ndarray | pd.Series, upper: np.ndarray | pd.Series,
            alpha: float = WINKLER_ALPHA) -> float:
    """Mean Winkler interval score, in MW: interval width, plus 2/alpha times the distance of an actual
    value outside the interval to the nearest bound. NaN if the model has no interval (e.g. persistence)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (actual, lower, upper))
    if np.isnan(lo).any() or np.isnan(hi).any():
        return float("nan")
    penalty = (2 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return float(np.mean(hi - lo + penalty))
>>>>>>> origin/Chronos-2-zero-shot


def select_window(predictions: pd.DataFrame, window: str = config.BACKTEST_EVAL_WINDOW) -> pd.DataFrame:
    """Rows of the evaluation window with a known actual value."""
    if window not in EVAL_WINDOWS:
        raise ValueError(f"window must be one of {EVAL_WINDOWS}, got {window!r}")
    rows = predictions[predictions["actual"].notna()]
    if window == "d+1":
        next_day = rows["issue_time"].dt.normalize() + pd.Timedelta(days=1)
        rows = rows[rows["timestamp"].dt.normalize() == next_day]
    return rows


def _scores(rows: pd.DataFrame) -> pd.Series:
    return pd.Series({"RMSE": rmse(rows["actual"], rows["q50"]),
                      "Winkler80": winkler(rows["actual"], rows["q10"], rows["q90"])})


def daily_scores(predictions: pd.DataFrame, window: str = config.BACKTEST_EVAL_WINDOW) -> pd.DataFrame:
    """RMSE and Winkler80 of every forecast: index (model, issue_time)."""
    rows = select_window(predictions, window)
    return rows.groupby(["model", "issue_time"])[["actual", "q10", "q50", "q90"]].apply(_scores)


def summary(predictions: pd.DataFrame, window: str = config.BACKTEST_EVAL_WINDOW) -> pd.DataFrame:
    """Per model: daily RMSE and Winkler80 averaged over the days (Predico), number of days,
    daily RMSE relative to persistence (if it was run). Sorted by RMSE."""
    rows = select_window(predictions, window)
    daily = daily_scores(predictions, window).groupby("model").mean()
    table = pd.DataFrame({
        "RMSE": daily["RMSE"],
        "Winkler80": daily["Winkler80"],
        "days": rows.groupby("model")["issue_time"].nunique(),
    })
    if "Persistence" in table.index:
        table["RMSE_vs_persistence"] = table["RMSE"] / table.loc["Persistence", "RMSE"] - 1
    return table.sort_values("RMSE")


SUMMARY_COLUMNS = {  # summary() column -> printed header
    "RMSE": "RMSE",
    "Winkler80": "Winkler 80%",
    "RMSE_vs_persistence": "vs Persistence",
    "days": "days",
}


def format_summary(table: pd.DataFrame, title: str, subtitle: str = "") -> str:
    """Plain-text table of `summary()`: best score of each column marked with *, missing scores as '-'."""
    columns = [key for key in SUMMARY_COLUMNS if key in table.columns]

    starred = ("RMSE", "Winkler80")

    def cell(key: str, value: float) -> str:
        if pd.isna(value):
            return "-" + ("  " if key in starred else "")
        if key == "days":
            return f"{int(value)}"
        if key == "RMSE_vs_persistence":
            return f"{value:+.0%}" if value != 0 else "ref"
        best = np.isclose(value, table[key].min())
        return f"{value:,.1f}" + (" *" if best else "  ")

    cells = {model: [cell(key, table.loc[model, key]) for key in columns] for model in table.index}
    name_width = max(len("Model"), *(len(model) for model in table.index))
    widths = [max(len(SUMMARY_COLUMNS[key]) + (2 if key in starred else 0),
                  *(len(cells[model][i]) for model in table.index)) for i, key in enumerate(columns)]

    def row(name: str, values: list[str]) -> str:
        return " " + name.ljust(name_width) + "".join("   " + v.rjust(w) for v, w in zip(values, widths))

    # Headers of the starred columns end two characters early, so they line up with the numbers, not the stars
    header = row("Model", [SUMMARY_COLUMNS[key] + ("  " if key in starred else "") for key in columns])
    width = max(len(header), len(title) + 1, len(subtitle) + 1)
    out = ["=" * width, f" {title}"] + ([f" {subtitle}"] if subtitle else []) + ["-" * width, header, "-" * width]
    out += [row(model, cells[model]) for model in table.index]
    out += ["-" * width,
            " MW, lower is better, * = best.",
            " RMSE, Winkler 80%: score of each forecast, averaged over the days (Predico).",
            " vs Persistence: RMSE relative to persistence.",
            "=" * width]
    return "\n".join(out)


def print_summary(predictions: pd.DataFrame, run_name: str, window: str = config.BACKTEST_EVAL_WINDOW) -> pd.DataFrame:
    """Print the metrics table of a backtest run and return it."""
    table = summary(predictions, window)
    rows = select_window(predictions, window)
    steps = rows.groupby(["model", "issue_time"]).size().max()
    first, last = rows["issue_time"].min(), rows["issue_time"].max()
    title = f"Backtest {run_name}: {table['days'].max()} forecasts issued {first:%Y-%m-%d} .. {last:%Y-%m-%d}"
    subtitle = f"evaluation window: {window} ({steps} steps per forecast)"
    print(format_summary(table, title, subtitle))
    return table
