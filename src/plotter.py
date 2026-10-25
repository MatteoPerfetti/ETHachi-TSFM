from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colors: categorical slots 1 & 2 of the default palette, recessive ink and grid
ACTUAL_COLOR = "#2a78d6"
PREDICTED_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_forecast(
    y_test: pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    rmse: float | None = None,
    path: str | Path | None = None,
    show: bool = True,
    lower: np.ndarray | pd.Series | None = None,
    upper: np.ndarray | pd.Series | None = None,
) -> plt.Figure:
    """Plot y_pred against y_test over the test time horizon, one panel per target column.

    y_pred may be the raw model.predict output; it takes its timestamps and column names from y_test.
    If `path` is given the figure is saved there; `show` opens it in a window.
    """
    y_pred = pd.DataFrame(np.asarray(y_pred).reshape(len(y_test), -1), index=y_test.index, columns=y_test.columns)
    lower_frame = None if lower is None else pd.DataFrame(
        np.asarray(lower).reshape(len(y_test), -1),
        index=y_test.index,
        columns=y_test.columns,
    )
    upper_frame = None if upper is None else pd.DataFrame(
        np.asarray(upper).reshape(len(y_test), -1),
        index=y_test.index,
        columns=y_test.columns,
    )

    n = len(y_test.columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True, squeeze=False, facecolor=SURFACE)
    for ax, col in zip(axes[:, 0], y_test.columns):
        ax.set_facecolor(SURFACE)
        ax.plot(y_test.index, y_test[col], color=ACTUAL_COLOR, linewidth=2, label="Actual")
        # Dashed line: prediction is distinguishable without relying on color alone
        ax.plot(y_pred.index, y_pred[col], color=PREDICTED_COLOR, linewidth=2, linestyle="--", label="Q50")
        if lower_frame is not None and upper_frame is not None:
            ax.fill_between(
                y_test.index,
                lower_frame[col],
                upper_frame[col],
                color=PREDICTED_COLOR,
                alpha=0.18,
                label="Q10-Q90",
            )
        ax.set_title(col, loc="left", color=TEXT_PRIMARY, fontsize=11)
        ax.set_ylabel("MW", color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY)

    title = "Forecast vs. actual (test period)"
    if rmse is not None:
        title += f"  ·  RMSE {rmse:,.1f} MW"
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    return fig

# SARIMAX ROLLING FORECASTS

def plot_rolling_forecasts(
    results: pd.DataFrame,
    rmse: float | None = None,
    timezone: str | None = None,
    path: str | Path | None = None,
    show: bool = True,
    forecast_name: str = "SARIMAX",
    lower_column: str | None = None,
    upper_column: str | None = None,
    title_note: str | None = "weather proxy; not vintage-verified",
) -> plt.Figure:
    """Plot every rolling forecast horizon against the actual series."""
    data = results.reset_index().copy()

    actual = (
        data[["target_time", "actual"]]
        .dropna(subset=["actual"])
        .drop_duplicates(subset="target_time")
        .sort_values("target_time")
    )

    fig, ax = plt.subplots(
        figsize=(14, 6),
        facecolor=SURFACE,
    )

    ax.set_facecolor(SURFACE)

    # Plot every rolling forecast as a separate trajectory.
    for number, (origin, fold) in enumerate(
        data.groupby("forecast_origin", sort=True)
    ):
        fold = fold.sort_values("target_time")

        ax.plot(
            fold["target_time"],
            fold["predicted"],
            color=PREDICTED_COLOR,
            linewidth=1.5,
            alpha=0.55,
            label=(
                f"Rolling {forecast_name} forecasts"
                if number == 0
                else None
            ),
        )

        if lower_column is not None and upper_column is not None:
            ax.fill_between(
                fold["target_time"],
                fold[lower_column],
                fold[upper_column],
                color=PREDICTED_COLOR,
                alpha=0.10,
                label="Q10-Q90" if number == 0 else None,
            )

        ax.scatter(
            fold["target_time"].iloc[0],
            fold["predicted"].iloc[0],
            color=PREDICTED_COLOR,
            s=22,
            alpha=0.8,
            zorder=3,
        )

    ax.plot(
        actual["target_time"],
        actual["actual"],
        color=ACTUAL_COLOR,
        linewidth=2.4,
        label="Actual",
        zorder=4,
    )

    title = f"Rolling forecast horizons · {forecast_name}"

    if title_note is not None:
        title += f" ({title_note})"

    if rmse is not None:
        title += f" · Overall RMSE {rmse:,.1f} MW"

    ax.set_title(
        title,
        loc="left",
        color=TEXT_PRIMARY,
        fontsize=13,
    )

    ax.set_xlabel(
        f"Target time"
        + (f" ({timezone})" if timezone else "")
    )
    ax.set_ylabel("MW", color=TEXT_SECONDARY)

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.8,
    )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color(GRID_COLOR)

    ax.legend(
        loc="upper right",
        frameon=False,
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=150,
            facecolor=SURFACE,
        )

    if show:
        plt.show()

    return fig
