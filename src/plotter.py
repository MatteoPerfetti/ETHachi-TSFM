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


def plot_forecast(y_test: pd.DataFrame, y_pred: np.ndarray | pd.DataFrame, rmse: float | None = None,
                  path: str | Path | None = None, show: bool = True) -> plt.Figure:
    """Plot y_pred against y_test over the test time horizon, one panel per target column.

    y_pred may be the raw model.predict output; it takes its timestamps and column names from y_test.
    If `path` is given the figure is saved there; `show` opens it in a window.
    """
    y_pred = pd.DataFrame(np.asarray(y_pred).reshape(len(y_test), -1), index=y_test.index, columns=y_test.columns)

    n = len(y_test.columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True, squeeze=False, facecolor=SURFACE)
    for ax, col in zip(axes[:, 0], y_test.columns):
        ax.set_facecolor(SURFACE)
        ax.plot(y_test.index, y_test[col], color=ACTUAL_COLOR, linewidth=2, label="Actual")
        # Dashed line: prediction is distinguishable without relying on color alone
        ax.plot(y_pred.index, y_pred[col], color=PREDICTED_COLOR, linewidth=2, linestyle="--", label="Predicted")
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
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    return fig
