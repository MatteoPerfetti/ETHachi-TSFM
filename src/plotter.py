from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Colors: categorical slots 1 & 2 of the default palette, recessive ink and grid
ACTUAL_COLOR = "#2a78d6"
PREDICTED_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"

MODEL_COLORS = {
    "Actual": "#1f77b4",
    "Chronos-2": "#2ca02c",
    "Timestral-209M": "#d62728",
    "TimesFM-3": "#9467bd",
    "XGBoost": "#ff7f0e",
}

MODEL_RGBA_BANDS = {
    "Chronos-2": "rgba(44, 160, 44, 0.18)",
    "Timestral-209M": "rgba(214, 39, 40, 0.18)",
    "TimesFM-3": "rgba(148, 103, 189, 0.18)",
    "XGBoost": "rgba(255, 127, 14, 0.18)",
}

MODEL_STYLES = {
    "Chronos-2": "-",
    "Timestral-209M": "--",
    "TimesFM-3": "-.",
    "XGBoost": ":",
}


def plot_forecast(y_test: pd.DataFrame, y_pred: np.ndarray | pd.DataFrame, rmse: float | None = None,
                  path: str | Path | None = None, show: bool = True) -> plt.Figure:
    """Plot y_pred against y_test over the test time horizon, one panel per target column."""
    y_pred = pd.DataFrame(np.asarray(y_pred).reshape(len(y_test), -1), index=y_test.index, columns=y_test.columns)

    n = len(y_test.columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True, squeeze=False, facecolor=SURFACE)
    for ax, col in zip(axes[:, 0], y_test.columns):
        ax.set_facecolor(SURFACE)
        ax.plot(y_test.index, y_test[col], color=ACTUAL_COLOR, linewidth=2, label="Actual")
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    return fig


def plot_benchmark_comparison(
    y_true: pd.Series | pd.DataFrame,
    forecasts: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]] | None = None,
    path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot observed values (Elia) against benchmark model forecasts using Matplotlib."""
    timestamps = y_true.index
    y_true_vals = np.asarray(y_true).squeeze()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(timestamps, y_true_vals, label="Actual (Elia)", color=MODEL_COLORS.get("Actual", "#1f77b4"), linewidth=2.5)

    for model_name, preds in forecasts.items():
        color = MODEL_COLORS.get(model_name, "#333333")
        linestyle = MODEL_STYLES.get(model_name, "-")

        median = preds.get("median")
        if median is None:
            continue

        rmse_str = ""
        if metrics and model_name in metrics and "RMSE" in metrics[model_name]:
            rmse_str = f" (RMSE: {metrics[model_name]['RMSE']:.1f} MW)"

        ax.plot(
            timestamps,
            median,
            label=f"{model_name}{rmse_str}",
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

        q10 = preds.get("q10")
        q90 = preds.get("q90")
        if q10 is not None and q90 is not None:
            ax.fill_between(
                timestamps,
                q10,
                q90,
                color=color,
                alpha=0.10,
                label=f"{model_name} 80% CI",
            )

    ax.set_title(f"Forecast Horizon ({len(timestamps)} timesteps): Actual vs Benchmark Models")
    ax.set_xlabel("Date & Time")
    ax.set_ylabel("Wind Power (MW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def create_plotly_comparison(
    y_true: pd.Series | pd.DataFrame,
    forecasts: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]] | None = None,
) -> go.Figure:
    """Create an interactive Plotly chart with toggleable forecast curves and prediction intervals."""
    timestamps = y_true.index
    y_true_vals = np.asarray(y_true).squeeze()

    fig = go.Figure()

    # Observed values
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true_vals,
            mode="lines",
            name="Actual (Elia)",
            line=dict(color=MODEL_COLORS.get("Actual", "#1f77b4"), width=3),
            hovertemplate="<b>Actual (Elia)</b><br>Date: %{x}<br>Power: %{y:.1f} MW<extra></extra>",
        )
    )

    for model_name, preds in forecasts.items():
        color = MODEL_COLORS.get(model_name, "#555555")
        rgba_band = MODEL_RGBA_BANDS.get(model_name, "rgba(100, 100, 100, 0.15)")
        median = preds.get("median")
        if median is None:
            continue

        rmse_str = ""
        if metrics and model_name in metrics and "RMSE" in metrics[model_name]:
            rmse_str = f" (RMSE: {metrics[model_name]['RMSE']:.1f} MW)"

        # 80% prediction interval (q10 -> q90), when available
        q10 = preds.get("q10")
        q90 = preds.get("q90")
        if q10 is not None and q90 is not None:
            # Upper bound q90 (transparent)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=q90,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound q10 with 'tonexty' fill
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=q10,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba_band,
                    name=f"{model_name} 80% CI",
                    hoverinfo="skip",
                )
            )

        # Main median forecast curve
        dash_style = "solid"
        if model_name == "Timestral-209M":
            dash_style = "dash"
        elif model_name == "TimesFM-3":
            dash_style = "dashdot"
        elif model_name == "XGBoost":
            dash_style = "dot"

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=median,
                mode="lines",
                name=f"{model_name}{rmse_str}",
                line=dict(color=color, width=2.2, dash=dash_style),
                hovertemplate=f"<b>{model_name}</b><br>Date: %{{x}}<br>Power: %{{y:.1f}} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title="39-hour forecast comparison: Observed vs Time Series Foundation Models",
        xaxis_title="Date and time",
        yaxis_title="Wind power (MW)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=550,
    )

    return fig
