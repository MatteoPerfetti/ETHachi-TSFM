from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

import config
from logger import logger
from src.metrics import rmse, select_window

# Colors: recessive ink, grid and surface of the default palette
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID_COLOR = "#e4e3df"
AXIS_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"

# Backtest plot: one fixed color per model (the color follows the model, whatever subset is plotted),
# categorical slots of the default palette in order; the actual values are always in primary ink.
BACKTEST_COLORS = {
    "Chronos-2": "#2a78d6",
    "Timestral-209M": "#eb6834",
    "TimesFM-3": "#1baf7a",
    "Persistence": "#eda100",
}
FALLBACK_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]  # next slots, for models not listed above


def model_colors(models: list[str]) -> dict[str, str]:
    """Fixed color of each model; models not in BACKTEST_COLORS take the next palette slots."""
    fallback = iter(FALLBACK_COLORS)
    return {m: BACKTEST_COLORS.get(m) or next(fallback, TEXT_SECONDARY) for m in models}


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def plot_backtest(predictions: pd.DataFrame, run_name: str,
                  start: str = config.BACKTEST_PLOT_START, end: str = config.BACKTEST_PLOT_END,
                  models: list[str] | None = config.BACKTEST_PLOT_MODELS,
                  path: str | Path | None = None, show: bool = False) -> plt.Figure:
    """Forecast vs actual over [start, end), one panel per model, from the backtest predictions table.

    Each panel stitches the D+1 part (next day 00:00-23:45) of consecutive daily forecasts, so every timestamp
    is shown once: a jump at midnight is the switch to the next forecast. Line = q50, band = q10-q90 (if any).
    Panels are ordered by RMSE in the window; `path` = None saves to PLOT_DIRECTORY/backtest_<run>_<start>_<end>.png.
    """
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rows = select_window(predictions, "d+1")
    rows = rows[(rows["timestamp"] >= start_ts) & (rows["timestamp"] < end_ts)]
    available = rows["model"].unique().tolist()
    models = available if models is None else [m for m in models if m in available]
    if not models:
        raise ValueError(f"no forecasts in {start}..{end} for models {models}, available: {available}")

    panels = {model: rows[rows["model"] == model].sort_values("timestamp") for model in models}
    errors = {model: rmse(d["actual"], d["q50"]) for model, d in panels.items()}
    order = sorted(models, key=errors.get)
    colors = model_colors(order)

    fig, axes = plt.subplots(len(order), 1, figsize=(13, 2.5 * len(order) + 1), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for ax, model in zip(axes[:, 0], order):
        d = panels[model]
        t = d["timestamp"].dt.tz_localize(None)
        ax.set_facecolor(SURFACE)
        if d["q10"].notna().all() and d["q90"].notna().all():
            ax.fill_between(t, d["q10"], d["q90"], color=colors[model], alpha=0.18, linewidth=0, label="q10–q90")
        ax.plot(t, d["actual"], color=TEXT_PRIMARY, linewidth=1.4, label="Actual (Elia)")
        ax.plot(t, d["q50"], color=colors[model], linewidth=2, label="Forecast q50")

        ax.set_title(f"{model}   ·   RMSE D+1 in this window {errors[model]:,.0f} MW", loc="left",
                     color=TEXT_PRIMARY, fontsize=11)
        ax.set_ylabel("Power [MW]", color=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        # Legend on the title line, outside the data area
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=9,
                  labelcolor=TEXT_SECONDARY, ncol=3, borderaxespad=0.2)

    days = (end_ts - start_ts).days
    locator = mdates.DayLocator(interval=max(1, days // 8))
    axes[-1, 0].set_xlabel("Time [UTC]", color=TEXT_SECONDARY)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    regions = "+".join(config.ELIA_SOURCES[config.TARGET_SOURCE]["region"])
    fig.suptitle(f"{config.TARGET_SOURCE.capitalize()} ({regions}): forecasts issued daily at "
                 f"{config.BACKTEST_ISSUE_TIME} UTC, D+1 part stitched, "
                 f"{start_ts:%d %b} – {end_ts - pd.Timedelta(days=1):%d %b %Y}\nrun {run_name}",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()

    if path is None:
        path = config.PLOT_DIRECTORY / f"backtest_{run_name}_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    logger.info("Forecast plot saved to %s", path)
    if show:
        plt.show()
    return fig


# ------------- Interactive (Plotly) versions, used by the dashboard (app.py) -------------
def _plotly_layout(fig: go.Figure, title: str, y_title: str = "Power [MW]", x_title: str = "Time [UTC]") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(color=TEXT_PRIMARY, size=15)),
        template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="x unified",
        # Vertical legend right of the plot area, so it never overlaps the title or the data
        showlegend=True,
        legend=dict(title=dict(text="Model", font=dict(color=TEXT_SECONDARY)), orientation="v",
                    x=1.01, xanchor="left", y=1, yanchor="top", itemwidth=40,
                    font=dict(color=TEXT_PRIMARY, size=13), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=60, b=60), height=520,
        yaxis=dict(title=y_title, gridcolor=GRID_COLOR, zeroline=False),
        xaxis=dict(title=x_title, showgrid=False, linecolor=AXIS_COLOR),
    )
    return fig


def _add_model_traces(fig: go.Figure, d: pd.DataFrame, model: str, color: str, bands: bool) -> None:
    """q10-q90 band (toggled with its line through the legend group) and q50 line of one model."""
    if bands and d["q10"].notna().all() and d["q90"].notna().all():
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["q90"], mode="lines", line=dict(width=0),
                                 legendgroup=model, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["q10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(color, 0.15), legendgroup=model,
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["q50"], mode="lines", name=model, legendgroup=model,
                             line=dict(color=color, width=2.5),
                             hovertemplate=f"{model}: %{{y:,.0f}} MW<extra></extra>"))


def _add_actual_trace(fig: go.Figure, d: pd.DataFrame) -> None:
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["actual"], mode="lines", name="Actual (Elia)",
                             line=dict(color=TEXT_PRIMARY, width=1.6),
                             hovertemplate="Actual: %{y:,.0f} MW<extra></extra>"))


def plotly_backtest_period(predictions: pd.DataFrame, start: str, end: str, models: list[str],
                           bands: bool = True) -> go.Figure:
    """Interactive plot_backtest: D+1 parts of consecutive forecasts stitched over [start, end), models overlaid."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rows = select_window(predictions, "d+1")
    rows = rows[(rows["timestamp"] >= start_ts) & (rows["timestamp"] < end_ts) & rows["model"].isin(models)]
    fig = go.Figure()
    if rows.empty:
        return _plotly_layout(fig, "No forecasts in the selected period")
    colors = model_colors(models)
    for model in models:
        _add_model_traces(fig, rows[rows["model"] == model].sort_values("timestamp"), model, colors[model], bands)
    _add_actual_trace(fig, rows.drop_duplicates("timestamp").sort_values("timestamp"))
    return _plotly_layout(fig, f"D+1 forecasts stitched, {start_ts:%d %b %Y} – "
                               f"{end_ts - pd.Timedelta(days=1):%d %b %Y} (line q50, band q10–q90)")


def plotly_single_forecast(predictions: pd.DataFrame, issue_time: pd.Timestamp, models: list[str],
                           bands: bool = True) -> go.Figure:
    """Every step of the forecasts issued at `issue_time`; the dotted line marks the start of D+1."""
    rows = predictions[(predictions["issue_time"] == issue_time) & predictions["model"].isin(models)]
    fig = go.Figure()
    if rows.empty:
        return _plotly_layout(fig, f"No forecast issued at {issue_time}")
    colors = model_colors(models)
    for model in models:
        _add_model_traces(fig, rows[rows["model"] == model].sort_values("timestamp"), model, colors[model], bands)
    _add_actual_trace(fig, rows.drop_duplicates("timestamp").sort_values("timestamp"))
    next_day = issue_time.normalize() + pd.Timedelta(days=1)
    fig.add_vline(x=next_day, line=dict(color=AXIS_COLOR, dash="dot"))
    fig.add_annotation(x=next_day, y=1, yref="paper", text="D+1", showarrow=False, xanchor="left",
                       font=dict(color=TEXT_MUTED))
    return _plotly_layout(fig, f"Forecast issued {issue_time:%Y-%m-%d %H:%M} UTC, {rows['lead_step'].max() + 1} steps")


def plotly_daily_scores(daily: pd.DataFrame, metric: str, models: list[str]) -> go.Figure:
    """Daily `metric` ("RMSE" or "Winkler80") of each model over the test period, from metrics.daily_scores."""
    fig = go.Figure()
    colors = model_colors(models)
    for model in models:
        if model not in daily.index.get_level_values("model"):
            continue
        d = daily.loc[model].reset_index()
        if d[metric].isna().all():
            continue
        fig.add_trace(go.Scatter(x=d["issue_time"], y=d[metric], mode="lines", name=model,
                                 line=dict(color=colors[model], width=1.6),
                                 hovertemplate=f"{model}: %{{y:,.0f}} MW<extra></extra>"))
    return _plotly_layout(fig, f"Daily {metric} per forecast (issue day on the x axis)", y_title=f"{metric} [MW]",
                          x_title="Issue day [UTC]")

# ------------- Feature selection plots -------------------
ACTUAL_COLOR = "#2a78d6"
PREDICTED_COLOR = "#eb6834"
def plot_tree_importance(importance: pd.DataFrame, curve: pd.DataFrame, n_selected: int | None, top_n: int,
                         filename: str = f"tree_{config.TARGET_SOURCE}_feature_importance.png", show: bool = False,
                         size_col: str = "n_features", unit: str = "features") -> plt.Figure:
    """Left: permutation importance (mean ± std over folds) of the top_n features, selected ones highlighted.

    Right: CV RMSE of the top-k features against k (curve[size_col]), with the selected k marked.
    n_selected=None draws no selection (all bars in one color, no marker).
    Expects the outputs of tree_feature_importance and cv_rmse_by_subset_size, or of tree_group_importance and
    cv_rmse_by_group_count with size_col="n_groups", unit="groups".
    The figure is saved to PLOT_DIRECTORY / filename.
    """
    shown = importance.head(top_n).iloc[::-1]
    colors = [ACTUAL_COLOR if n_selected is None or rank <= n_selected else GRID_COLOR for rank in shown["rank"]]

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(15, max(5, 0.25 * len(shown))), facecolor=SURFACE,
                                           gridspec_kw={"width_ratios": [3, 2]})
    names = shown.index
    ax_bar.barh(names, shown["permutation_mean"], xerr=shown["permutation_std"], color=colors,
                error_kw={"ecolor": TEXT_SECONDARY, "linewidth": 0.8})
    selected_note = "" if n_selected is None else f", {n_selected} selected"
    ax_bar.set_title(f"Permutation importance (top {len(shown)} {unit}{selected_note})",
                     loc="left", color=TEXT_PRIMARY, fontsize=11)
    ax_bar.set_xlabel("RMSE increase when shuffled (MW)", color=TEXT_SECONDARY)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    handles = [plt.Line2D([], [], color=TEXT_SECONDARY, linewidth=0.8, marker="|", label="± 1 std over CV folds")]
    if n_selected is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=ACTUAL_COLOR, label="Selected"),
                   plt.Rectangle((0, 0), 1, 1, color=GRID_COLOR, label="Not selected")] + handles
    ax_bar.legend(handles=handles, loc="lower right", frameon=False, labelcolor=TEXT_SECONDARY)

    ax_curve.errorbar(curve[size_col], curve["cv_rmse"], yerr=curve["cv_rmse_std"], color=PREDICTED_COLOR,
                      linewidth=2, marker="o", markersize=4, ecolor=GRID_COLOR, capsize=3)
    full_rmse = curve.loc[curve[size_col].idxmax(), "cv_rmse"]
    ax_curve.axhline(full_rmse, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2, label=f"All {unit}: {full_rmse:.2f} MW")
    if n_selected is not None:
        ax_curve.axvline(n_selected, color=ACTUAL_COLOR, linestyle="--", linewidth=1.5, label=f"Selected: {n_selected}")
    ax_curve.set_title(f"CV RMSE of the top-k {unit}", loc="left", color=TEXT_PRIMARY, fontsize=11)
    ax_curve.set_xlabel(f"Number of {unit} (k)", color=TEXT_SECONDARY)
    ax_curve.set_ylabel("CV RMSE (MW)", color=TEXT_SECONDARY)
    ax_curve.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_curve.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY)

    for ax in (ax_bar, ax_curve):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
    fig.tight_layout()

    config.PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PLOT_DIRECTORY / filename, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    return fig