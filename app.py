"""Dashboard of the backtest results: pick a saved run, inspect metrics and forecasts, or run a new backtest.

    streamlit run app.py

Saved runs are read from results/ (written by main.py or by the form in the sidebar), nothing is recomputed
to look at them. Defaults come from config.py.
"""
import pandas as pd
import streamlit as st

import config
from src.backtest import BacktestSettings, load_backtest, run_backtest, save_backtest, saved_runs
from src.dataset import load_dataset
from src.metrics import EVAL_WINDOWS, daily_scores, summary
from src.plotter import plotly_backtest_period, plotly_daily_scores, plotly_single_forecast
from src.validate_config import validate_config

st.set_page_config(page_title="ETHachi-TSFM backtest", page_icon="⚡", layout="wide")
# Charts use theme=None: our colors and legend layout (src/plotter.py) instead of Streamlit's chart theme

RUNNABLE_MODELS = [
    key
    for key, value in config.MODELS_REGISTRY.items()
    if value["type"] in ("tsfm", "traditional", "baseline")
]
WINDOW_LABELS = {"all": "All forecast steps", "d+1": "Only D+1 (next day 00:00–23:45, as Predico)"}
SCORE_LABELS = {"RMSE": "RMSE [MW]", "Winkler80": "Winkler 80% [MW]", "RMSE_vs_persistence": "vs Persistence",
                "days": "days"}


@st.cache_resource(show_spinner="Loading dataset...")
def get_dataset():
    return load_dataset()


@st.cache_data(show_spinner="Loading backtest results...")
def get_run(run_name: str, modified: float) -> pd.DataFrame:
    """Predictions of a saved run; `modified` (file time) is part of the cache key, so a re-saved run is reloaded."""
    return load_backtest(run_name)[0]


def new_backtest_form(expanded: bool) -> None:
    """Sidebar form: run a backtest with the chosen models, context, horizon, latency and covariates, then save it."""
    with st.sidebar.expander("Run a new backtest", expanded=expanded):
        dataset = get_dataset()
        with st.form("new_backtest"):
            models = st.multiselect("Models", RUNNABLE_MODELS,
                                    default=[m for m in config.BACKTEST_MODELS if m in RUNNABLE_MODELS])
            context = st.number_input("Context [15 min steps]", min_value=1, max_value=30000,
                                      value=config.BACKTEST_CONTEXT_STEPS, step=96)
            horizon = st.number_input("Horizon [steps]", min_value=1, max_value=1000,
                                      value=config.BACKTEST_HORIZON_STEPS)
            latency = st.number_input("Data latency [steps]", min_value=0, max_value=96,
                                      value=config.BACKTEST_LATENCY_STEPS)
            traditional_strategy = st.selectbox(
                "Traditional ML feature mode",
                config.TRADITIONAL_STRATEGIES,
                index=config.TRADITIONAL_STRATEGIES.index(config.TRADITIONAL_STRATEGY),
                help=(
                    "origin_aligned repeats the latest target context across the horizon; "
                    "fixed_gap uses horizon-safe timestamp-wise lags. Optuna is not run here."
                ),
            )
            covariates = st.multiselect("Weather covariates", list(dataset.weather.columns),
                                        default=config.TSFM_COVARIATES[config.TARGET_SOURCE])
            submitted = st.form_submit_button("Run backtest", type="primary")
        st.caption("A full year takes minutes per TSFM on CPU. Re-running a model with the same settings "
                   "replaces only its forecasts in the saved run.")
    if submitted:
        if not models:
            st.sidebar.error("Choose at least one model.")
            return
        settings = BacktestSettings(context_steps=int(context), horizon_steps=int(horizon),
                                    latency_steps=int(latency), covariates=covariates,
                                    traditional_strategy=traditional_strategy)
        with st.spinner(f"Running backtest {settings.run_name} for {', '.join(models)}..."):
            save_backtest(run_backtest(dataset, models, settings))
        st.session_state["run"] = settings.run_name
        st.rerun()


def run_caption(meta: dict) -> str:
    s = meta["settings"]
    covariates = ", ".join(s["covariates"]) if s["covariates"] else "none"
    caption = (f"Target **{meta['target']}** · issue {s['issue_time']} UTC · horizon {s['horizon_steps']} steps · "
            f"context {s['context_steps']} steps · latency {s['latency_steps']} steps  \n"
            f"Test {s['test_start'][:10]} – {s['test_end'][:10]} · skipped days: {len(meta['skipped'])}  \n"
            f"Covariates: {covariates}")
    traditional = [
        f"{model}: {values['strategy']}"
        for model, values in meta.get("model_configs", {}).items()
        if values.get("strategy")
    ]
    if traditional:
        caption += "  \nTraditional benchmarks: " + ", ".join(traditional)
    return caption


def metrics_tab(predictions: pd.DataFrame, models: list[str], window: str, meta: dict) -> None:
    table = summary(predictions, window)
    shown = table.rename(columns=SCORE_LABELS)[[SCORE_LABELS[c] for c in SCORE_LABELS if c in table.columns]]
    formats = {"RMSE [MW]": "{:,.1f}", "Winkler 80% [MW]": "{:,.1f}", "vs Persistence": "{:+.0%}", "days": "{:d}"}
    st.dataframe(shown.style.format({k: v for k, v in formats.items() if k in shown.columns}, na_rep="–")
                 .highlight_min(subset=["RMSE [MW]", "Winkler 80% [MW]"], color="#d4edda"), width="stretch")
    st.caption("Scores of each forecast, averaged over the days (Predico); lower is better, best in green. "
               "Winkler 80%: width of the q10–q90 interval plus 10× the distance of actual values outside it. "
               "vs Persistence: RMSE relative to persistence (shown if persistence is selected).")

    metric = st.radio("Daily score", ["RMSE", "Winkler80"], horizontal=True,
                      format_func=lambda m: SCORE_LABELS[m])
    daily = daily_scores(predictions, window)
    st.plotly_chart(plotly_daily_scores(daily, metric, models), width="stretch", theme=None)
    st.download_button("Download daily scores (CSV)", daily.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name=f"daily_scores_{window}.csv", mime="text/csv")

    timings = pd.DataFrame(meta["timings"]).T.loc[lambda t: t.index.isin(models)]
    if len(timings):
        st.markdown("**Timings** (seconds, whole test period)")
        st.dataframe(timings.rename(columns={"load_s": "load", "fit_s": "fit", "predict_s": "predict"})
                     .style.format("{:,.1f}"), width="content")


def period_tab(predictions: pd.DataFrame, models: list[str], bands: bool) -> None:
    first = predictions["timestamp"].min().date()
    last = predictions["timestamp"].max().date()
    default_start = min(max(pd.Timestamp(config.BACKTEST_PLOT_START).date(), first), last)
    default_end = min(max(pd.Timestamp(config.BACKTEST_PLOT_END).date(), default_start), last)
    left, right = st.columns(2)
    start = left.date_input("From", value=default_start, min_value=first, max_value=last)
    end = right.date_input("To (excluded)", value=default_end, min_value=first, max_value=last)
    if start >= end:
        st.warning("'From' must be before 'To'.")
        return
    st.plotly_chart(plotly_backtest_period(predictions, str(start), str(end), models, bands), width="stretch",
                    theme=None)
    st.caption("For every daily forecast only its D+1 part is shown, so consecutive forecasts tile the timeline; "
               "a jump at midnight is the switch to the next forecast. Click the legend to hide a model.")


def single_tab(predictions: pd.DataFrame, models: list[str], window: str, bands: bool) -> None:
    issues = sorted(predictions["issue_time"].unique())
    issue = st.select_slider("Issue time", options=issues, value=issues[0],
                             format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"))
    issue = pd.Timestamp(issue)
    st.plotly_chart(plotly_single_forecast(predictions, issue, models, bands), width="stretch", theme=None)
    scores = daily_scores(predictions[predictions["issue_time"] == issue], window).droplevel("issue_time")
    st.dataframe(scores.rename(columns=SCORE_LABELS).style.format("{:,.1f}", na_rep="–"), width="content")
    st.caption(f"Scores of this forecast on the evaluation window ({WINDOW_LABELS[window]}).")


def main() -> None:
    validate_config()
    st.title("⚡ ETHachi-TSFM backtest")

    runs = saved_runs()
    new_backtest_form(expanded=not runs)
    if not runs:
        st.info("No saved backtest in results/ yet: run one with the form in the sidebar or with `python main.py`.")
        return

    names = list(runs)
    current = st.session_state.get("run")
    run_name = st.sidebar.selectbox("Saved run", names, index=names.index(current) if current in names else 0)
    st.session_state["run"] = run_name
    meta = runs[run_name]
    predictions = get_run(run_name, (config.BACKTEST_RESULTS_DIR / f"{run_name}.parquet").stat().st_mtime)

    available = sorted(predictions["model"].unique())
    models = st.sidebar.multiselect("Models", available, default=available)
    window = st.sidebar.radio("Evaluation window", EVAL_WINDOWS, format_func=WINDOW_LABELS.get,
                              index=EVAL_WINDOWS.index(config.BACKTEST_EVAL_WINDOW))
    bands = st.sidebar.checkbox("Show q10–q90 bands", value=True)
    st.markdown(run_caption(meta))
    if not models:
        st.warning("Choose at least one model in the sidebar.")
        return

    selected = predictions[predictions["model"].isin(models)]
    tab_metrics, tab_period, tab_single = st.tabs(["Metrics", "Forecasts over a period", "Single forecast"])
    with tab_metrics:
        metrics_tab(selected, models, window, meta)
    with tab_period:
        period_tab(selected, models, bands)
    with tab_single:
        single_tab(selected, models, window, bands)


if __name__ == "__main__":
    main()
