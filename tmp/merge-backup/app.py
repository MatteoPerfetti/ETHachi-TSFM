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

<<<<<<< HEAD
# Streamlit page configuration
st.set_page_config(
    page_title="ETHachi-TSFM Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading Elia and Open-Meteo datasets...")
def get_cached_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare historical data from the local cache or APIs."""
    validate_config()
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(
                name, client, src["region"], config.DATA_START, config.DATA_END, config.RESOLUTIONCODE
            )
    for name, src in config.OPENMETEO_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            features_data[name] = load_openmeteo(
                name, client, src["latitude"], src["longitude"], src["hourly"],
                config.DATA_START, config.DATA_END
            )

    # Prepare target
    target = targets_data[config.TARGET_SOURCE]
    if config.MULTILE_TARGETS:
        target = target.add_prefix(f"{config.TARGET_SOURCE}_").add_suffix("_MW")
    else:
        target = target.sum(axis=1).to_frame(name=f"{config.TARGET_SOURCE}_MW")

    # Prepare features
    if config.DATA_USE_DWD:
        features = merge_dwd_weather(features_data["weather"], config.DATA_DWD_PATH)
    else:
        for name, df in features_data.items():
            features_data[name] = interpolate_to_15min(df)
        features = merge(features_data)

    features = encode_directions(features)
    lag_features = add_lag_features(target, config.CONTEXT_LENGTH, config.PREDICTION_LENGTH)
    features = features.join(lag_features, how="inner")
    features = add_calendar_features(features, config.TIMEZONE, config.COUNTRY)

    target = target.loc[features.index]
    return features, target


@st.cache_resource(show_spinner="Initializing model...")
def get_cached_model(model_key: str):
    """Load model weights into RAM/GPU memory once."""
    return TSFMRunner.load_model(model_key)


def main():
    st.title("⚡ ETHachi-TSFM: Benchmark Time Series Foundation Models")
    st.markdown(
        "Dashboard for comparing **Chronos-2**, **Timestral-209M**, "
        "**TimesFM-3.0** and the **XGBoost** baseline on wind power generation (Elia Open Data)."
    )

    # ==================== SIDEBAR ====================
    st.sidebar.header("⚙ Benchmark settings")

    st.sidebar.subheader("1. Models to run")
    sel_chronos = st.sidebar.checkbox("Chronos-2 (Amazon)", value=True)
    sel_timestral = st.sidebar.checkbox("Timestral-209M (Mistral)", value=True)
    sel_timesfm = st.sidebar.checkbox("TimesFM-3.0 (Google)", value=True)
    sel_xgboost = st.sidebar.checkbox("XGBoost Baseline", value=False)

    selected_models = []
    if sel_chronos:
        selected_models.append("chronos2")
    if sel_timestral:
        selected_models.append("timestral_209m")
    if sel_timesfm:
        selected_models.append("timesfm3")
    if sel_xgboost:
        selected_models.append("xgboost")

    st.sidebar.subheader("2. Context window")
    context_options = list(config.CONTEXT_WINDOWS.keys())
    default_ctx_idx = context_options.index("14D") if "14D" in context_options else 0
    selected_context = st.sidebar.selectbox(
        "Historical context window:",
        options=context_options,
        index=default_ctx_idx,
        help="Length of historical data supplied to the models to infer patterns and seasonality.",
    )
    context_steps = config.CONTEXT_WINDOWS[selected_context]

    st.sidebar.subheader("3. Forecast start date")
    forecast_date = st.sidebar.date_input(
        "Forecast date (UTC):",
        value=datetime.date(2025, 1, 1),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2025, 12, 28),
    )
    start_ts_str = f"{forecast_date.strftime('%Y-%m-%d')}T00:00:00Z"

    st.sidebar.markdown(f"**Fixed horizon:** 39 hours ({config.HORIZON_STEPS} steps of 15 minutes)")

    cache_path = config.DATA_DIR / "predictions_cache.parquet"
    has_cache = cache_path.exists()
    use_cache = False
    if has_cache:
        use_cache = st.sidebar.checkbox("Use the latest cached predictions", value=True)

    run_btn = st.sidebar.button("🚀 Run benchmark", type="primary", use_container_width=True)

    # Load dataset
    features, target = get_cached_dataset()

    # Check timestamps
    target_idx = target.index
    horizon = config.HORIZON_STEPS

    # Filter context and ground truth
    context_target_all = target.loc[target_idx < start_ts_str]
    y_true_horizon = target.loc[target_idx >= start_ts_str].iloc[:horizon]

    if len(context_target_all) < context_steps:
        st.warning(
            f"Available context ({len(context_target_all)} steps) is shorter than requested ({context_steps} steps). "
            "All available preceding observations will be used."
        )
        context_target = context_target_all
    else:
        context_target = context_target_all.iloc[-context_steps:]

    if len(y_true_horizon) < horizon:
        st.error(f"Insufficient observations from {start_ts_str} to cover the 39-hour horizon ({horizon} steps).")
        return

    y_true_vals = y_true_horizon.values.squeeze()

    # Results containers
    forecasts = {}
    metrics_all = {}

    should_run = run_btn or (use_cache and has_cache)

    if not should_run:
        st.info("👈 Configure the settings in the sidebar and click **'Run benchmark'** to begin.")
        return

    # Use cached predictions when requested unless a fresh run is triggered
    if use_cache and has_cache and not run_btn:
        st.success(f"Loading predictions from cache (`{cache_path.name}`).")
        cache_df = pd.read_parquet(cache_path)
        common_idx = y_true_horizon.index.intersection(cache_df.index)
        if len(common_idx) == horizon:
            y_true_horizon = y_true_horizon.loc[common_idx]
            y_true_vals = y_true_horizon.values.squeeze()
            for col in cache_df.columns:
                if col.endswith("_median"):
                    m_name = col.replace("_median", "")
                    q10 = cache_df[f"{m_name}_q10"].values if f"{m_name}_q10" in cache_df.columns else None
                    q90 = cache_df[f"{m_name}_q90"].values if f"{m_name}_q90" in cache_df.columns else None
                    median = cache_df[col].values
                    forecasts[m_name] = {"median": median, "q10": q10, "q90": q90}
                    metrics_all[m_name] = calculate_metrics(y_true_vals, median, q10, q90)
        else:
            st.warning("The existing cache does not match the selected date. Recomputing...")
            should_run_fresh = True
    else:
        should_run_fresh = True

    if not forecasts:
        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
=======
st.set_page_config(page_title="ETHachi-TSFM backtest", page_icon="⚡", layout="wide")
# Charts use theme=None: our colors and legend layout (src/plotter.py) instead of Streamlit's chart theme

RUNNABLE_MODELS = [k for k, v in config.MODELS_REGISTRY.items() if v["type"] in ("tsfm", "baseline")]
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
            covariates = st.multiselect("Weather covariates (TSFMs)", list(dataset.weather.columns),
                                        default=config.TSFM_COVARIATES[config.TARGET_SOURCE])
            submitted = st.form_submit_button("Run backtest", type="primary")
        st.caption("A full year takes minutes per TSFM on CPU. Re-running a model with the same settings "
                   "replaces only its forecasts in the saved run.")
    if submitted:
        if not models:
            st.sidebar.error("Choose at least one model.")
>>>>>>> origin/Chronos-2-zero-shot
            return
        settings = BacktestSettings(context_steps=int(context), horizon_steps=int(horizon),
                                    latency_steps=int(latency), covariates=covariates)
        with st.spinner(f"Running backtest {settings.run_name} for {', '.join(models)}..."):
            save_backtest(run_backtest(dataset, models, settings))
        st.session_state["run"] = settings.run_name
        st.rerun()

<<<<<<< HEAD
        progress_bar = st.progress(0, text="Starting forecast computation...")
        total_steps = len(selected_models)
        current_step = 0

        # Chronos-2
        if "chronos2" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Running Chronos-2...")
            m = get_cached_model("chronos2")
            res = TSFMRunner.run_chronos(context_target, horizon=horizon, model=m)
            forecasts["Chronos-2"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["Chronos-2"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # Timestral-209M
        if "timestral_209m" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Running Timestral-209M...")
            m = get_cached_model("timestral_209m")
            res = TSFMRunner.run_timestral(context_target, horizon=horizon, model=m)
            forecasts["Timestral-209M"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["Timestral-209M"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # TimesFM-3
        if "timesfm3" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Running TimesFM-3.0...")
            m = get_cached_model("timesfm3")
            res = TSFMRunner.run_timesfm(context_target, horizon=horizon, model=m)
            forecasts["TimesFM-3"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["TimesFM-3"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # XGBoost Baseline
        if "xgboost" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Running XGBoost...")
            try:
                X_train, X_test, y_train, y_test = split_train_test(
                    features, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END
                )
                _, y_pred_xgb_full = run_model("xgboost", X_train, X_test, y_train, y_test)
                y_xgb_39h = np.asarray(y_pred_xgb_full[:horizon]).squeeze()
                forecasts["XGBoost"] = {"median": y_xgb_39h}
                metrics_all["XGBoost"] = calculate_metrics(y_true_vals, y_xgb_39h)
            except Exception as e:
                st.error(f"Error running XGBoost: {e}")
            current_step += 1

        progress_bar.progress(100, text="Computation complete!")

        # Update the Parquet cache
        cache_dict = {"Actual_MW": y_true_vals}
        for name, preds in forecasts.items():
            cache_dict[f"{name}_median"] = preds["median"]
            if preds.get("q10") is not None and preds.get("q90") is not None:
                cache_dict[f"{name}_q10"] = preds["q10"]
                cache_dict[f"{name}_q90"] = preds["q90"]
        cache_df = pd.DataFrame(cache_dict, index=y_true_horizon.index)
        cache_df.to_parquet(cache_path)

    # ==================== METRIC SCORECARDS ====================
    st.markdown("### 📊 Results and metric scorecards")
    cols = st.columns(len(forecasts))
    for col, (model_name, m_dict) in zip(cols, metrics_all.items()):
        with col:
            st.subheader(model_name)
            st.metric("RMSE", f"{m_dict['RMSE']:.2f} MW")
            st.metric("MAE", f"{m_dict['MAE']:.2f} MW")
            st.metric("WAPE", f"{m_dict['WAPE']:.2f}%")
            if "Coverage_80CI" in m_dict:
                st.metric("Coverage (80% CI)", f"{m_dict['Coverage_80CI']:.1f}%")

    # ==================== PLOTLY CHART ====================
    st.markdown("### 📈 Interactive forecast chart")
    st.caption("Click legend entries to show or hide forecast curves and prediction intervals.")
    fig = create_plotly_comparison(y_true_horizon, forecasts, metrics=metrics_all)
    st.plotly_chart(fig, use_container_width=True)

    # ==================== DATA TABLE AND EXPORT ====================
    with st.expander("📋 Detailed data table and CSV export"):
        summary_table = metrics_summary_table(metrics_all)
        st.dataframe(summary_table.style.highlight_min(axis=0, color="#d4edda", subset=["RMSE", "MAE", "WAPE"]), use_container_width=True)

        # Predictions at each time step
        table_dict = {"Actual (MW)": y_true_vals}
        for name, preds in forecasts.items():
            table_dict[f"{name} Median"] = preds["median"]
            if preds.get("q10") is not None:
                table_dict[f"{name} Q10"] = preds["q10"]
                table_dict[f"{name} Q90"] = preds["q90"]
=======

def run_caption(meta: dict) -> str:
    s = meta["settings"]
    covariates = ", ".join(s["covariates"]) if s["covariates"] else "none"
    return (f"Target **{meta['target']}** · issue {s['issue_time']} UTC · horizon {s['horizon_steps']} steps · "
            f"context {s['context_steps']} steps · latency {s['latency_steps']} steps  \n"
            f"Test {s['test_start'][:10]} – {s['test_end'][:10]} · skipped days: {len(meta['skipped'])}  \n"
            f"Covariates: {covariates}")


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

>>>>>>> origin/Chronos-2-zero-shot

def main() -> None:
    validate_config()
    st.title("⚡ ETHachi-TSFM backtest")

<<<<<<< HEAD
        csv_data = df_preds.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download predictions as CSV",
            data=csv_data,
            file_name=f"benchmark_tsfm_{forecast_date}_{selected_context}.csv",
            mime="text/csv",
        )
=======
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
>>>>>>> origin/Chronos-2-zero-shot


if __name__ == "__main__":
    main()
