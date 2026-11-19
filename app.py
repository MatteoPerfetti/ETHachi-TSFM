"""Dashboard of the backtest results: pick a saved run, inspect metrics and forecasts, or run a new backtest.

    streamlit run app.py

Saved runs are read from results/ (written by main.py or by the form in the sidebar), nothing is recomputed
to look at them. Defaults come from config.py.
"""
import matplotlib.pyplot as plt
import json
import pandas as pd
import streamlit as st

import config
from src.backtest import BacktestSettings, load_backtest, run_recorded_backtest, saved_runs
from src.dataset import load_dataset
from src.feature_selection import load_selection, run_tree_selection, save_selection, tsfm_covariates
from src.metrics import EVAL_WINDOWS, daily_scores, summary
from src.plotter import plot_tree_importance, plotly_backtest_period, plotly_daily_scores, plotly_single_forecast
from src.validate_config import validate_config
from src.tsfm_runs import available_adapters, training_records

st.set_page_config(page_title="ETHachi-TSFM backtest", page_icon="⚡", layout="wide")
# Charts use theme=None: our colors and legend layout (src/plotter.py) instead of Streamlit's chart theme

RUNNABLE_MODELS = [
    key
    for key, value in config.MODELS_REGISTRY.items()
    if value["type"] in ("tsfm", "traditional", "baseline", "ensemble")
]
WINDOW_LABELS = {"all": "All forecast steps", "d+1": "Only D+1 (next day 00:00–23:45, as Predico)"}
SCORE_LABELS = {"RMSE": "RMSE [MW]", "Winkler80": "Winkler 80% [MW]", "RMSE_vs_persistence": "vs Persistence",
                "days": "days"}


@st.cache_resource(show_spinner="Loading dataset...")
def get_dataset(target_source: str = config.TARGET_SOURCE):
    return load_dataset(target_source)


def feature_selection_section() -> None:
    """Feature selection for the TSFM covariates: settings first in the sidebar, one tab per target source."""
    modes = config.TREE_SELECTION_MODES
    with st.sidebar.expander("Feature selection - Random Forest Tree", expanded=False):
        mode = st.radio("Rank", list(modes), format_func=lambda m: modes[m][2],
                        index=list(modes).index("groups" if config.TREE_GROUP_SELECTION_BOOL else "features"))
        lags = st.checkbox("Target lags as candidates", value=config.TREE_SELECTION_LAGS)
        calendar = st.checkbox("Calendar dummies as candidates", value=config.TREE_SELECTION_CALENDAR)
        top_n = st.number_input("Bars shown", min_value=5, max_value=200, value=config.TREE_SELECTION_TOP_N_PLOT)
        st.caption("Note: Weather covariates are included by default")
    with st.expander("🔎 Feature selection for the TSFM covariates", expanded=False):
        sources = list(config.ELIA_SOURCES)
        for tab, source in zip(st.tabs([s.capitalize() for s in sources]), sources):
            with tab:
                selection_tab(source, mode, lags, calendar, int(top_n))


def selection_tab(source: str, mode: str, lags: bool, calendar: bool, top_n: int) -> None:
    """Run (or re-run) the selection of one target, then show its graph and a table to pick the TSFM covariates."""
    if st.button(f"Run feature selection for {source}", key=f"selection_{source}"):
        dataset = get_dataset(source)
        with st.spinner(f"Ranking the features of {source} on the training period..."):
            save_selection(run_tree_selection(source, dataset.target, dataset.weather, mode, lags, calendar))
    result = load_selection(source, mode, lags, calendar)
    if result is None:
        st.info("No saved selection for these settings yet: run it with the button. RandomForest fits on the "
                f"training period {config.TRAIN_START[:10]} – {config.TRAIN_END[:10]}, takes minutes.")
        return

    size_col, unit, _ = config.TREE_SELECTION_MODES[mode]
    fig = plot_tree_importance(result.importance, result.curve, None, top_n,
                               filename=f"tree_{source}_{mode}_importance.png", size_col=size_col, unit=unit)
    st.pyplot(fig)
    plt.close(fig)

    top = result.importance.head(top_n)
    current = config.TSFM_COVARIATES.get(source, [])
    in_config = [bool(set(tsfm_covariates(result, [name])) & set(current)) for name in top.index]
    table = pd.DataFrame({"select": False, "importance [MW]": top["permutation_mean"].round(3), "in config.py": in_config},
                         index=top.index.rename(unit[:-1]))
    if "statistics" in top:
        table.insert(1, "statistics", top["statistics"].fillna(""))
    edited = st.data_editor(table, disabled=[c for c in table.columns if c != "select"], width="content",
                            key=f"covariate_table_{source}_{mode}_{lags}_{calendar}")
    covariates = tsfm_covariates(result, list(edited.index[edited["select"]]))
    st.code(f'"{source}": {covariates},', language="python")
    st.caption(f"Tick the {unit} to use: their weather columns are added to the list above (lags and calendar dummies "
               "are left out). Paste it into TSFM_COVARIATES in config.py, or run a backtest: the form in the sidebar "
               "takes the ticked covariates of the target source as its default.")
    ticked = st.session_state.setdefault("ticked_covariates", {})
    if covariates:
        ticked[source] = covariates
    else:
        ticked.pop(source, None)


@st.cache_data(show_spinner="Loading backtest results...")
def get_run(run_name: str, modified: float) -> pd.DataFrame:
    """Predictions of a saved run; `modified` (file time) is part of the cache key, so a re-saved run is reloaded."""
    return load_backtest(run_name)[0]


def new_backtest_form(expanded: bool) -> None:
    """Sidebar form: run a backtest with the chosen models, context, horizon, latency and covariates, then save it."""
    with st.sidebar.expander("Run a new backtest", expanded=expanded):
        # outside the form: changing the target reloads the dataset and the covariate options right away
        sources = list(config.ELIA_SOURCES)
        target = st.selectbox("Target source", sources, index=sources.index(config.TARGET_SOURCE))
        dataset = get_dataset(target)
        adapters = available_adapters() if target == "wind" else {}
        def model_label(key: str) -> str:
            if key not in adapters:
                return config.MODELS_REGISTRY[key].get("name", key)
            record = adapters[key]
            _, family, run_id = key.split(":", 2)
            return (f"{family} LoRA — {run_id} (context {record.get('context_steps', '?')}, "
                    f"steps {record.get('steps', '?')}, rank {record.get('rank', '?')})")
        with st.form("new_backtest"):
            models = st.multiselect("Models and saved adapters", [*RUNNABLE_MODELS, *adapters],
                                    default=[m for m in config.BACKTEST_MODELS if m in RUNNABLE_MODELS],
                                    format_func=model_label)
            start_date = st.date_input("First issue date (UTC)", value=pd.Timestamp(config.TEST_START).date())
            end_date = st.date_input("Last issue date (UTC, included)", value=pd.Timestamp(config.TEST_END).date())
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
            st.caption(
                "Ensembles automatically run all traditional base models using the selected feature mode. "
                "Their lookback, top-K, and decay defaults are defined in src/ensembles.py."
            )
            # default: the covariates ticked in the feature selection table, else the ones from config.py
            default = st.session_state.get("ticked_covariates", {}).get(target, config.TSFM_COVARIATES.get(target, []))
            covariates = st.multiselect("Weather covariates (TSFMs) - selection", list(dataset.weather.columns),
                                        default=[c for c in default if c in dataset.weather.columns],
                                        key=f"covariates_{target}")
            submitted = st.form_submit_button("Run backtest", type="primary")
        st.caption("Each backtest is saved separately. Training does not start a backtest automatically. "
                   "Use validation dates to compare settings; reserve the 2025 holdout for the final comparison.")
    if submitted:
        if not models:
            st.sidebar.error("Choose at least one model.")
            return
        if start_date > end_date:
            st.sidebar.error("The first issue date must not be after the last issue date.")
            return
        settings = BacktestSettings(test_start=pd.Timestamp(start_date, tz="UTC").isoformat(),
                                    test_end=(pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).isoformat(),
                                    context_steps=int(context), horizon_steps=int(horizon),
                                    latency_steps=int(latency), covariates=covariates,
                                    traditional_strategy=traditional_strategy, target_source=target)
        with st.spinner(f"Running backtest {settings.run_name} for {', '.join(models)}..."):
            path = run_recorded_backtest(dataset, models, settings)
        st.session_state["run"] = path.stem
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
        if values.get("strategy") and not values.get("ensemble_key")
    ]
    if traditional:
        caption += "  \nTraditional benchmarks: " + ", ".join(traditional)
    ensembles = [
        f"{model}: {values['strategy']}, bases={len(values['base_models'])}, "
        f"parameters={values['parameters']}"
        for model, values in meta.get("model_configs", {}).items()
        if values.get("ensemble_key")
    ]
    if ensembles:
        caption += "  \nEnsembles: " + ", ".join(ensembles)
    return caption


def collect_results(runs: dict[str, dict]) -> tuple[list[dict], set[str]]:
    """Collect scored models from current and earlier standalone evaluations."""
    rows, evaluated = [], set()
    for name, meta in runs.items():
        scores = meta.get("scores_all")
        if scores is None:
            try:
                scores = summary(load_backtest(name)[0], "all").to_dict(orient="index")
            except (OSError, ValueError, KeyError):
                continue
        settings = meta.get("settings", {})
        for model, score in scores.items():
            cfg = meta.get("model_configs", {}).get(model, {})
            if cfg.get("run_id"):
                evaluated.add(f"lora:{cfg.get('model_key')}:{cfg['run_id']}")
            rows.append({"Evaluation": name, "Model": model, "Training run": cfg.get("run_id"),
                         "Target": meta.get("target", settings.get("target_source", "unknown")),
                         "Start UTC": settings.get("test_start"), "End UTC (exclusive)": settings.get("test_end"),
                         "Issue UTC": settings.get("issue_time"), "Context": settings.get("context_steps"),
                         "Horizon": settings.get("horizon_steps"), "Latency": settings.get("latency_steps"),
                         "Covariates": ", ".join(settings.get("covariates", [])) or "none",
                         "RMSE [MW]": score.get("RMSE"), "Winkler80 [MW]": score.get("Winkler80"),
                         "Origins": score.get("days")})
    for path in (config.BACKTEST_RESULTS_DIR / "tsfm_evaluations").glob("*/run.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            settings = meta.get("settings", {})
            run_id, family = settings.get("adapter_run_id"), settings.get("model")
            if run_id and meta.get("status") == "completed":
                evaluated.add(f"lora:{family}:{run_id}")
            for model, score in meta.get("scores", {}).items():
                rows.append({"Evaluation": path.parent.name, "Model": model, "Training run": run_id if "LoRA" in model else None,
                             "Target": "wind", "Start UTC": settings.get("start"),
                             "End UTC (exclusive)": settings.get("end_exclusive"), "Issue UTC": settings.get("issue_time_utc"),
                             "Context": settings.get("context_steps"), "Horizon": settings.get("horizon_steps"),
                             "Latency": settings.get("latency_steps"), "Covariates": ", ".join(settings.get("covariates", [])) or "none",
                             "RMSE [MW]": score.get("RMSE"), "Winkler80 [MW]": score.get("Winkler80"), "Origins": score.get("days")})
        except (OSError, ValueError, TypeError):
            continue
    return rows, evaluated


def results_overview(rows: list[dict]) -> None:
    """Show saved model scores together, with evaluation settings for context."""
    st.subheader("Model results overview")
    if not rows:
        st.info("No saved backtest scores yet. Run a backtest to populate this table.")
        return
    table = pd.DataFrame(rows)
    targets = sorted(table["Target"].dropna().unique())
    target = st.selectbox("Filter results by target", ["All targets", *targets])
    if target != "All targets":
        table = table[table["Target"] == target]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_order=["Model", "Training run", "RMSE [MW]", "Winkler80 [MW]", "Origins", "Target",
                               "Start UTC", "End UTC (exclusive)", "Issue UTC", "Context", "Horizon", "Latency",
                               "Covariates", "Evaluation"],
                 column_config={"RMSE [MW]": st.column_config.NumberColumn(format="%.1f"),
                                "Winkler80 [MW]": st.column_config.NumberColumn(format="%.1f")},
                 alt="Scores for each model and saved evaluation")
    st.caption("RMSE and Winkler80 are averaged over forecast origins; lower is better. Results from different "
               "dates, origins, targets or settings are not directly comparable. For a paired comparison, select "
               "models from the same backtest and verify their inputs.")


def experiment_history(evaluated: set[str]) -> None:
    """List evaluation attempts and the training settings recorded for each adapter."""
    with st.expander("Training and evaluation history", expanded=False):
        adapters = training_records()
        selectable = available_adapters()
        attempts = []
        for path in (config.BACKTEST_RESULTS_DIR / "backtest_attempts").glob("*.json"):
            try:
                attempt = json.loads(path.read_text(encoding="utf-8"))
                if attempt.get("status") != "completed":
                    attempts.append({"Started UTC": attempt.get("started_at"), "Status": attempt.get("status"),
                                     "Models": ", ".join(attempt.get("model_keys", [])), "Error": attempt.get("error")})
            except (OSError, ValueError, TypeError):
                continue
        if attempts:
            st.caption("Failed or interrupted backtest attempts")
            st.dataframe(pd.DataFrame(attempts), hide_index=True, width="stretch",
                         alt="Failed and interrupted backtest attempts")
        training_rows = []
        for key, record in adapters.items():
            _, family, run_id = key.split(":", 2)
            training_rows.append({"Model": family, "Run": run_id, "Training context": record.get("context_steps"),
                                  "Horizon": record.get("horizon_steps"), "Steps": record.get("steps"),
                                  "Batch size": record.get("batch_size"), "Rank": record.get("rank"),
                                  "Learning rate": record.get("learning_rate"),
                                  "Training status": record.get("status", "legacy"),
                                  "Selectable": key in selectable,
                                  "Rolling backtest": "Recorded" if key in evaluated else "Not yet recorded"})
        if training_rows:
            st.caption("Training runs without a recorded shared backtest can be selected in the sidebar.")
            st.dataframe(pd.DataFrame(training_rows), hide_index=True, width="stretch",
                         alt="Training parameters and rolling backtest status")
            selected = st.selectbox("Training run details", list(adapters), format_func=lambda k: k.replace("lora:", ""))
            st.json(adapters[selected])


def metrics_tab(predictions: pd.DataFrame, models: list[str], window: str, meta: dict) -> None:
    table = summary(predictions, window)
    shown = table.rename(columns=SCORE_LABELS)[[SCORE_LABELS[c] for c in SCORE_LABELS if c in table.columns]]
    formats = {"RMSE [MW]": "{:,.1f}", "Winkler 80% [MW]": "{:,.1f}", "vs Persistence": "{:+.0%}", "days": "{:d}"}
    best_result_style = "background-color: #d4edda; color: #0b0b0b;"
    st.dataframe(
        shown.style
        .format({k: v for k, v in formats.items() if k in shown.columns}, na_rep="–")
        .highlight_min(
            subset=["RMSE [MW]", "Winkler 80% [MW]"],
            props=best_result_style,
        ),
        width="stretch",
    )
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
    feature_selection_section()

    runs = saved_runs()
    new_backtest_form(expanded=not runs)
    result_rows, evaluated = collect_results(runs)
    results_overview(result_rows)
    experiment_history(evaluated)
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
    training_configs = {name: details.get("training") for name, details in meta.get("model_configs", {}).items()
                        if details.get("training")}
    if training_configs:
        with st.expander("Training parameters for this backtest"):
            st.json(training_configs)
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
