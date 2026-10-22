import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config
from logger import logger
from src.datasource import load_elia, load_openmeteo
from src.features import (
    add_calendar_features,
    add_lag_features,
    encode_directions,
    interpolate_to_15min,
    merge,
    merge_dwd_weather,
)
from src.http_client import HttpClient
from src.metrics import calculate_metrics, metrics_summary_table
from src.models import run_model, split_train_test
from src.models_tsfm import TSFMRunner
from src.plotter import create_plotly_comparison
from src.validate_config import validate_config

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="ETHachi-TSFM Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Caricamento dataset Elia e Open-Meteo...")
def get_cached_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica e prepara i dati storici (cache locale o API)."""
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

    # Preparazione target
    target = targets_data[config.TARGET_SOURCE]
    if config.MULTILE_TARGETS:
        target = target.add_prefix(f"{config.TARGET_SOURCE}_").add_suffix("_MW")
    else:
        target = target.sum(axis=1).to_frame(name=f"{config.TARGET_SOURCE}_MW")

    # Preparazione feature
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


@st.cache_resource(show_spinner="Inizializzazione modello...")
def get_cached_model(model_key: str):
    """Carica i pesi del modello una sola volta in memoria RAM/GPU."""
    return TSFMRunner.load_model(model_key)


def main():
    st.title("⚡ ETHachi-TSFM: Benchmark Time Series Foundation Models")
    st.markdown(
        "Dashboard per la valutazione comparativa di **Chronos-2**, **Timestral-209M**, "
        "**TimesFM-3.0** e baseline **XGBoost** sulla produzione eolica (Elia Open Data)."
    )

    # ==================== SIDEBAR ====================
    st.sidebar.header("⚙ Parametri Benchmark")

    st.sidebar.subheader("1. Modelli da Eseguire")
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

    st.sidebar.subheader("2. Finestra di Contesto")
    context_options = list(config.CONTEXT_WINDOWS.keys())
    default_ctx_idx = context_options.index("14D") if "14D" in context_options else 0
    selected_context = st.sidebar.selectbox(
        "Profondità Storica (Context Window):",
        options=context_options,
        index=default_ctx_idx,
        help="Durata dell'orizzonte passato passato ai modelli per inferire pattern e stagionalità.",
    )
    context_steps = config.CONTEXT_WINDOWS[selected_context]

    st.sidebar.subheader("3. Data Inizio Previsione")
    forecast_date = st.sidebar.date_input(
        "Data Forecast (UTC):",
        value=datetime.date(2025, 1, 1),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2025, 12, 28),
    )
    start_ts_str = f"{forecast_date.strftime('%Y-%m-%d')}T00:00:00Z"

    st.sidebar.markdown(f"**Orizzonte Fisso:** 39 ore ({config.HORIZON_STEPS} step da 15min)")

    cache_path = config.DATA_DIR / "predictions_cache.parquet"
    has_cache = cache_path.exists()
    use_cache = False
    if has_cache:
        use_cache = st.sidebar.checkbox("Usa ultima cache calcolata", value=True)

    run_btn = st.sidebar.button("🚀 Esegui Benchmark", type="primary", use_container_width=True)

    # Carica dataset
    features, target = get_cached_dataset()

    # Verifica timestamp
    target_idx = target.index
    horizon = config.HORIZON_STEPS

    # Filtra context e ground truth
    context_target_all = target.loc[target_idx < start_ts_str]
    y_true_horizon = target.loc[target_idx >= start_ts_str].iloc[:horizon]

    if len(context_target_all) < context_steps:
        st.warning(
            f"Dati di contesto disponibili ({len(context_target_all)}) inferiori agli step richiesti ({context_steps}). "
            "Verranno usati tutti i dati antecedenti disponibili."
        )
        context_target = context_target_all
    else:
        context_target = context_target_all.iloc[-context_steps:]

    if len(y_true_horizon) < horizon:
        st.error(f"Dati reali insufficienti a partire da {start_ts_str} per coprire l'orizzonte di 39 ore ({horizon} step).")
        return

    y_true_vals = y_true_horizon.values.squeeze()

    # Contenitore risultati
    forecasts = {}
    metrics_all = {}

    should_run = run_btn or (use_cache and has_cache)

    if not should_run:
        st.info("👈 Configura i parametri nella barra laterale e premi **'Esegui Benchmark'** per iniziare.")
        return

    # Esecuzione da cache se richiesta e non premuto il tasto di ricalcolo forzato
    if use_cache and has_cache and not run_btn:
        st.success(f"Caricamento previsioni dalla cache (`{cache_path.name}`).")
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
            st.warning("La cache esistente non corrisponde alla data selezionata. Ricalcolo in corso...")
            should_run_fresh = True
    else:
        should_run_fresh = True

    if not forecasts:
        if not selected_models:
            st.warning("Seleziona almeno un modello nella sidebar.")
            return

        progress_bar = st.progress(0, text="Inizio calcolo previsioni...")
        total_steps = len(selected_models)
        current_step = 0

        # Chronos-2
        if "chronos2" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Esecuzione Chronos-2...")
            m = get_cached_model("chronos2")
            res = TSFMRunner.run_chronos(context_target, horizon=horizon, model=m)
            forecasts["Chronos-2"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["Chronos-2"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # Timestral-209M
        if "timestral_209m" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Esecuzione Timestral-209M...")
            m = get_cached_model("timestral_209m")
            res = TSFMRunner.run_timestral(context_target, horizon=horizon, model=m)
            forecasts["Timestral-209M"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["Timestral-209M"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # TimesFM-3
        if "timesfm3" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Esecuzione TimesFM-3.0...")
            m = get_cached_model("timesfm3")
            res = TSFMRunner.run_timesfm(context_target, horizon=horizon, model=m)
            forecasts["TimesFM-3"] = {"median": res.median, "q10": res.q10, "q90": res.q90}
            metrics_all["TimesFM-3"] = calculate_metrics(y_true_vals, res.median, res.q10, res.q90)
            current_step += 1

        # XGBoost Baseline
        if "xgboost" in selected_models:
            progress_bar.progress(int((current_step / total_steps) * 100), text="Esecuzione XGBoost...")
            try:
                X_train, X_test, y_train, y_test = split_train_test(
                    features, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END
                )
                _, y_pred_xgb_full = run_model("xgboost", X_train, X_test, y_train, y_test)
                y_xgb_39h = np.asarray(y_pred_xgb_full[:horizon]).squeeze()
                forecasts["XGBoost"] = {"median": y_xgb_39h}
                metrics_all["XGBoost"] = calculate_metrics(y_true_vals, y_xgb_39h)
            except Exception as e:
                st.error(f"Errore durante l'esecuzione di XGBoost: {e}")
            current_step += 1

        progress_bar.progress(100, text="Calcolo completato!")

        # Aggiorna la cache parquet
        cache_dict = {"Actual_MW": y_true_vals}
        for name, preds in forecasts.items():
            cache_dict[f"{name}_median"] = preds["median"]
            if preds.get("q10") is not None and preds.get("q90") is not None:
                cache_dict[f"{name}_q10"] = preds["q10"]
                cache_dict[f"{name}_q90"] = preds["q90"]
        cache_df = pd.DataFrame(cache_dict, index=y_true_horizon.index)
        cache_df.to_parquet(cache_path)

    # ==================== SEZIONE METRICHE (SCORECARDS) ====================
    st.markdown("### 📊 Risultati & Scorecard Metriche")
    cols = st.columns(len(forecasts))
    for col, (model_name, m_dict) in zip(cols, metrics_all.items()):
        with col:
            st.subheader(model_name)
            st.metric("RMSE", f"{m_dict['RMSE']:.2f} MW")
            st.metric("MAE", f"{m_dict['MAE']:.2f} MW")
            st.metric("WAPE", f"{m_dict['WAPE']:.2f}%")
            if "Coverage_80CI" in m_dict:
                st.metric("Coverage (80% CI)", f"{m_dict['Coverage_80CI']:.1f}%")

    # ==================== GRAFICO PLOTLY ====================
    st.markdown("### 📈 Visualizzazione Grafica Interattiva")
    st.caption("Fai clic sulle voci della legenda per attivare/disattivare curve e bande di confidenza.")
    fig = create_plotly_comparison(y_true_horizon, forecasts, metrics=metrics_all)
    st.plotly_chart(fig, use_container_width=True)

    # ==================== TABELLA DATI ED EXPORT ====================
    with st.expander("📋 Tabella Dati Dettagliata & Export CSV"):
        summary_table = metrics_summary_table(metrics_all)
        st.dataframe(summary_table.style.highlight_min(axis=0, color="#d4edda", subset=["RMSE", "MAE", "WAPE"]), use_container_width=True)

        # Tabella previsioni punto per punto
        table_dict = {"Actual (MW)": y_true_vals}
        for name, preds in forecasts.items():
            table_dict[f"{name} Median"] = preds["median"]
            if preds.get("q10") is not None:
                table_dict[f"{name} Q10"] = preds["q10"]
                table_dict[f"{name} Q90"] = preds["q90"]

        df_preds = pd.DataFrame(table_dict, index=y_true_horizon.index)
        st.dataframe(df_preds, use_container_width=True)

        csv_data = df_preds.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Scarica Previsioni in CSV",
            data=csv_data,
            file_name=f"benchmark_tsfm_{forecast_date}_{selected_context}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

