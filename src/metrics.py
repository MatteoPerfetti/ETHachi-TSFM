import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series | None = None,
    q90: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calcola le metriche puntuali (RMSE, MAE, WAPE) e di copertura probabilistica (80% CI).

    Parametri:
        y_true: valori reali osservati (MW)
        y_pred: valori predetti dal modello (mediana / punto) (MW)
        q10: quantile al 10% (limite inferiore intervallo di confidenza 80%) [opzionale]
        q90: quantile al 90% (limite superiore intervallo di confidenza 80%) [opzionale]

    Ritorna:
        Dizionario con metriche calcolate:
        - "RMSE": Root Mean Squared Error (in MW)
        - "MAE": Mean Absolute Error (in MW)
        - "WAPE": Weighted Absolute Percentage Error (in %)
        - "Coverage_80CI": PICP - Prediction Interval Coverage Probability (in %, se q10 e q90 presenti)
    """
    y_t = np.asarray(y_true, dtype=np.float64).squeeze()
    y_p = np.asarray(y_pred, dtype=np.float64).squeeze()

    rmse = float(root_mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))

    # WAPE: somma errori assoluti diviso somma totale energia reale (%)
    total_energy = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(np.abs(y_t - y_p)) / total_energy * 100.0) if total_energy > 0 else 0.0

    results = {
        "RMSE": rmse,
        "MAE": mae,
        "WAPE": wape,
    }

    # Prediction Interval Coverage Probability (PICP) per l'intervallo 80% (q10 -> q90)
    if q10 is not None and q90 is not None:
        q_low = np.asarray(q10, dtype=np.float64).squeeze()
        q_high = np.asarray(q90, dtype=np.float64).squeeze()
        inside = (y_t >= q_low) & (y_t <= q_high)
        results["Coverage_80CI"] = float(np.mean(inside) * 100.0)

    return results


def metrics_summary_table(all_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Converte un dizionario {nome_modello: {metrica: valore}} in un DataFrame riepilogativo formattato."""
    df = pd.DataFrame.from_dict(all_metrics, orient="index")
    # Ordina colonne se presenti
    cols_order = [c for c in ["RMSE", "MAE", "WAPE", "Coverage_80CI"] if c in df.columns]
    return df[cols_order]

