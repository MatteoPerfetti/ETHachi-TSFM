import json
import pandas as pd
from src import experiment_tracking as tracking


def test_rolling_evaluations_log_scores_and_are_idempotent(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'BACKTEST_RESULTS_DIR', tmp_path)
    times = pd.to_datetime(['2025-01-01T09:00:00Z', '2025-01-01T09:15:00Z'])
    df = pd.DataFrame({'model':['XGBoost','XGBoost'], 'issue_time':[times[0]] * 2,
                       'timestamp':times, 'actual':[10., 14.], 'q10':[8., 8.],
                       'q50':[8., 12.], 'q90':[15., 15.]})
    path = tmp_path / 'evaluation.parquet'
    df.to_parquet(path)
    path.with_suffix('.json').write_text(json.dumps({'settings': {'covariates': []}, 'dataset_fingerprint':'data'}))
    tracking.track_evaluation(path)
    tracking.track_evaluation(path)
    client, experiment = tracking.client_and_experiment()
    records = client.search_runs([experiment])
    assert len(records) == 1
    assert records[0].data.metrics['rolling_RMSE'] == 2.0
    assert records[0].data.metrics['rolling_days'] == 1


def test_compatibility_checks_actuals_settings_and_complete_model_origins(monkeypatch):
    from src import backtest
    issue = pd.Timestamp('2025-01-01T09:00:00Z')
    frame = pd.DataFrame({'issue_time':[issue, issue], 'timestamp':[issue, issue + pd.Timedelta(minutes=15)],
                          'actual':[1., 2.], 'model':['base','base']})
    meta = {'dataset_fingerprint':'same', 'settings':{'covariates':['wind']}}
    changed = frame.copy(); changed.loc[0, 'actual'] = 4.
    incomplete = pd.concat([frame, frame.iloc[[0]].assign(model='incomplete')])
    items = {'anchor': (frame, meta), 'match': (frame.copy(), meta), 'changed': (changed, meta),
             'incomplete': (incomplete, meta), 'different': (frame, {'dataset_fingerprint':'other', 'settings':meta['settings']})}
    monkeypatch.setattr(backtest, 'load_backtest', lambda name: items[name])
    runs = {name: data[1] for name, data in items.items()}
    assert tracking.compatible_evaluations('anchor', runs) == ['anchor', 'match']
