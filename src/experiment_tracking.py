"""Local MLflow tracking for training records and rolling evaluations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import config


def tracking_uri() -> str:
    return 'sqlite:///' + (config.BACKTEST_RESULTS_DIR / 'mlflow.db').resolve().as_posix()


def client_and_experiment():
    from mlflow.tracking import MlflowClient
    config.BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_uri())
    exp = client.get_experiment_by_name('ETHachi forecasting')
    if exp is None:
        artifacts = (config.BACKTEST_RESULTS_DIR / 'mlflow_artifacts').resolve().as_uri()
        exp_id = client.create_experiment('ETHachi forecasting', artifact_location=artifacts)
    else:
        exp_id = exp.experiment_id
    return client, exp_id


def _existing(client, exp_id, source):
    runs = client.search_runs([exp_id], filter_string=f"tags.source_key = '{source}'", max_results=100)
    return [run for run in runs if run.data.tags.get("import_complete") == "true" or run.info.status == "FINISHED"]


def _params(client, run_id, values):
    for key, value in values.items():
        if value is not None:
            client.log_param(run_id, key, str(value)[:500])


def _metrics(client, run_id, values, prefix=''):
    for key, value in (values or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            client.log_metric(run_id, prefix + key, value)


def track_training(run_dir: Path, record: dict) -> None:
    """Import a completed or failed trainer record; never duplicate a source."""
    client, exp_id = client_and_experiment()
    source = 'training:' + record.get('model', run_dir.parent.name) + ':' + run_dir.name
    if _existing(client, exp_id, source):
        return
    run = client.create_run(exp_id, run_name=source, tags={'source_key': source, 'kind': 'training'})
    rid = run.info.run_id
    try:
        _params(client, rid, {k: record.get(k) for k in ('model','run_id','backend','base_checkpoint','base_revision','rank','steps','batch_size','learning_rate','seed','context_steps','horizon_steps','train_start','validation_start','validation_end_exclusive','git_commit','git_dirty','covariates')})
        _params(client, rid, {'parameters': json.dumps(record.get('parameters', {}), sort_keys=True)})
        _metrics(client, rid, record.get('zero_shot'), 'validation_base_')
        _metrics(client, rid, record.get('best'), 'validation_best_')
        for entry in record.get('validation_history', []):
            step = entry.get('step', 0)
            for k, v in entry.items():
                if k != 'step' and isinstance(v, (float, int)) and math.isfinite(v):
                    client.log_metric(rid, 'history_' + k, v, step=int(step))
        client.log_artifact(rid, str(run_dir / 'run.json'))
        artifact = run_dir / record.get('adapter_path', 'finetuned-ckpt' if record.get('backend') == 'peft' else 'adapter.safetensors')
        if artifact.is_dir(): client.log_artifacts(rid, str(artifact), 'adapter')
        elif artifact.is_file(): client.log_artifact(rid, str(artifact), 'adapter')
        client.set_tag(rid, 'import_complete', 'true')
        client.set_terminated(rid, 'FINISHED' if record.get('status', 'completed') == 'completed' else 'FAILED')
    except Exception:
        client.set_terminated(rid, 'FAILED')
        raise


def track_evaluation(path: Path) -> None:
    """Log one MLflow run per model, with scores computed from rolling predictions."""
    from src.metrics import summary
    predictions = pd.read_parquet(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    client, exp_id = client_and_experiment()
    scores = summary(predictions, 'all')
    for model in predictions['model'].unique():
        source = 'evaluation:' + path.stem + ':' + hashlib.sha256(model.encode()).hexdigest()[:12]
        if _existing(client, exp_id, source): continue
        run = client.create_run(exp_id, run_name=model, tags={'source_key': source, 'kind': 'rolling_evaluation', 'model': model, 'evaluation_file': path.name})
        rid = run.info.run_id
        try:
            settings = meta.get('settings', {})
            _params(client, rid, {k: settings.get(k) for k in ('target_source','test_start','test_end','issue_time','context_steps','horizon_steps','latency_steps','traditional_strategy')})
            _params(client, rid, {'covariates': json.dumps(settings.get('covariates', [])), 'dataset_fingerprint': meta.get('dataset_fingerprint'), 'evaluation_window': 'all'})
            cfg = meta.get('model_configs', {}).get(model, {})
            _params(client, rid, {'training_run': cfg.get('run_id'), 'base_revision': cfg.get('base_revision')})
            if cfg.get('run_id') and cfg.get('model_key'):
                train_source = 'training:' + cfg['model_key'] + ':' + cfg['run_id']
                linked = _existing(client, exp_id, train_source)
                if linked:
                    client.set_tag(rid, 'training_mlflow_run_id', linked[0].info.run_id)
            _metrics(client, rid, scores.loc[model].to_dict(), 'rolling_')
            client.log_artifact(rid, str(path.with_suffix('.json')))
            client.log_artifact(rid, str(path), 'predictions')
            client.set_tag(rid, 'import_complete', 'true')
            client.set_terminated(rid, 'FINISHED')
        except Exception:
            client.set_terminated(rid, 'FAILED')
            raise


def import_existing() -> tuple[int, list[str]]:
    """Import existing records idempotently; leave legacy provenance unchanged."""
    count, errors = 0, []
    for p in sorted((config.BACKTEST_RESULTS_DIR / 'finetuning').glob('*/*/run.json')):
        try:
            record = json.loads(p.read_text(encoding='utf-8'))
            if record.get('status') == 'running': continue
            track_training(p.parent, record); count += 1
        except Exception as exc: errors.append(f'{p}: {exc}')
    for p in sorted(config.BACKTEST_RESULTS_DIR.glob('*.parquet')):
        if not p.with_suffix('.json').exists(): continue
        try:
            track_evaluation(p); count += 1
        except Exception as exc: errors.append(f'{p}: {exc}')
    return count, errors


def compatible_evaluations(anchor: str, runs: dict) -> list[str]:
    """Only match runs with identical data fingerprint, settings, origins and actual labels."""
    from src.backtest import load_backtest
    baseline, meta = load_backtest(anchor)
    fingerprint = meta.get('dataset_fingerprint')
    if not fingerprint: return [anchor]
    settings = meta.get('settings', {})
    keys = ('target_source','issue_time','context_steps','horizon_steps','latency_steps','covariates','test_start','test_end','train_start','train_end','traditional_strategy')
    def actuals(df):
        cols = ['issue_time','timestamp','actual']
        d = df[cols].drop_duplicates().sort_values(cols[:2]).reset_index(drop=True)
        if d.duplicated(cols[:2]).any(): raise ValueError('Inconsistent actual values')
        return d
    base_actual = actuals(baseline)
    matched = []
    for name, candidate in runs.items():
        if candidate.get('dataset_fingerprint') != fingerprint: continue
        opts = candidate.get('settings', {})
        if any(opts.get(k) != settings.get(k) for k in keys): continue
        try:
            other, _ = load_backtest(name)
            if (actuals(other).equals(base_actual) and all(
                actuals(group).equals(base_actual) for _, group in other.groupby('model')
            )): matched.append(name)
        except (OSError, ValueError, KeyError): continue
    return matched
