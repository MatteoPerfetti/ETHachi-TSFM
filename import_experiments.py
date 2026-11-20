"""Import existing local training and rolling backtest records into MLflow."""
from src.experiment_tracking import import_existing, tracking_uri

if __name__ == '__main__':
    count, errors = import_existing()
    print(f'Processed {count} local records. MLflow database: {tracking_uri()}')
    for error in errors:
        print(f'FAILED: {error}')
    if errors:
        raise SystemExit(1)
