import xgboost as xgb
import config
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.models import cross_validation_splits
from config import MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, OPTUNA_N_TRIALS
"""Hyperparameter tuning for XGBoost using cross-validation"""

max_depths = MAX_DEPTH
n_estimators = N_ESTIMATORS
learning_rates = LEARNING_RATE
n_trials = OPTUNA_N_TRIALS

def tune_xgboost(X_train, y_train, max_depths, n_estimators, learning_rates):
    results = []

    for cross_validation_split in cross_validation_splits(X_train, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP, config.CV_MAX_TRAIN_SIZE):
        X_train_fold, X_validation_fold = cross_validation_split

        for max_depth in max_depths:
            for n_estimator in n_estimators:
                for learning_rate in learning_rates:
                    model = xgb.XGBRegressor(objective='reg:squarederror',
                                            n_estimators=n_estimator, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                    
                    model.fit(X_train_fold, y_train.loc[X_train_fold.index])   # retrieve the corresponding y_train values for the training fold
                    y_pred = model.predict(X_validation_fold)  # retrieve the corresponding y_train values for the validation fold
                    
                    rmse = np.sqrt(mean_squared_error(y_train.loc[X_validation_fold.index], y_pred))
                    # r2 = r2_score(y_train.loc[X_validation_fold.index], y_pred)

                    results.append({'max_depth': max_depth, 'n_estimators': n_estimator, 'learning_rate': learning_rate, 'rmse': rmse}) # add r2 if you wish

    rmse_by_params = {}

    for result in results:
        params = (
            result["max_depth"],
            result["n_estimators"],
            result["learning_rate"],
        )

        if params not in rmse_by_params:
            rmse_by_params[params] = []   # create the list if it doesn't exist yet

        rmse_by_params[params].append(result["rmse"])

    best_values, fold_rmses = min(
    rmse_by_params.items(),  # create a list of tuples (params, rmses)
    key=lambda item: np.mean(item[1]),  # compute the average rmse for each tuple
    )   

    best_params = {
    "max_depth": best_values[0],
    "n_estimators": best_values[1],
    "learning_rate": best_values[2],
    }

    print("Best parameters:", best_params)
    print("Average validation RMSE:", np.mean(fold_rmses))

    return best_params, fold_rmses

def tune_optuna(X_train, y_train, n_trials):
    folds = list(cross_validation_splits(X_train, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP, config.CV_MAX_TRAIN_SIZE))

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int("max_depth", 1, 9, step=1),
            "n_estimators": trial.suggest_int("n_estimators", 50, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.9, step=0.1),
        }

        fold_rmses = []

        for X_train_fold, X_validation_fold in folds:
            y_train_fold = y_train.loc[X_train_fold.index]
            y_val_fold = y_train.loc[X_validation_fold.index]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict(X_validation_fold)
            fold_rmses.append(np.sqrt(mean_squared_error(y_val_fold, predictions)))

        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def evaluate_baseline_cv(X_train, y_train):
    scores = []
    folds = cross_validation_splits(
        X_train,
        config.CV_N_SPLITS,
        config.CV_TEST_SIZE,
        config.CV_GAP,
        config.CV_MAX_TRAIN_SIZE,
    )

    for X_fold_train, X_fold_val in folds:
        y_fold_train = y_train.loc[X_fold_train.index]
        y_fold_val = y_train.loc[X_fold_val.index]

        model = xgb.XGBRegressor()  # same default model as the untuned baseline
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)
        scores.append(np.sqrt(mean_squared_error(y_fold_val, predictions)))

    return float(np.mean(scores)), scores