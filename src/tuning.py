import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.models import cross_validation_splits
"""Hyperparameter tuning for XGBoost using cross-validation"""

def tune_xgboost(X_train, y_train):
    results = []

    for cross_validation_split in cross_validation_splits(X_train, n_splits=5, test_size=None, gap="0min", max_train_size=None):
        X_train_fold, X_validation_fold = cross_validation_split

        for max_depth in [3, 5, 7]:
            for n_estimators in [50, 100, 200]:
                for learning_rate in [0.01, 0.1, 0.2]:
                    model = xgb.XGBRegressor(objective='reg:squarederror',
                                            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                    
                    model.fit(X_train_fold, y_train.loc[X_train_fold.index])   # retrieve the corresponding y_train values for the training fold
                    y_pred = model.predict(X_validation_fold)  # retrieve the corresponding y_train values for the validation fold
                    
                    rmse = np.sqrt(mean_squared_error(y_train.loc[X_validation_fold.index], y_pred))
                    # r2 = r2_score(y_train.loc[X_validation_fold.index], y_pred)

                    results.append({'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate, 'rmse': rmse}) # add r2 if you wish

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

    best_params, fold_rmses = min(
    rmse_by_params.items(),  # create a list of tuples (params, rmses)
    key=lambda item: np.mean(item[1]),  # compute the average rmse for each tuple
)     

    print("Best parameters:", best_params)
    print("Average validation RMSE:", np.mean(fold_rmses))

    return best_params, fold_rmses

