### Chronos-2 implementation

what was done so far:
- downloaded and used the library to have the model weights
- succesfully implemented zero shot for wind_MW over the first 39 time steps of 2026 (39*15 minutes) with context = target variable in 2024

next steps:
- insert covariates (future values of weather) in the context input
- implementing the rolling window algorithm (for now, we are only predicting a few hours; we want a loop that increases the context and predicts the whole 2025 one iteration at a time)
- implementing a metrics_evaluation script/function that computes our two metrics confronting the prediction with the true value

notes:
- context_df starts in DATA_START and finishes in DATA_END (TRAIN/TEST_START/END are not relevant for this zero shot approach as we don't do train --> test but only give context and get the prediction)
- pred_df starts in DATA_END and finishes in DATA_END + PREDICTION_LENGHT
