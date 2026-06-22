# SARIMA Weather Forecast Pipeline

This folder implements a protocol-compatible SARIMA baseline for the weather
forecasting comparison.

SARIMA is univariate, so the pipeline fits one seasonal ARIMA model per target
variable and combines the seven forecasts into the required `(24, 7)` output:

```text
temp, rh, dew, pressure, cloud, wx, wy
```

## Protocol mapping

- Data source: `../weather_dataset.csv`
- Preprocessing:
  - parse `time`
  - sort and enforce hourly frequency
  - clip wind speed/gust columns to non-negative values
  - convert wind speed/direction to `wx`, `wy`
  - interpolate incidental gaps, then forward/back fill
- Split:
  - train: 2021-2023
  - validation: 2024
  - test: 2025
- Forecast artifact: 24 hours x 7 variables
- Test evaluation: rolling-origin forecasts across 2025
- Metrics: MAE and RMSE on real-scale values

## Install dependencies

```powershell
pip install -r weather-sarima/requirements.txt
```

## Dry run

Use this first to validate the dataset and splits without fitting SARIMA models:

```powershell
python weather-sarima/train.py --dry-run
```

## Train

```powershell
python weather-sarima/train.py
```

The default model is:

```text
SARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24), trend="c")
```

This uses daily seasonality for hourly data. The pipeline selects the best model
per target variable by validation MAE, then saves the selected fitted models.
You can override or expand the candidate grid:

```powershell
python weather-sarima/train.py --order 2,0,2 --seasonal-order 1,0,1,24
python weather-sarima/train.py --order "1,0,1;2,0,1" --seasonal-order "1,0,1,24;0,0,1,24"
```

## Predict

After training, run prediction from the saved models without fitting again:

```powershell
python weather-sarima/predict.py
```

For a quick prediction smoke test after models have been saved:

```powershell
python weather-sarima/predict.py --max-origins 5 --bootstrap 100 --no-plots
```

The old end-to-end command still trains and predicts in one run:

```powershell
python weather-sarima/run.py
```

## Outputs

Results are written to `weather-sarima/results/`:

- `metrics.csv`: overall, per-horizon, and per-variable MAE/RMSE
- `overall_metrics.csv`
- `per_horizon_metrics.csv`
- `per_variable_metrics.csv`
- `forecasts.csv`: long-form predictions and actuals
- `diagnostics.txt`: ADF, Ljung-Box, Jarque-Bera, AIC/BIC, convergence
- `models/*.pkl`: selected fitted SARIMA model for each target variable
- `models/best_models.csv`: selected order, validation score, and model path
- `models/model_selection.csv`: scores for every candidate tried
- `models/training_metadata.json`: fit scope and model selection settings
- `plots/`: error curves and predicted-vs-actual samples

## Notes

By default, model parameters are fit on 2021-2023 only. The validation year
2024 is appended with `refit=False` before the first 2025 forecast so each
origin conditions on true pre-test history without changing model parameters.

Use `--fit-through-val` if the group decides final statistical baselines should
refit on all pre-test data, 2021-2024. State that choice in the report.
