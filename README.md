# GARCH Volatility Forecast
This project models and predicts volatility using a GARCH(2,2) model on synthetic financial data. It includes data generation, model fitting, forecasting, and visualization of results.

## Files
- `garch_volatility.py`: Main script for generating synthetic data, fitting the GARCH model, and creating forecasts.
- `data_vol.png`: Visualization of synthetic data and its volatility.
- `short_term_vol.png`: Visualization of short-term volatility predictions.
- `long_term_vol.png`: Visualization of long-term volatility predictions.
- `rolling_vol.png`: Visualization of rolling volatility predictions.
- `formula.png`: Image of the GARCH(2,2) formula used in the model.

## Libraries Used
- `numpy`
- `random`
- `plotly`
- `arch`

## Timeframe
- **Input**: Synthetic dataset of 1000 timesteps, generated using GARCH(2,2) parameters.
- **Output**: 
  - Short-term forecast: Predicts volatility for the next 100 timesteps (10% of the data).
  - Long-term forecast: Predicts volatility for the next 1000 timesteps.
  - Rolling forecast: Predicts volatility one step ahead over the last 100 timesteps.
