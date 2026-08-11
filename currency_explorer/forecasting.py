"""Forecasting and backtesting functions for exchange-rate data."""

import warnings
from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm


def find_best_sarima(history):
    """Find the SARIMA configuration with the lowest AIC."""
    if len(history) < 10:
        raise ValueError("At least 10 observations are required for SARIMA tuning")

    orders = list(product([0, 1], [0, 1], [0, 1]))
    seasonal_orders = [
        (p, d, q, 5)
        for p, d, q in product([0, 1], [0, 1], [0, 1])
    ]
    parameter_combinations = list(product(orders, seasonal_orders))
    results = []

    for order, seasonal_order in tqdm(
        parameter_combinations,
        desc="SARIMA grid search",
        unit="model"
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = SARIMAX(
                    history,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False, maxiter=100)

            results.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": fitted_model.aic
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("No SARIMA configuration could be fitted")

    return results_df.sort_values("aic").reset_index(drop=True)


def sarima_forecast(history, order, seasonal_order):
    """Fit SARIMA to the supplied history and forecast one observation."""
    if len(history) < 10:
        raise ValueError("At least 10 observations are required for SARIMA")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=100)

    prediction = fitted_model.forecast(steps=1)
    return float(np.asarray(prediction)[0])


def holt_forecast(history):
    """Fit a damped Holt trend and forecast one observation."""
    if len(history) < 3:
        raise ValueError("At least 3 observations are required for Holt")

    model = ExponentialSmoothing(
        history,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    )
    fitted_model = model.fit(optimized=True)
    prediction = fitted_model.forecast(1)

    return float(np.asarray(prediction)[0])


def create_monday_forecasts(rates_df):
    """Create Monday forecast targets and the naive baseline."""
    required_columns = {"date", "rate"}

    if not required_columns.issubset(rates_df.columns):
        raise ValueError("rates_df must contain date and rate columns")

    prepared_df = rates_df[["date", "rate"]].copy()
    prepared_df = prepared_df.sort_values("date").reset_index(drop=True)
    prepared_df["previous_date"] = prepared_df["date"].shift(1)
    prepared_df["previous_rate"] = prepared_df["rate"].shift(1)

    monday_forecasts = prepared_df.loc[
        prepared_df["date"].dt.weekday == 0,
        ["date", "rate", "previous_date", "previous_rate"]
    ].copy()

    monday_forecasts = monday_forecasts.dropna()
    monday_forecasts = monday_forecasts.rename(columns={
        "date": "forecast_date",
        "rate": "actual_rate",
        "previous_rate": "naive_prediction"
    })

    return prepared_df, monday_forecasts


def backtest_monday_forecasts(
    rates_df,
    order,
    seasonal_order,
    tuning_size=252,
    holt_window=20,
    sarima_window=252,
    minimum_sarima_history=60
):
    """Backtest one-step forecasts for Mondays after the tuning period."""
    if tuning_size <= 0:
        raise ValueError("tuning_size must be greater than zero")

    if holt_window < 3:
        raise ValueError("holt_window must contain at least 3 observations")

    prepared_df, monday_forecasts = create_monday_forecasts(rates_df)
    model_results = monday_forecasts.loc[
        monday_forecasts.index >= tuning_size
    ].copy()

    model_results["sarima_prediction"] = np.nan
    model_results["holt_prediction"] = np.nan

    for monday_index in tqdm(
        model_results.index,
        desc="Monday backtest",
        unit="forecast"
    ):
        available_history = prepared_df.loc[
            :monday_index - 1,
            "rate"
        ]
        holt_history = available_history.tail(holt_window)
        sarima_history = available_history.tail(sarima_window)

        if len(holt_history) >= holt_window:
            model_results.loc[
                monday_index,
                "holt_prediction"
            ] = holt_forecast(holt_history)

        if len(sarima_history) >= minimum_sarima_history:
            model_results.loc[
                monday_index,
                "sarima_prediction"
            ] = sarima_forecast(
                sarima_history,
                order,
                seasonal_order
            )

    return model_results


def forecast_next_monday(
    rates_df,
    order,
    seasonal_order,
    holt_window=20,
    sarima_window=252
):
    """Forecast the first Monday strictly after the latest observation."""
    prepared_df = rates_df[["date", "rate"]].copy()
    prepared_df = prepared_df.sort_values("date").reset_index(drop=True)

    if prepared_df.empty:
        raise ValueError("At least one historical observation is required")

    latest_date = pd.Timestamp(prepared_df["date"].iloc[-1]).normalize()
    days_until_monday = int((7 - latest_date.weekday()) % 7)

    if days_until_monday == 0:
        days_until_monday = 7

    forecast_date = latest_date + timedelta(days=days_until_monday)
    history = prepared_df["rate"]
    holt_history = history.tail(holt_window)
    sarima_history = history.tail(sarima_window)

    return {
        "forecast_date": forecast_date,
        "last_observation_date": latest_date,
        "naive_prediction": float(history.iloc[-1]),
        "sarima_prediction": sarima_forecast(
            sarima_history,
            order,
            seasonal_order
        ),
        "holt_prediction": holt_forecast(holt_history),
    }


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE for aligned actual and predicted values."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal lengths")

    if len(actual) == 0:
        raise ValueError("Metrics require at least one observation")

    errors = actual - predicted

    return {
        "MAE": errors.abs().mean(),
        "RMSE": errors.pow(2).mean() ** 0.5
    }
