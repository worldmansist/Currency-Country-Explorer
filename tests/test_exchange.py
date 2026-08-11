"""Tests for exchange-rate and forecasting helpers."""

import pandas as pd
import pytest

from currency_explorer.data import get_exchange_rate
from currency_explorer.forecasting import (
    calculate_metrics,
    create_monday_forecasts,
    forecast_next_monday,
)


def test_equal_currencies_have_rate_one():
    assert get_exchange_rate("usd", "USD") == 1.0


def test_create_monday_forecasts_uses_previous_observation():
    rates_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-07", "2026-08-10"]),
        "rate": [1.10, 1.12],
    })

    _, forecasts = create_monday_forecasts(rates_df)

    assert len(forecasts) == 1
    assert forecasts.iloc[0]["actual_rate"] == pytest.approx(1.12)
    assert forecasts.iloc[0]["naive_prediction"] == pytest.approx(1.10)


def test_calculate_metrics():
    actual = pd.Series([1.0, 2.0])
    predicted = pd.Series([1.0, 1.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["MAE"] == pytest.approx(0.5)
    assert metrics["RMSE"] == pytest.approx(2 ** -0.5)


def test_next_monday_is_after_latest_observation(monkeypatch):
    rates_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-06", "2026-08-07"]),
        "rate": [1.10, 1.11],
    })

    monkeypatch.setattr(
        "currency_explorer.forecasting.sarima_forecast",
        lambda *args: 1.12
    )
    monkeypatch.setattr(
        "currency_explorer.forecasting.holt_forecast",
        lambda *args: 1.13
    )

    result = forecast_next_monday(
        rates_df,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, 5),
        holt_window=2,
        sarima_window=2
    )

    assert result["forecast_date"] == pd.Timestamp("2026-08-10")
    assert result["forecast_date"] > result["last_observation_date"]
