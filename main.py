"""Run the complete exchange-rate forecasting pipeline."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from currency_explorer.data import (
    get_country_currency,
    get_exchange_rate,
    get_historical_rates,
)
from currency_explorer.forecasting import (
    backtest_monday_forecasts,
    calculate_metrics,
    find_best_sarima,
    forecast_next_monday,
)
from currency_explorer.reporting import generate_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an exchange-rate forecast report."
    )
    parser.add_argument("country", nargs="?", default="Germany")
    parser.add_argument("--to", default="USD", dest="target_currency")
    parser.add_argument("--days", default=730, type=int)
    parser.add_argument("--tuning-size", default=252, type=int)
    parser.add_argument("--output", default="results")
    return parser.parse_args()


def main() -> int:
    """Run data collection, tuning, backtesting, forecasting, and reporting."""
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError("API_KEY is not configured in .env")

    country_info = get_country_currency(args.country, api_key)
    current_rate = get_exchange_rate(
        country_info["currency"],
        args.target_currency
    )
    historical_rates = get_historical_rates(
        country_info["currency"],
        args.target_currency,
        args.days
    )

    if len(historical_rates) <= args.tuning_size:
        raise ValueError("Historical data must be longer than the tuning period")

    tuning_history = historical_rates["rate"].iloc[:args.tuning_size]
    search_results = find_best_sarima(tuning_history)
    best_order = search_results.loc[0, "order"]
    best_seasonal_order = search_results.loc[0, "seasonal_order"]

    model_results = backtest_monday_forecasts(
        rates_df=historical_rates,
        order=best_order,
        seasonal_order=best_seasonal_order,
        tuning_size=args.tuning_size
    )
    predictions = model_results.dropna(subset=[
        "naive_prediction",
        "sarima_prediction",
        "holt_prediction",
    ]).copy()

    if predictions.empty:
        raise ValueError("No complete Monday forecasts were produced")

    metrics = pd.DataFrame({
        "Naive": calculate_metrics(predictions["actual_rate"], predictions["naive_prediction"]),
        "SARIMA": calculate_metrics(predictions["actual_rate"], predictions["sarima_prediction"]),
        "Holt": calculate_metrics(predictions["actual_rate"], predictions["holt_prediction"]),
    }).T
    next_forecast = forecast_next_monday(
        historical_rates,
        best_order,
        best_seasonal_order
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=True)

    model_config = {
        "country": country_info["country"],
        "base_currency": country_info["currency"],
        "target_currency": args.target_currency.upper(),
        "current_rate": current_rate,
        "order": list(best_order),
        "seasonal_order": list(best_seasonal_order),
        "tuning_size": args.tuning_size,
        "data_start": historical_rates["date"].min().date().isoformat(),
        "data_end": historical_rates["date"].max().date().isoformat(),
        "tuning_start": historical_rates["date"].iloc[0].date().isoformat(),
        "tuning_end": historical_rates["date"].iloc[args.tuning_size - 1].date().isoformat(),
        "evaluation_start": predictions["forecast_date"].min().date().isoformat(),
        "evaluation_end": predictions["forecast_date"].max().date().isoformat(),
    }
    (output_dir / "sarima_config.json").write_text(
        json.dumps(model_config, indent=4),
        encoding="utf-8"
    )

    serializable_forecast = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in next_forecast.items()
    }
    (output_dir / "next_monday_forecast.json").write_text(
        json.dumps(serializable_forecast, indent=4),
        encoding="utf-8"
    )
    report_path = generate_report(
        predictions,
        metrics,
        model_config,
        serializable_forecast,
        output_dir
    )

    print("Best SARIMA order:", best_order)
    print("Best seasonal order:", best_seasonal_order)
    print(metrics.to_string())
    print("Next Monday forecast:")
    print(pd.DataFrame([serializable_forecast]).to_string(index=False))
    print("Report:", report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
