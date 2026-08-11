"""Report and chart generation for forecast results."""

import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREDICTION_COLUMNS = {
    "Naive": "naive_prediction",
    "SARIMA": "sarima_prediction",
    "Holt": "holt_prediction",
}


def _save_forecast_chart(predictions_df, figures_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        predictions_df["forecast_date"],
        predictions_df["actual_rate"],
        color="black",
        linewidth=2.5,
        label="Actual"
    )

    for model_name, column in PREDICTION_COLUMNS.items():
        ax.plot(
            predictions_df["forecast_date"],
            predictions_df[column],
            linestyle="--",
            label=model_name
        )

    ax.set_title("Actual Monday rates and one-step forecasts")
    ax.set_xlabel("Forecast date")
    ax.set_ylabel("Exchange rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "01_forecasts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_metrics_chart(metrics_df, figures_dir):
    ax = metrics_df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Forecast error by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric")
    fig = ax.get_figure()
    fig.tight_layout()
    path = figures_dir / "02_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_error_frame(predictions_df):
    error_df = predictions_df[["forecast_date"]].copy()

    for model_name, column in PREDICTION_COLUMNS.items():
        error_df[model_name] = (
            predictions_df["actual_rate"] - predictions_df[column]
        ).abs()

    return error_df


def _save_error_charts(error_df, figures_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    for model_name in PREDICTION_COLUMNS:
        ax.plot(
            error_df["forecast_date"],
            error_df[model_name],
            label=model_name
        )

    ax.set_title("Absolute forecast errors over time")
    ax.set_xlabel("Forecast date")
    ax.set_ylabel("Absolute error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    time_path = figures_dir / "03_absolute_errors.png"
    fig.savefig(time_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ax = error_df[list(PREDICTION_COLUMNS)].plot.box(
        figsize=(9, 5),
        showfliers=True
    )
    ax.set_title("Distribution of absolute forecast errors")
    ax.set_xlabel("Model")
    ax.set_ylabel("Absolute error")
    ax.grid(axis="y", alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    distribution_path = figures_dir / "04_error_distributions.png"
    fig.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return time_path, distribution_path


def generate_report(
    predictions_df,
    metrics_df,
    model_config,
    next_forecast,
    output_dir
):
    """Create charts and an HTML summary, and return the report path."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_forecast_chart(predictions_df, figures_dir)
    _save_metrics_chart(metrics_df, figures_dir)
    error_df = _build_error_frame(predictions_df)
    _save_error_charts(error_df, figures_dir)

    periods_df = pd.DataFrame([
        {"Period": "Historical data", "Start": model_config["data_start"], "End": model_config["data_end"]},
        {"Period": "SARIMA tuning", "Start": model_config["tuning_start"], "End": model_config["tuning_end"]},
        {"Period": "Backtest", "Start": model_config["evaluation_start"], "End": model_config["evaluation_end"]},
    ])
    future_df = pd.DataFrame([next_forecast])
    best_mae = metrics_df["MAE"].idxmin()
    best_rmse = metrics_df["RMSE"].idxmin()
    title = (
        f"{model_config['country']} — "
        f"{model_config['base_currency']}/{model_config['target_currency']}"
    )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)} forecast report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #202124; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; margin: 12px 0 30px; }}
    .note {{ background: #f4f6f8; padding: 14px; border-left: 4px solid #607d8b; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)} forecast report</h1>
  <p class="note">Educational forecast only. The future Monday has no actual rate yet, so no error metric is available for it.</p>
  <h2>Analyzed periods</h2>
  {periods_df.to_html(index=False, border=0)}
  <h2>Backtest errors</h2>
  {metrics_df.to_html(border=0, float_format=lambda value: f'{value:.6f}')}
  <p>Lowest MAE: <strong>{html.escape(best_mae)}</strong>. Lowest RMSE: <strong>{html.escape(best_rmse)}</strong>.</p>
  <h2>Next Monday forecast</h2>
  {future_df.to_html(index=False, border=0, float_format=lambda value: f'{value:.6f}')}
  <h2>Forecasts</h2><img src="figures/01_forecasts.png" alt="Forecasts">
  <h2>Error metrics</h2><img src="figures/02_metrics.png" alt="Metrics">
  <h2>Absolute errors</h2><img src="figures/03_absolute_errors.png" alt="Errors">
  <h2>Error distributions</h2><img src="figures/04_error_distributions.png" alt="Error distributions">
</body>
</html>
"""
    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path
