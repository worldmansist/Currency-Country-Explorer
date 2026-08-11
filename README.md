# Currency & Country Explorer

An educational Python project that combines country metadata, exchange-rate
APIs, time-series forecasting, backtesting, and automated HTML reporting.

The user selects a country. The project identifies its primary currency,
downloads historical rates, compares three Monday forecasting models, and
produces a forecast for the next Monday that is not yet present in the data.

## Workflow

```text
Country name
    -> REST Countries API v5
    -> currency code
    -> Frankfurter historical rates
    -> Naive / SARIMA / Holt forecasts
    -> Monday backtest and MAE/RMSE
    -> next Monday forecast
    -> HTML report and PNG charts
```

This project is for learning and demonstration. Its forecasts are not
financial advice.

## Project structure

```text
currency_explorer/
    data.py          API clients and historical-data preparation
    forecasting.py   SARIMA tuning, Holt, naive baseline, and backtesting
    reporting.py     charts and HTML report generation
notebooks/
    01_rest_countries.ipynb
    02_exchange_rate_forecasting.ipynb
    03_forecast_analysis.ipynb
tests/               offline unit tests
main.py              complete command-line pipeline
requirements.txt     runtime dependencies
requirements-dev.txt notebook and test dependencies
```

The `results/` directory is generated at runtime and is intentionally ignored
by Git, except for its placeholder file.

## Prerequisites

- Python 3.11 or newer is recommended.
- A REST Countries v5 API key. Create an account at
  [restcountries.com/sign-up](https://restcountries.com/sign-up).

Never paste an API key into a notebook, commit, issue, or chat message. If a
key is exposed, revoke it and generate a replacement.

## Installation on Windows

Create a virtual environment:

```powershell
python -m venv .venv
```

If `python` opens the Microsoft Store or is not found, install Python from
[python.org](https://www.python.org/downloads/) and enable **Add Python to
PATH** during installation.

PowerShell activation is optional. Using the environment's executable
directly avoids execution-policy problems:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder:

```dotenv
API_KEY=your_new_rest_countries_key
```

The `.env` file is ignored by Git.

## Complete run

From the project root, run:

```powershell
.\.venv\Scripts\python.exe main.py Germany --to USD --days 730
```

Options:

- positional `country`: country to analyze; default is `Germany`
- `--to`: target currency; default is `USD`
- `--days`: calendar days of historical data; default is `730`
- `--tuning-size`: observations used to select SARIMA; default is `252`
- `--output`: generated-results directory; default is `results`

SARIMA tuning and the Monday backtest can take several minutes. The command
creates:

```text
results/
    report.html
    predictions.csv
    metrics.csv
    sarima_config.json
    next_monday_forecast.json
    figures/
        01_forecasts.png
        02_metrics.png
        03_absolute_errors.png
        04_error_distributions.png
```

The future Monday has no actual rate yet, so the report does not assign an
error to that forecast. MAE and RMSE are calculated only from historical
Monday forecasts with known outcomes.

## Opening the report

Serve the generated directory locally:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory results
```

Then open:

```text
http://127.0.0.1:8000/report.html
```

Stop the server with `Ctrl+C`. Do not open the `[::]:8000` listening address;
it is not a browser URL.

## Jupyter notebooks

Start JupyterLab without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Run the notebooks in order:

1. `01_rest_countries.ipynb` — country and exchange-rate data
2. `02_exchange_rate_forecasting.ipynb` — tuning and Monday backtest
3. `03_forecast_analysis.ipynb` — charts and result interpretation

The Python modules contain the reusable implementation; notebooks document
the learning process and analysis.

## Tests

The test suite does not require API access:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Modeling notes

- Naive uses the latest observation before Monday.
- Holt uses a damped trend fitted to recent observations.
- SARIMA uses a five-observation seasonal period and parameters selected by
  AIC on an initial tuning period.
- Backtest predictions begin after the tuning period to avoid using future
  observations during model selection.
