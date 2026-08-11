"""API clients for country metadata and exchange-rate data."""

from datetime import date, timedelta

import pandas as pd
import requests


COUNTRIES_URL = "https://api.restcountries.com/countries/v5"
LATEST_RATES_URL = "https://api.frankfurter.app/latest"
HISTORICAL_RATES_URL = "https://api.frankfurter.app/{start}..{end}"


def get_country_currency(country: str, api_key: str) -> dict[str, str]:
    """Return the country name and its primary currency code."""
    country = country.strip()

    if not country:
        raise ValueError("Country name cannot be empty")

    if not api_key:
        raise ValueError("A REST Countries API key is required")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "q": country,
        "pretty": 1
    }

    response = requests.get(
        COUNTRIES_URL,
        headers=headers,
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    objects = data["data"]["objects"]

    if not objects:
        raise ValueError(f"Country not found: {country}")

    country_data = objects[0]

    return {
        "country": country_data["names"]["common"],
        "currency": country_data["currencies"][0]["code"]
    }


def get_exchange_rate(
    base_currency: str,
    target_currency: str = "USD"
) -> float:
    """Return the latest exchange rate for a currency pair."""
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()

    if base_currency == target_currency:
        return 1.0

    params = {
        "from": base_currency,
        "to": target_currency
    }

    response = requests.get(
        LATEST_RATES_URL,
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()

    return float(data["rates"][target_currency])


def get_historical_rates(
    base_currency: str,
    target_currency: str = "USD",
    days: int = 730
) -> pd.DataFrame:
    """Return historical rates as a date-sorted pandas DataFrame."""
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()

    if base_currency == target_currency:
        raise ValueError("Base and target currencies must be different")

    if days <= 0:
        raise ValueError("Days must be greater than zero")

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    url = HISTORICAL_RATES_URL.format(
        start=start_date,
        end=end_date
    )

    params = {
        "from": base_currency,
        "to": target_currency
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )
    response.raise_for_status()

    data = response.json()
    rows = []

    for rate_date, currencies in data["rates"].items():
        rows.append({
            "date": pd.to_datetime(rate_date),
            "rate": currencies[target_currency]
        })

    rates_df = pd.DataFrame(rows)

    if rates_df.empty:
        raise ValueError("No historical exchange rates were returned")

    rates_df = rates_df.sort_values("date")
    rates_df = rates_df.reset_index(drop=True)

    return rates_df
