"""Tests for data validation that does not require network access."""

import pytest

from currency_explorer.data import get_country_currency, get_historical_rates


def test_country_name_cannot_be_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        get_country_currency("   ", "example-key")


def test_historical_days_must_be_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        get_historical_rates("EUR", "USD", days=0)
