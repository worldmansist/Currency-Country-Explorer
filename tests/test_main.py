"""Smoke tests for the command-line entry point."""

import main


def test_cli_parser_has_expected_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])

    args = main.parse_args()

    assert args.country == "Germany"
    assert args.target_currency == "USD"
    assert args.days == 730
    assert args.tuning_size == 252
    assert args.output == "results"
