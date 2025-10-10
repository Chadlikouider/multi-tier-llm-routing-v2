"""Unit tests for :mod:`src.forecasting`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import forecasting


@pytest.fixture(autouse=True)
def restore_cache_dir(monkeypatch):
    """Ensure the cache directory is restored after each test."""

    original_cache_dir = forecasting._CACHE_DIR
    yield
    forecasting._CACHE_DIR = original_cache_dir


def _sample_forecast_dataframe() -> pd.DataFrame:
    """Create a minimal dataframe that mimics prophet output columns."""

    return pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=4, freq="h"),
            "yhat": [1.0, 2.0, 3.0, 4.0],
            "yhat_lower": [0.5, 1.5, 2.5, 3.5],
            "yhat_upper": [1.5, 2.5, 3.5, 4.5],
        }
    )


def test_load_prophet_forecast_uses_cache(monkeypatch, tmp_path):
    """Loading from cache should avoid re-computing the forecast."""

    cache_file = tmp_path / "cached_forecast.pkl"
    _sample_forecast_dataframe().to_pickle(cache_file)

    forecasting._CACHE_DIR = tmp_path
    monkeypatch.setattr(forecasting, "_forecast_cache_path", lambda key: cache_file)

    # Patch the generator to raise if called, ensuring the cache is used.
    def _unexpected_call(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("generate_prophet_forecast should not be called when cache exists")

    monkeypatch.setattr(forecasting, "generate_prophet_forecast", _unexpected_call)

    result = forecasting.load_prophet_forecast(pd.DataFrame(), 0, {}, "cached_forecast")

    pd.testing.assert_frame_equal(result, _sample_forecast_dataframe())


def test_load_prophet_forecast_generates_when_missing(monkeypatch, tmp_path):
    """When the cache is missing a forecast should be generated and stored."""

    forecasting._CACHE_DIR = tmp_path
    cache_path = tmp_path / "new_forecast.pkl"

    generated = _sample_forecast_dataframe()
    monkeypatch.setattr(
        forecasting, "generate_prophet_forecast", lambda *args, **kwargs: (generated, None)
    )

    result = forecasting.load_prophet_forecast(pd.DataFrame(), 0, {}, "new_forecast")

    pd.testing.assert_frame_equal(result, generated)
    assert cache_path.exists()
    cached = pd.read_pickle(cache_path)
    pd.testing.assert_frame_equal(cached, generated)


def test_generate_bootstrapped_forecast_samples_errors():
    """Bootstrapped forecasts should apply sampled relative errors."""

    baseline = np.array([10.0, 10.0, 10.0])
    historical = np.array([0.1, -0.2, 0.05])
    rng = np.random.default_rng(42)

    result = forecasting.generate_bootstrapped_forecast(baseline, historical, rng)

    # With the fixed seed we expect a deterministic sampled error vector.
    expected_errors = np.array([0.1, 0.05, -0.2])
    np.testing.assert_allclose(result, baseline * (1 + expected_errors))


def test_generate_bootstrapped_forecast_no_errors():
    """If no historical errors are available the forecast should be unchanged."""

    baseline = np.array([5.0, 5.0])
    rng = np.random.default_rng(123)

    result = forecasting.generate_bootstrapped_forecast(baseline, np.array([]), rng)

    np.testing.assert_allclose(result, baseline)
