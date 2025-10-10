"""Integration test exercising Electricity Maps data with forecasting utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import forecasting
from src.util import DT_INDEX, load_electricitymaps_timeseries


@pytest.fixture()
def cache_dir(tmp_path):
    """Point the forecasting cache to a temporary directory for the test."""

    original_cache_dir = forecasting._CACHE_DIR
    forecasting._CACHE_DIR = tmp_path
    yield tmp_path
    forecasting._CACHE_DIR = original_cache_dir


def test_train_and_cache_electricitymaps_forecast(cache_dir):
    """Train a Prophet model on Electricity Maps data and verify caching works."""

    data = load_electricitymaps_timeseries("US-CAL-CISO", split="train", year=2023)
    assert not data.empty
    # Ensure we have sufficient history to produce a non-empty training window.
    assert len(data) >= len(DT_INDEX) - 24

    cache_key = "US-CAL-CISO-train"
    forecast = forecasting.load_prophet_forecast(
        data=data,
        i=len(DT_INDEX) - 24,
        forecast_params={},
        cache_key=cache_key,
    )

    assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(forecast.columns)
    assert len(forecast) >= 24

    cache_file = cache_dir / f"{cache_key}.pkl"
    assert cache_file.exists()

    cached = pd.read_pickle(cache_file)
    pd.testing.assert_frame_equal(cached, forecast)

    cached_again = forecasting.load_prophet_forecast(
        data=data,
        i=len(DT_INDEX) - 24,
        forecast_params={},
        cache_key=cache_key,
    )
    pd.testing.assert_frame_equal(cached_again, forecast)
