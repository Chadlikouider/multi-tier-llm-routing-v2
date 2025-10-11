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



if __name__ == "__main__":
    zone = "US-CAL-CISO"
    split = "train"
    year = 2023

    print(f"Loading Electricity Maps data for {zone} ({split=} {year=})...")
    data = load_electricitymaps_timeseries(zone, split=split, year=year)
    if data.empty:
        raise RuntimeError("The loaded dataset is empty; cannot train a forecast model.")

    horizon = len(DT_INDEX) - 24
    print("Training Prophet model...")
    forecast, model = forecasting.generate_prophet_forecast(
        data=data,
        i=horizon,
        forecast_params={},
    )

    output_dir = PROJECT_ROOT / "cache" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"{zone}_forecast.pkl"
    model_path = output_dir / f"{zone}_prophet.json"

    print(f"Saving forecast to {forecast_path}...")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_pickle(forecast_path)

    print(f"Serializing Prophet model to {model_path}...")
    from prophet.serialize import model_to_json

    model_path.write_text(model_to_json(model))
    print("Example run complete.")

