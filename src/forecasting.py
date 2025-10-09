"""Forecast generation utilities.

The module wraps the prophet based demand and carbon forecasts used throughout
the project.  The functions provide light-weight caching as well as helper
utilities to emulate short-term forecast errors by bootstrapping historical
residuals from the carbon intensity signal.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.util import DT_INDEX

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")


def load_prophet_forecast(
    data: pd.DataFrame,
    i: int,
    forecast_params: dict,
    cache_key: str,
) -> pd.DataFrame:
    """Load a prophet forecast from disk or compute it on-demand."""

    path = f"{_CACHE_DIR}/{cache_key}.pkl"
    try:
        fc = pd.read_pickle(path)
    except FileNotFoundError:
        print(f"Generating forecast for {cache_key}...")
        fc, _ = generate_prophet_forecast(data, i, forecast_params)
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fc.to_pickle(path)
    return fc


def generate_prophet_forecast(
    data: pd.DataFrame,
    i: int,
    forecast_params: dict,
    prophet_params: dict | None = None,
) -> Tuple[pd.DataFrame, "Prophet"]:
    """Fit a Prophet model and return the forecast starting at step ``i``."""

    from prophet import Prophet

    if prophet_params is None:
        prophet_params = {}

    cutoff = -len(DT_INDEX) + i
    train = data[:cutoff].copy()

    # Configure the growth model depending on the requested bounds.
    if "cap" in forecast_params:
        train["cap"] = forecast_params["cap"]
        train["floor"] = forecast_params["floor"]
        model = Prophet(growth="logistic", **prophet_params)
    elif "flat" in forecast_params:
        model = Prophet(growth="flat", **prophet_params)
    else:
        model = Prophet(**prophet_params)
    model.fit(train)

    future = model.make_future_dataframe(freq="h", periods=24 * 365)
    if "cap" in forecast_params:
        future["cap"] = forecast_params["cap"]
        future["floor"] = forecast_params["floor"]
    forecast = model.predict(future)

    return forecast[cutoff:], model


def generate_bootstrapped_forecast(
    forecast: np.ndarray, historical_errors: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Perturb a forecast by resampling historical relative errors.

    Parameters
    ----------
    forecast:
        The baseline forecast values for the horizon of interest.
    historical_errors:
        Relative errors observed on previous horizons (``(actual - forecast) / forecast``).
    rng:
        Random number generator used for reproducible bootstrapping.
    """

    if historical_errors.size == 0:
        return forecast

    sampled_errors = rng.choice(historical_errors, size=len(forecast), replace=True)
    sampled_errors = np.clip(sampled_errors, -0.95, 5.0)
    return forecast * (1 + sampled_errors)
