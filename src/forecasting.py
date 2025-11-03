"""Utilities for generating and caching forecast data.

This module provides a thin wrapper around :mod:`prophet` so that training a
forecast model and retrieving the resulting predictions can be handled in a
consistent manner throughout the project.  The helpers also offer utilities for
light-weight on-disk caching as well as simple stochastic perturbations that are
used to emulate short-term forecast errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet  # Imported lazily to keep import time minimal.


from src.util import DT_INDEX

# Cache forecasts so expensive prophet fits are only performed once per set of
# parameters.  ``Path`` provides convenient / operator overloads and better type
# checking compared to manual ``os.path`` manipulation.
_CACHE_DIR = (Path(__file__).resolve().parent / ".." / "cache").resolve()


def _forecast_cache_path(cache_key: str) -> Path:
    """Return the absolute path on disk for a forecast cache file."""

    return _CACHE_DIR / f"{cache_key}.pkl"


def load_prophet_forecast(
    data: pd.DataFrame,
    i: int,
    forecast_params: Dict,
    cache_key: str,
) -> pd.DataFrame:
    """Load a cached forecast or generate one on demand.

    The function first attempts to retrieve a previously cached forecast based on
    ``cache_key``.  When the cache is missing, a new prophet model is fitted via
    :func:`generate_prophet_forecast`, the results are stored on disk, and the
    essential columns are returned to the caller.
    """

    path = _forecast_cache_path(cache_key)
    try:
        forecast = pd.read_pickle(path)
    except FileNotFoundError:
        print(f"Generating forecast for {cache_key}...")
        forecast, _ = generate_prophet_forecast(data, i, forecast_params)
        forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_pickle(path)
    return forecast


def generate_prophet_forecast(
    data: pd.DataFrame,
    i: int,
    forecast_params: Dict,
    prophet_params: Dict | None = None,
) -> Tuple[pd.DataFrame, "Prophet"]:
    """Fit a Prophet model and return the forecast starting at step ``i``.

    Parameters
    ----------
    data:
        Full historical time-series data containing ``ds`` and ``y`` columns.
    i:
        Index into :data:`~src.util.DT_INDEX` indicating the first horizon step
        that should be returned from the forecast.
    forecast_params:
        Configuration for the Prophet growth model (e.g. logistic bounds).
    prophet_params:
        Additional keyword arguments forwarded directly to :class:`prophet.Prophet`.
    """


    if prophet_params is None:
        prophet_params = {}

    # ``cutoff`` represents the location where the forecast horizon begins.
    cutoff = -len(DT_INDEX) + i
    print("cutoff:", cutoff)
    train = data[:cutoff].copy()
    
    print("values of train:", train)
    
    # Configure the growth model depending on the requested bounds.  Logistic
    # growth requires ``cap`` and ``floor`` columns, whereas ``flat`` simply
    # holds the series constant outside the observed range.
    if "cap" in forecast_params:
        train["cap"] = forecast_params["cap"]
        train["floor"] = forecast_params["floor"]
        model = Prophet(growth="logistic", **prophet_params)
    elif "flat" in forecast_params:
        model = Prophet(growth="flat", **prophet_params)
    else:
        model = Prophet(**prophet_params)
    model.fit(train)

    # Generate a full year (365 days) of hourly predictions, which matches the
    # expected resolution for downstream consumers in the project.
    future = model.make_future_dataframe(freq="h", periods=24 * 365)
    if "cap" in forecast_params:
        future["cap"] = forecast_params["cap"]
        future["floor"] = forecast_params["floor"]
    forecast = model.predict(future)

    return forecast[cutoff:], model


def generate_bootstrapped_forecast(
    forecast: np.ndarray,
    historical_errors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb a forecast by resampling historical relative errors.

    Bootstrapping allows us to simulate alternative forecast trajectories by
    sampling previously observed errors.  The sampled errors are clipped to avoid
    unrealistic negative or explosive corrections, then scaled to the base
    forecast.
    """

    if historical_errors.size == 0:
        return forecast

    sampled_errors = rng.choice(historical_errors, size=len(forecast), replace=True)
    sampled_errors = np.clip(sampled_errors, -0.95, 5.0)
    return forecast * (1 + sampled_errors)

