"""Shared utilities for time-series processing across the project."""

from __future__ import annotations

import pandas as pd

# ``DT_INDEX`` represents the canonical time range used across the optimisation
# pipeline.  A full year of hourly data keeps the arrays manageable while still
# providing sufficient coverage for seasonal patterns in the historical data.
DT_INDEX = pd.date_range("2023-01-01", periods=24 * 365, freq="h")

__all__ = ["DT_INDEX"]
