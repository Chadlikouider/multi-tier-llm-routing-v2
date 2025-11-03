"""Shared utilities for time-series processing across the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ``DT_INDEX`` represents the canonical time range used across the optimisation
# pipeline.  A full year of hourly data keeps the arrays manageable while still
# providing sufficient coverage for seasonal patterns in the historical data.
DT_INDEX = pd.date_range("2024-01-01", periods=24 * 365, freq="h")

_ELECTRICITYMAPS_DIR = (Path(__file__).resolve().parent / ".." / "data" / "electricitymaps").resolve()


def _extract_year_from_filename(path: Path) -> int:
    """Extract the four digit year from an Electricity Maps CSV filename."""

    for part in path.stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"Could not determine year from Electricity Maps filename '{path.name}'")


def electricitymaps_csv_path(zone: str, split: str = "train", year: int | None = None) -> Path:
    """Return the path to an Electricity Maps CSV file for ``zone`` and ``split``.

    Parameters
    ----------
    zone:
        Electricity Maps zone identifier, e.g. ``"US-CAL-CISO"``.
    split:
        Dataset split folder to load data from (usually ``"train"`` or ``"test"``).
    year:
        Optional four digit year.  When omitted the most recent available year is
        returned.

    Raises
    ------
    FileNotFoundError
        If the requested split or zone file does not exist on disk.
    ValueError
        If the year cannot be parsed from a matching file name.
    """

    split_dir = _ELECTRICITYMAPS_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Electricity Maps split '{split}' not found at {split_dir}")

    matches = sorted(split_dir.glob(f"{zone}_*_hourly.csv"))
    if not matches:
        raise FileNotFoundError(f"No Electricity Maps files found for zone '{zone}' in '{split_dir}'")

    if year is not None:
        matches = [path for path in matches if f"_{year}_" in path.stem]
        if not matches:
            raise FileNotFoundError(
                f"No Electricity Maps files found for zone '{zone}' and year '{year}' in '{split_dir}'"
            )

    if len(matches) > 1:
        matches.sort(key=_extract_year_from_filename)
    return matches[-1]


def load_electricitymaps_timeseries(
    zone: str,
    split: str = "train",
    value_column: str = "Carbon intensity gCO₂eq/kWh (direct)",
    year: int | None = None,
) -> pd.DataFrame:
    """Load Electricity Maps carbon-intensity data as a prophet-compatible frame.

    The returned data frame contains ``ds`` and ``y`` columns sorted by timestamp
    and aligned to :data:`DT_INDEX` to guarantee compatibility with the
    forecasting utilities.  When ``year`` is not provided the latest available
    dataset for the requested ``zone`` is used.
    """

    csv_path = electricitymaps_csv_path(zone, split, year)

    df = pd.read_csv(csv_path, parse_dates=["Datetime (UTC)"])
    if value_column not in df.columns:
        raise KeyError(
            f"Column '{value_column}' not present in Electricity Maps file '{csv_path.name}'"
        )

    timeseries = (
        df.rename(columns={"Datetime (UTC)": "ds", value_column: "y"})[["ds", "y"]]
        .sort_values("ds")
        .set_index("ds")
    )

    aligned = timeseries.reindex(DT_INDEX)
    aligned = aligned.dropna(subset=["y"]).rename_axis("ds").reset_index()
    return aligned


__all__ = ["DT_INDEX", "electricitymaps_csv_path", "load_electricitymaps_timeseries"]
