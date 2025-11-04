
from __future__ import annotations


import pandas as pd

# ``DT_INDEX`` represents the canonical time range used across the optimisation
# pipeline.  A full year of hourly data keeps the arrays manageable while still
# providing sufficient coverage for seasonal patterns in the historical data.
DT_INDEX = pd.date_range("2024-01-01", periods=24 * 365, freq="h")





def get_validity_periods(window: list[int],
                         vp: int,
                         past: bool = True,
                         future: bool = True) -> list[list[int]]:
    """
    Generate lists of index ranges (validity periods) within a given time window.

    Args:
        window (list[int]): A two-element list [start_idx, end_idx] defining the 
            inclusive range of valid indices (the current time window).
        vp (int): The validity period length — the number of consecutive indices 
            in each period.
        past (bool, optional): Whether to include periods that start before the 
            current window (historical context). Defaults to True.
        future (bool, optional): Whether to include periods that end after the 
            current window (future projections). Defaults to True.

    Returns:
        list[list[int]]: A list of index lists, where each inner list represents 
        a consecutive range of indices (a validity period) within the DT_INDEX timeline.

    Description:
        The function slides a window of length `vp` across the full range of `DT_INDEX`,
        constructing lists of consecutive indices (from `start` to `end`) that 
        overlap with the specified `window`. 

        It respects the `past` and `future` flags:
          - If `past=False`, periods that start before the given window are excluded.
          - If `future=False`, iteration stops once the end index exceeds the window.

        The iteration halts early when the start index moves beyond the end of the window 
        to improve efficiency.

    Example:
        Suppose DT_INDEX = range(10), window = [3, 6], vp = 3
        get_validity_periods(window, vp)
        ➜ [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    """
    periods = []
    for end in range(len(DT_INDEX)):
        start = end - vp + 1
        l = list(range(start, end + 1))
        if not future and end > window[-1]:
            break
        if 0 <= start <= window[-1] and end >= window[0]:
            if past or start >= window[0]:
                periods.append(l)
        if start > window[-1]:
            break
    return periods



__all__ = ["DT_INDEX", "get_validity_periods"]
