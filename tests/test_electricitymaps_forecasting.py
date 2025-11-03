"""
Configurable rolling Prophet forecast for a single region.

- History: 2022–2023
- Forecast target: full year 2024 (hourly)
- Retraining frequency: daily / weekly / monthly (configurable)
- Output: CSV with forecasts & errors + PNG plot (actual vs latest forecast)

Usage:
    python forecast_configurable_retraining.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


# ===================== Config =====================

DATA_DIR = Path("data/electricitymaps")
VALUE_COL = "Carbon intensity gCO₂eq/kWh (Life cycle)"
OUTPUT_DIR = Path("output_forecasts")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# >>>>>>>>>>>>>>>>> SELECT YOUR REGION & RETRAINING CADENCE <<<<<<<<<<<<<<<<<
REGION = "AU-QLD"                # e.g., "US-MIDA-PJM", "DE", "US-NY-NYIS"
RETRAIN_FREQUENCY = "weekly"     # "daily", "weekly", or "monthly"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ===================== Data IO =====================

def _read_csv(file: Path) -> pd.DataFrame:
    """Read electricitymaps CSV and standardize columns to Prophet schema (ds,y)."""
    df = pd.read_csv(file, parse_dates=["Datetime (UTC)"])
    df = df.rename(columns={"Datetime (UTC)": "ds", VALUE_COL: "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df[["ds", "y"]].sort_values("ds")

def load_region_train(region: str) -> pd.DataFrame:
    """Load 2022–2023 training data for a region."""
    train_dir = DATA_DIR / "train"
    files = sorted(train_dir.glob(f"{region}_20[2][2-3]_hourly.csv"))
    if not files:
        raise FileNotFoundError(f"No training CSVs for {region} in {train_dir}")
    df = pd.concat((_read_csv(f) for f in files), ignore_index=True)
    df["y"] = df["y"].ffill().bfill()
    return df

def load_region_2024(region: str) -> pd.DataFrame:
    """Load 2024 test data for a region."""
    test_dir = DATA_DIR / "test"
    file = test_dir / f"{region}_2024_hourly.csv"
    if not file.exists():
        raise FileNotFoundError(f"Missing test CSV {file}")
    df = _read_csv(file)
    df["y"] = df["y"].ffill().bfill()
    return df

def load_history_plus_2024(region: str) -> pd.DataFrame:
    """Concatenate 2022–2023 train + 2024 test for slicing history at cutoffs."""
    hist = load_region_train(region)
    y2024 = load_region_2024(region)
    return pd.concat([hist, y2024], ignore_index=True).sort_values("ds")


# ===================== Prophet Model =====================

def make_prophet() -> Prophet:
    """Configure Prophet for hourly data with explicit intraday seasonality."""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.8,
    )
    # Explicit 24-hour cycle for sub-daily patterns
    m.add_seasonality(name="intraday", period=24, fourier_order=10)
    return m


# ===================== Rolling Forecast =====================

def _freq_code(name: str) -> str:
    """
    Map human-friendly frequency to pandas freq code for cutoffs:
      - "daily"   -> "D"      (every day at 00:00)
      - "weekly"  -> "W-MON"  (every Monday at 00:00)
      - "monthly" -> "MS"     (first of each month at 00:00)
    """
    name = name.strip().lower()
    if name in ("d", "day", "daily"):
        return "D"
    if name in ("w", "week", "weekly"):
        return "W-MON"
    if name in ("m", "month", "monthly"):
        return "MS"
    raise ValueError(f"Unknown retrain frequency: {name}")

def simulate_rolling_forecasts(region: str, retrain_frequency: str) -> pd.DataFrame:
    """
    Retrain a Prophet model on a configurable cadence in 2024.
    - At each cutoff, fit on the previous ~2 years (730 days) up to (but not including) cutoff.
    - Forecast hourly from cutoff to 2024-12-31 23:00.
    - Keep the latest forecast for each timestamp.

    Returns:
        DataFrame[ds, yhat, cutoff] for all hours in 2024.
    """
    all_data = load_history_plus_2024(region)

    start_2024 = pd.Timestamp("2024-01-01 00:00:00")
    end_2024   = pd.Timestamp("2024-12-31 23:00:00")

    # Initialize "latest" forecast container for 2024 hours
    timeline = pd.date_range(start=start_2024, end=end_2024, freq="h")
    latest_forecast = pd.DataFrame({"ds": timeline, "yhat": np.nan, "cutoff": pd.NaT})

    # Build cutoff schedule
    cutoff_freq = _freq_code(retrain_frequency)
    cutoffs = pd.date_range(start=start_2024, end=end_2024, freq=cutoff_freq)
    two_years = pd.Timedelta(days=730)

    print(f"\n▶ Region: {region}")
    print(f"▶ Retraining cadence: {retrain_frequency}  (cutoffs: {len(cutoffs)})")

    # Iterate over each cutoff, fit, and update the latest forecast
    for cutoff in cutoffs:
        window_start = cutoff - two_years
        history = all_data[(all_data["ds"] >= window_start) & (all_data["ds"] < cutoff)]

        # Ensure we have enough history to fit a decent model
        if len(history) < 1000:
            # If early in the year you don't have a full 2-year window (e.g., data starts 2022-01-01),
            # this still fits once there are enough points.
            continue

        m = make_prophet()
        m.fit(history)

        future = pd.DataFrame({"ds": pd.date_range(start=cutoff, end=end_2024, freq="h")})
        fcst = m.predict(future)[["ds", "yhat"]]
        fcst["cutoff"] = cutoff

        # Merge and overwrite with the most recent forecast
        latest_forecast = latest_forecast.merge(fcst, on="ds", how="left", suffixes=("", "_new"))
        mask = latest_forecast["yhat_new"].notna()
        latest_forecast.loc[mask, "yhat"] = latest_forecast.loc[mask, "yhat_new"]
        latest_forecast.loc[mask, "cutoff"] = latest_forecast.loc[mask, "cutoff_new"]
        latest_forecast = latest_forecast.drop(columns=["yhat_new", "cutoff_new"])

    return latest_forecast


# ===================== Evaluation & Plot =====================

def evaluate_and_plot(region: str, retrain_frequency: str) -> pd.DataFrame:
    """Run rolling forecasts, compute MAE vs 2024 actuals, and plot results."""
    fcst = simulate_rolling_forecasts(region, retrain_frequency)
    actual = load_region_2024(region).rename(columns={"y": "y_actual"})

    eval_df = fcst.merge(actual, on="ds", how="left")
    eval_df["abs_error"] = (eval_df["yhat"] - eval_df["y_actual"]).abs()
    mae = eval_df["abs_error"].mean()
    print(f"✅ {region} — MAE ({retrain_frequency} retraining): {mae:.2f} gCO₂eq/kWh")

    # Save CSV
    tag = retrain_frequency.lower()
    out_csv = OUTPUT_DIR / f"{region}_rolling_{tag}_forecast_2024.csv"
    eval_df.to_csv(out_csv, index=False)
    print(f"Saved results: {out_csv}")

    # Plot Actual vs Latest Forecast (full year)
    plot_actual_vs_forecast(eval_df, region, retrain_frequency)
    return eval_df

def plot_actual_vs_forecast(eval_df: pd.DataFrame, region: str, retrain_frequency: str):
    """Plot actual vs latest forecast for 2024."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eval_df["ds"], eval_df["y_actual"], label="Actual", linewidth=1)
    ax.plot(eval_df["ds"], eval_df["yhat"], label="Latest forecast (yhat)", linewidth=1)

    ax.set_title(f"{region} — 2024 Actual vs Forecast (retraining: {retrain_frequency})")
    ax.set_xlabel("Datetime (UTC)")
    ax.set_ylabel("gCO₂eq/kWh")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    tag = retrain_frequency.lower()
    out_png = OUTPUT_DIR / f"{region}_rolling_{tag}_forecast_2024.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    # plt.show()  # uncomment if running interactively


# ===================== Entry Point =====================

if __name__ == "__main__":
    evaluate_and_plot(REGION, RETRAIN_FREQUENCY)
