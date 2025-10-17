"""Demonstrations for the :mod:`src.scenario` module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import scenario
from src import util as util_module
from src.util import DT_INDEX


def demonstrate_load_requests(dataset_name: str) -> None:
    """Show how to load and scale a request time series."""

    # Load the synthetic request trace created in ``main``.
    scaled, raw, scaling_factor = scenario.load_requests(dataset_name)
    print("Scaled request matrix shape:", scaled.shape)
    print("First three scaled rows:\n", scaled[:3])
    print("Scaling factor applied to requests:", scaling_factor)
    print("Raw frame columns:", raw.columns.tolist())


def demonstrate_zone_resolution(region: str) -> None:
    """Resolve the Electricity Maps zone name and preview carbon data."""

    # ``_resolve_electricitymaps_zone`` falls back to scanning the data folder
    # when the region name is an alias.
    zone = scenario._resolve_electricitymaps_zone(region, "train")
    print("Resolved zone identifier:", zone)

    # ``load_carbon_intensity`` returns scaled carbon intensities together with
    # the aligned raw dataframe.
    carbon_series, raw = scenario.load_carbon_intensity(region)
    print("Carbon intensity array length:", len(carbon_series))
    print("Sample carbon intensity values:", carbon_series[:3])
    print("Raw dataframe columns:", raw.columns.tolist())


def demonstrate_machine() -> None:
    """Instantiate a :class:`scenario.Machine` and inspect its accessors."""

    quality_to_index = {f"tier_{idx}": idx for idx in range(5)}
    scaling_factor = 1 / 50_000

    # ``power_usage`` only describes a constant draw, mirroring the production config.
    machine = scenario.Machine(
        name="AWS_p4d.24xlarge",
        performance={
            "tier_0": 42_000,
            "tier_1": 32_000,
            "tier_2": 25_000,
            "tier_3": 20_000,
            "tier_4": 15_000,
        },
        embedded_carbon=135,
        request_scaling_factor=scaling_factor,
        power_usage=3.7818,
        quality_to_index=quality_to_index,
        quality_count=len(quality_to_index),
    )
    print("Constant power draw machine:")
    print("  Normalized performance vector:", machine.performance)
    print("  Embedded carbon (tCO2eq):", machine.embedded_carbon)
    print("  Load-independent power usage:", machine.load_independent_power_usage())

    # Dynamic models can add ``idle``/``max`` definitions per tier for heterogeneous fleets.
    dynamic_machine = scenario.Machine(
        name="lab-prototype",
        performance={
            "tier_0": 10_000,
            "tier_1": 7_500,
            "tier_2": 5_000,
            "tier_3": 3_500,
            "tier_4": 2_500,
        },
        embedded_carbon=90,
        request_scaling_factor=scaling_factor,
        idle_power_usage=0.8,
        max_power_usage=[1.6, 1.5, 1.4, 1.2, 1.0],
        pue=1,
        quality_to_index=quality_to_index,
        quality_count=len(quality_to_index),
    )
    print("Dynamic power draw machine:")
    print("  Normalized performance vector:", dynamic_machine.performance)
    print(
        "  Load-dependent power at 60% utilization (tier_2):",
        dynamic_machine.load_dependent_power_usage(2, util=0.6, n=1.5),
    )


def demonstrate_scenario(dataset_name: str, region: str) -> None:
    """Create a :class:`scenario.Scenario` and produce short-term forecasts."""

    # Define a simple user-group configuration with lower and upper QoR bounds.
    user_groups = [
        {
            "name": "best-effort",
            "weight": 1.0,
            "slo_lower": {
                "tier_0": 1.0,
                "tier_1": 0.75,
                "tier_2": 0.5,
                "tier_3": 0.25,
                "tier_4": 0.0,
            },
            "slo_upper": {
                "tier_0": 0.0,
                "tier_1": 0.25,
                "tier_2": 0.5,
                "tier_3": 0.75,
                "tier_4": 1.0,
            },
        }
    ]

    # Machine definitions mirror the Hydra config structure used by the project.
    machines = [
        {
            "_target_": "src.scenario.Machine",
            "name": "AWS_p4d.24xlarge",
            "power_usage": 3.7818,
            "pue": 1,
            "performance": {
                "tier_0": 42_000,
                "tier_1": 32_000,
                "tier_2": 25_000,
                "tier_3": 20_000,
                "tier_4": 15_000,
            },
            "embedded_carbon": 135,
        },
        {
            "_target_": "src.scenario.Machine",
            "name": "AWS_p5.48xlarge",
            "power_usage": 4.4,
            "pue": 1,
            "performance": {
                "tier_0": 52_000,
                "tier_1": 41_000,
                "tier_2": 33_000,
                "tier_3": 27_000,
                "tier_4": 21_000,
            },
            "embedded_carbon": 150,
        }
    ]

    sc = scenario.Scenario(
        seed=123,
        requests_dataset=dataset_name,
        region=region,
        vp=24,
        user_groups_scenario="demo",
        user_groups=user_groups,
        model_qualities=["tier_0", "tier_1", "tier_2", "tier_3", "tier_4"],
        machines=machines,
    )
    print("Scenario identifier:", sc.name)
    print("Machine throughput matrix K shape:", sc.K.shape)
    print("Machine throughput matrix K sample:\n", sc.K[:, :2])

    # Produce forecasts starting at t=12 to illustrate both helpers.
    demand_forecast = sc.generate_R_hat(12, "yhat")
    carbon_forecast = sc.generate_C_hat(12, "yhat")
    print("Forecast demand sample (rows 12-14):\n", demand_forecast[12:15])
    print("Forecast carbon sample (rows 12-14):\n", carbon_forecast[12:15])


def main() -> None:
    """Generate synthetic data and run all demonstrations."""

    # Temporary directory keeps the example self-contained and tidy.
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = Path(tmp_dir)
        request_dir = data_root / "request_traces"
        electricity_dir = data_root / "electricitymaps" / "train"
        request_dir.mkdir(parents=True)
        electricity_dir.mkdir(parents=True)

        dataset_name = "example_trace"
        region = "EXAMPLE-REGION"

        # Create a short synthetic request trace to feed into ``load_requests``.
        request_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=len(DT_INDEX), freq="h"),
                "requests": np.linspace(50_000, 60_000, len(DT_INDEX)),
            }
        )
        request_df.to_csv(request_dir / f"{dataset_name}.csv", index=False)

        # Build a matching Electricity Maps style CSV file for ``load_carbon_intensity``.
        carbon_df = pd.DataFrame(
            {
                "Datetime (UTC)": DT_INDEX,
                "Carbon intensity gCO₂eq/kWh (Life cycle)": np.linspace(400, 600, len(DT_INDEX)),
            }
        )
        electricity_file = electricity_dir / f"{region}_2023_hourly.csv"
        carbon_df.to_csv(electricity_file, index=False)

        # Patch module-level paths so the scenario helpers load the synthetic files.
        original_data_dir = scenario._DATA_DIR
        original_em_dir = util_module._ELECTRICITYMAPS_DIR
        scenario._DATA_DIR = str(data_root)
        util_module._ELECTRICITYMAPS_DIR = electricity_dir.parent

        # Patch Hydra's instantiate helper to bypass actual Hydra integration.
        original_instantiate = scenario.hydra.utils.instantiate

        def instantiate(cfg, **kwargs):
            cfg = dict(cfg)
            cfg.pop("_target_", None)
            return scenario.Machine(**cfg, **kwargs)

        scenario.hydra.utils.instantiate = instantiate

        # Provide deterministic forecast helpers so the output remains stable.
        original_prophet = scenario.load_prophet_forecast
        original_bootstrap = scenario.generate_bootstrapped_forecast

        def fake_prophet(raw, start, cache_key=None, forecast_params=None):
            horizon = len(DT_INDEX) - start
            values = np.linspace(0.9, 1.1, horizon)
            return pd.DataFrame(
                {
                    "yhat": values,
                    "yhat_lower": values * 0.95,
                    "yhat_upper": values * 1.05,
                }
            )

        def fake_bootstrap(forecast, historical_errors, rng):
            return forecast

        scenario.load_prophet_forecast = fake_prophet
        scenario.generate_bootstrapped_forecast = fake_bootstrap

        try:
            demonstrate_load_requests(dataset_name)
            demonstrate_zone_resolution(region)
            demonstrate_machine()
            demonstrate_scenario(dataset_name, region)
        finally:
            # Restore patched globals before leaving the context manager.
            scenario._DATA_DIR = original_data_dir
            util_module._ELECTRICITYMAPS_DIR = original_em_dir
            scenario.hydra.utils.instantiate = original_instantiate
            scenario.load_prophet_forecast = original_prophet
            scenario.generate_bootstrapped_forecast = original_bootstrap


if __name__ == "__main__":
    main()