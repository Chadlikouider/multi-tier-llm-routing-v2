#!/usr/bin/env python3
"""
Example test script to demonstrate input/output for functions and classes in the scenario definitions module.
This follows the structure of the provided demonstration, using synthetic data and patching for self-containment.
It shows how to run the code as tests, checking inputs and outputs for each function/class in the module.

Assumes the module is installed or available as `src.scenario` (adjust import as needed).
For real tests, consider using pytest with fixtures for better isolation and assertions.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.scenario import Scenario, Machine, load_requests, load_carbon_intensity, _resolve_electricitymaps_zone
from src.util import DT_INDEX


def test_load_requests(dataset_name: str) -> None:
    """Show how to load and scale a request time series."""
    print("=== Testing load_requests ===")
    print(f"Input: dataset_name='{dataset_name}'")
    
    scaled, raw, scaling_factor = load_requests(dataset_name)
    print(f"Output: scaled shape={scaled.shape}, raw shape={raw.shape}, scaling_factor={scaling_factor}")
    print(f"Sample scaled values: {scaled[:3]}")
    print(f"Raw columns: {raw.columns.tolist()}")


def test_resolve_zone(region: str) -> None:
    """Resolve the Electricity Maps zone name."""
    print("=== Testing _resolve_electricitymaps_zone ===")
    print(f"Input: region='{region}', split='train'")
    
    zone = _resolve_electricitymaps_zone(region, "train")
    print(f"Output: resolved_zone='{zone}'")


def test_load_carbon_intensity(region: str) -> None:
    """Load carbon intensity data."""
    print("=== Testing load_carbon_intensity ===")
    print(f"Input: region='{region}'")
    
    carbon_series, raw = load_carbon_intensity(region)
    print(f"Output: carbon_series shape={carbon_series.shape}, raw shape={raw.shape}")
    print(f"Sample carbon values: {carbon_series[:3]}")
    print(f"Raw columns: {raw.columns.tolist()}")


def test_machine() -> None:
    """Instantiate and test a Machine."""
    print("=== Testing Machine class ===")
    
    quality_to_index = {f"tier_{idx}": idx for idx in range(5)}
    scaling_factor = 1 / 50_000
    
    # Test constant power machine
    machine = Machine(
        name="TestConstant",
        performance={f"tier_{i}": 50000 - i * 10000 for i in range(5)},
        embedded_carbon=100,
        request_scaling_factor=scaling_factor,
        power_usage=3.0,
        quality_to_index=quality_to_index,
        quality_count=5,
    )
    print(f"Input: name, performance dict, embedded_carbon=100, etc.")
    print(f"Output: normalized_performance={machine.performance}, embedded_carbon={machine.embedded_carbon}")
    print(f"Load-independent power: {machine.load_independent_power_usage()}")
    
    # Test dynamic power machine
    dynamic_machine = Machine(
        name="TestDynamic",
        performance={f"tier_{i}": 30000 - i * 5000 for i in range(5)},
        embedded_carbon=80,
        request_scaling_factor=scaling_factor,
        idle_power_usage=1.0,
        max_power_usage=[2.0 + i * 0.1 for i in range(5)],
        pue=1.1,
        quality_to_index=quality_to_index,
        quality_count=5,
    )
    print(f"Load-dependent power (q=2, util=0.7, n=2): {dynamic_machine.load_dependent_power_usage(2, 0.7, 2)}")


def test_scenario(dataset_name: str, region: str) -> None:
    """Create and test a Scenario."""
    print("=== Testing Scenario class ===")
    print(f"Input: seed=42, requests_dataset='{dataset_name}', region='{region}', vp=1, etc.")
    
    user_groups = [
        {
            "name": "best-effort",
            "weight": 0.3,
            "slo_lower": {f"tier_{i}": max(0, 0.2 * (4 - i)) for i in range(5)},
            "slo_upper": {f"tier_{i}": min(1.0, 0.2 * (i + 1)) for i in range(5)},
        },
        {
            "name": "premium",
            "weight": 0.7,
            "slo_lower": {f"tier_{i}": max(0.5, 0.1 * (4 - i + 1)) for i in range(5)},
            "slo_upper": {f"tier_{i}": 0.1 * (i + 6) for i in range(5)},
        }
    ]
    
    machines = [
        {
            "_target_": "src.scenario.Machine",
            "name": "TestMachine1",
            "performance": {f"tier_{i}": 40000 - i * 8000 for i in range(5)},
            "embedded_carbon": 130,
            "power_usage": 3.5,
        },
        {
            "_target_": "src.scenario.Machine",
            "_partial_": False,  # Ensure full instantiation
            "name": "TestMachine2",
            "performance": {f"tier_{i}": 50000 - i * 10000 for i in range(5)},
            "embedded_carbon": 140,
            "power_usage": 4.0,
        }
    ]
    
    sc = Scenario(
        seed=42,
        requests_dataset=dataset_name,
        region=region,
        vp=1,
        user_groups_scenario="test",
        user_groups=user_groups,
        model_qualities=[f"tier_{i}" for i in range(5)],
        machines=machines,
    )
    
    print(f"Output: name='{sc.name}', U={sc.U}, Q={sc.Q}, M={sc.M}")
    print(f"K shape={sc.K.shape}, sample K=\n{sc.K}")
    
    # Test forecast methods
    R_hat = sc.generate_R_hat(10, "oracle")
    print(f"generate_R_hat('oracle', i=10) shape={R_hat.shape}, sample={R_hat[10:13]}")
    
    C_hat = sc.generate_C_hat(10, "yhat")
    print(f"generate_C_hat('yhat', i=10) shape={C_hat.shape}, sample={C_hat[10:13]}")


def main() -> None:
    """Set up synthetic data and run all tests."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = Path(tmp_dir)
        request_dir = data_root / "request_traces"
        electricity_dir = data_root / "electricitymaps" / "train"
        request_dir.mkdir(parents=True)
        electricity_dir.mkdir(parents=True)
        
        dataset_name = "test_trace"
        region = "TEST-REGION"
        
        # Create synthetic request data
        request_df = pd.DataFrame({
            "timestamp": DT_INDEX,
            "requests": np.sin(np.linspace(0, 2*np.pi, len(DT_INDEX))) * 10000 + 50000,
        })
        request_df.to_csv(request_dir / f"{dataset_name}.csv", index=False)
        
        # Create synthetic carbon data
        carbon_df = pd.DataFrame({
            "Datetime (UTC)": DT_INDEX,
            "Carbon intensity gCO₂eq/kWh (Life cycle)": np.cos(np.linspace(0, 4*np.pi, len(DT_INDEX))) * 100 + 500,
        })
        electricity_file = electricity_dir / f"{region}_2023_hourly.csv"
        carbon_df.to_csv(electricity_file, index=False)
        
        # Patch module globals
        from src import scenario as scenario_module
        from src import util as util_module
        original_data_dir = scenario_module._DATA_DIR
        original_em_dir = util_module._ELECTRICITYMAPS_DIR
        scenario_module._DATA_DIR = str(data_root)
        util_module._ELECTRICITYMAPS_DIR = electricity_dir.parent
        
        # Mock Hydra's instantiate function
        original_instantiate = scenario_module.hydra.utils.instantiate
        
        def fake_instantiate(cfg, **kwargs):
            cfg = {k: v for k, v in cfg.items() if k not in ["_target_", "_partial_"]}
            return Machine(**cfg, **kwargs)
        
        scenario_module.hydra.utils.instantiate = fake_instantiate
        
        # Mock prophet and bootstrap for stability
        original_prophet = getattr(scenario_module, 'load_prophet_forecast', None)
        original_bootstrap = getattr(scenario_module, 'generate_bootstrapped_forecast', None)
        
        def fake_prophet(raw, start, **kwargs):
            horizon = len(raw) - start
            values = np.random.normal(1.0, 0.1, horizon)
            return pd.DataFrame({
                "yhat": values,
                "yhat_lower": values * 0.9,
                "yhat_upper": values * 1.1,
            })
        
        def fake_bootstrap(forecast, **kwargs):
            return forecast * 1.05  # Slight adjustment for demo
        
        if original_prophet:
            scenario_module.load_prophet_forecast = fake_prophet
        if original_bootstrap:
            scenario_module.generate_bootstrapped_forecast = fake_bootstrap
        
        try:
            test_load_requests(dataset_name)
            test_resolve_zone(region)
            test_load_carbon_intensity(region)
            test_machine()
            test_scenario(dataset_name, region)
            print("\n=== All tests completed ===")
        finally:
            # Restore patches
            scenario_module._DATA_DIR = original_data_dir
            util_module._ELECTRICITYMAPS_DIR = original_em_dir
            scenario_module.hydra.utils.instantiate = original_instantiate
            if original_prophet:
                scenario_module.load_prophet_forecast = original_prophet
            if original_bootstrap:
                scenario_module.generate_bootstrapped_forecast = original_bootstrap


if __name__ == "__main__":
    main()