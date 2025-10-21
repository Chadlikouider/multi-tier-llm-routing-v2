from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.scenario import Scenario


def load_configuration() -> DictConfig:
    """Load the Hydra configuration used by the project."""
    # Resolve the repository root relative to this file so the example works
    # regardless of the current working directory.
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"

    # Hydra composes configuration files from the specified directory. The
    # context manager ensures the global Hydra state is initialised properly.
    with initialize_config_dir(config_dir=str(config_dir), job_name="scenario_example", version_base=None):
        cfg = compose(config_name="config")

    return cfg


def main() -> None:
    """Entry point that loads the config and runs the scenario."""
    # First load the configuration before running the scenario logic.
    cfg = load_configuration()

    # Use the convenience constructor to build a Scenario instance from the
    # Hydra configuration. This triggers loading request traces, carbon
    # intensity data, and instantiation of machine definitions.
    scenario = Scenario.from_config(cfg)

    # Print a few key attributes so users can verify the scenario ran
    # successfully.
    print("Scenario name:", scenario.name)
    print("Number of machines:", len(scenario.machines))
    print("Demand matrix shape:", scenario.R.shape)
    print("Carbon intensity vector length:", len(scenario.C))

    # Generate forecasts for requests and carbon intensity starting from the
    # beginning (interval i=0). This will load from cache or generate new
    # Prophet models as needed via load_prophet_forecast in forecasting.py.
    print("Generating/Loading request forecast (oracle)...")
    R_forecast_oracle = scenario.generate_R_hat(i=0, kind="oracle")
    print("Request forecast (oracle) shape:", R_forecast_oracle.shape)

    print("Generating/Loading carbon intensity forecast (oracle)...")
    C_forecast_oracle = scenario.generate_C_hat(i=0, kind="oracle")
    print("Carbon forecast (oracle) length:", len(C_forecast_oracle))

    # Example for predicted forecasts (using yhat for demonstration).
    # For real prediction horizons, call with i set to a specific start interval
    # and kind="yhat", "yhat_lower", or "yhat_upper". Using "yhat" as the kind
    # will load/generate the Prophet model and apply caching with the updated key.
    print("Generating/Loading request forecast (predicted, yhat)...")
    try:
        R_forecast_pred = scenario.generate_R_hat(i=0, kind="yhat")  # Changed to a safe i=0 to avoid issues with model fitting on insufficient data
        print("Request forecast (predicted, yhat) shape:", R_forecast_pred.shape)
    except (IndexError, ValueError) as e:
        print(f"Request forecast (predicted): Error - {e}. Check data length or valid i values.")

    print("Generating/Loading carbon intensity forecast (predicted, yhat)...")
    try:
        C_forecast_pred = scenario.generate_C_hat(i=0, kind="yhat")  # Changed to a safe i=0 to avoid issues with model fitting on insufficient data
        print("Carbon forecast (predicted, yhat) length:", len(C_forecast_pred))
    except (IndexError, ValueError) as e:
        print(f"Carbon forecast (predicted): Error - {e}. Check data length or valid i values.")

    # Optionally, you could save the forecasts to files or integrate into further processing,
    # e.g., optimization. Forecasts are numpy arrays, so they can be used directly:
    # import numpy as np
    # np.save("R_forecast.npy", R_forecast_oracle)
    # np.save("C_forecast.npy", C_forecast_oracle)


if __name__ == "__main__":
    main()