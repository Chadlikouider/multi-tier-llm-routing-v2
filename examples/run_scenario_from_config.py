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


if __name__ == "__main__":
    main()