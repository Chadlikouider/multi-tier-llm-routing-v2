"""Scenario definitions and data loading helpers.

This module contains lightweight container classes that bundle together the
various data inputs used by the optimisation layer of the project.  The goal of
the refactor is to make the construction of scenarios easier to follow and to
document the data manipulation that happens along the way so future changes can
be made with confidence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, List, Any

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize

from src.forecasting import generate_bootstrapped_forecast, load_prophet_forecast
from src.util import DT_INDEX

# ``_DATA_DIR`` is resolved relative to the repository so the module can be
# imported from both application code and tests without having to rely on the
# current working directory.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


class Scenario:
    """Container describing a full optimisation scenario.

    Encapsulates all parameters, data, and metadata for simulation, including
    request forecasts, carbon intensity, user groups, and machine configurations.
    Data is loaded, normalized for numerical stability, and materialized during
    initialization, allowing reproducible optimization runs.
    """

    def __init__(
        self,
        seed: int,
        requests_dataset: str,
        region: str,
        vp: int,
        user_groups_scenario: str,
        user_groups: List[Dict],
        model_qualities: List[str],
        machines: List[Dict],
    ) -> None:
        # Validate essential inputs to prevent silent failures
        if not user_groups:
            raise ValueError("User groups list cannot be empty.")
        if not model_qualities:
            raise ValueError("Model qualities list cannot be empty.")
        if not machines:
            raise ValueError("Machines list cannot be empty.")

        # Random numbers are used for synthetic carbon intensity data. We keep
        # a dedicated generator so repeated runs with the same configuration are
        # reproducible.
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Configuration identifiers used when serializing results.
        self.requests_dataset = requests_dataset
        self.region = region
        self.user_groups_scenario = user_groups_scenario
        self.vp = vp

        # User group metadata is reshaped into NumPy-friendly structures. The
        # weights are used to scale demand data per group, while the QoS targets
        # are mapped to matrices for fast lookups during optimization.
        self.user_group_weights = [u["weight"] for u in user_groups]
        self.model_qualities = model_qualities
        self.quality_to_index = {quality: idx for idx, quality in enumerate(model_qualities)}

        self.slo_lower = self._build_slo_matrix(user_groups, "slo_lower")
        self.slo_upper = self._build_slo_matrix(user_groups, "slo_upper")

        # Load the time series backing the scenario. 'request_scaling_factor'
        # keeps numerical values in a narrow range to improve solver stability
        # and is also passed to machine definitions so their throughput is
        # scaled consistently.
        self.C, self.C_raw = load_carbon_intensity(region)

        self.R, self.R_raw, request_scaling_factor = load_requests(
            requests_dataset, weights=self.user_group_weights
        )
        self.request_scaling_factor = request_scaling_factor

        self.machines = self._instantiate_machines(machines, request_scaling_factor)

        # Convenience sets used by the optimization models to iterate over the
        # scenario entities.
        self.U = list(range(len(user_groups)))
        self.Q = list(range(len(model_qualities)))
        self.M = list(range(len(machines)))
        self.I = list(range(len(DT_INDEX)))  # Set of intervals

    @property
    def name(self) -> str:
        """Return a descriptive identifier used in file names and logs."""
        return f"{self.requests_dataset},{self.region},{self.user_groups_scenario},vp={self.vp}"

    def _instantiate_machines(
        self, machines: Iterable[Dict], request_scaling_factor: float
    ) -> Dict[int, "Machine"]:
        """Instantiate machines defined in the configuration.

        Hydra allows us to specify different machine classes in configuration
        files; we pass the bookkeeping parameters required by :class:`Machine`
        so that custom classes can consume them as well.

        Parameters
        ----------
        machines : Iterable[Dict]
            Configuration dictionaries for machines.
        request_scaling_factor : float
            Scaling factor for throughput normalization.
        """
        if not machines:
            raise ValueError("Machines iterable cannot be empty.")
        return {
            i: hydra.utils.instantiate(
                machine_cfg,
                request_scaling_factor=request_scaling_factor,
                quality_to_index=self.quality_to_index,
                quality_count=len(self.model_qualities),
            )
            for i, machine_cfg in enumerate(machines)
        }

    @classmethod
    def from_config(cls, cfg: Any):
        """Create a scenario directly from a Hydra configuration object.

        Parameters
        ----------
        cfg : Any
            Hydra config object containing scenario parameters.
        """
        return cls(
            seed=cfg.seed,
            requests_dataset=cfg.requests_dataset,
            region=cfg.region,
            vp=cfg.vp,
            user_groups_scenario=cfg.user_groups["name"],
            user_groups=cfg.user_groups["groups"],
            model_qualities=cfg.model_qualities,
            machines=cfg.machines,
        )

    @classmethod
    def from_name(cls, name: str):
        """Load a scenario by parsing the string returned from :attr:`name`.

        Parameters
        ----------
        name : str
            Serialized scenario name (e.g., "dataset,region,scenario,vp=1").
        """
        parts = name.split(",")
        if len(parts) != 4:
            raise ValueError(f"Invalid scenario name format: '{name}'. Expected 'dataset,region,scenario,vp=X'.")
        requests_dataset, region, user_group_scenario, vp_str = parts
        if "=" not in vp_str:
            raise ValueError(f"Invalid vp format in scenario name '{name}': missing '='.")
        vp = int(vp_str.split("=")[1])  # Raises ValueError if not an int
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"requests_dataset={requests_dataset}",
                    f"region={region}",
                    f"user_groups={user_group_scenario}",
                    f"vp={vp}",
                ],
            )
        return cls.from_config(cfg)

    @property
    def K(self) -> np.ndarray:
        """Return a matrix of machine throughput indexed by quality and machine."""
        return np.array([[self.machines[m].performance[q] for m in self.M] for q in self.Q])

    def _build_slo_matrix(self, user_groups: List[Dict], key: str) -> np.ndarray:
        """Convert QoS constraints per user group into a matrix.

        Parameters
        ----------
        user_groups : List[Dict]
            Raw configuration objects containing QoS definitions.
        key : str
            Either "slo_lower" or "slo_upper" specifying which bounds to extract.
        """
        matrix = np.zeros((len(user_groups), len(self.model_qualities)))
        expected_keys = set(self.model_qualities)
        for user_idx, user_group in enumerate(user_groups):
            slo_values = user_group.get(key, {})
            missing = expected_keys - set(slo_values)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"User group '{user_group.get('name', user_idx)}' is missing QoS targets for: {missing_str}"
                )
            for quality, value in slo_values.items():
                if quality not in self.quality_to_index:
                    raise ValueError(
                        f"Unknown model quality '{quality}' referenced in {key} for user group "
                        f"'{user_group.get('name', user_idx)}'"
                    )
                matrix[user_idx, self.quality_to_index[quality]] = value
        return matrix

    def generate_R_hat(self, i: int, kind: Literal["oracle", "yhat", "yhat_lower", "yhat_upper"]) -> np.array:
        # TODO implement multiple users
        R = np.copy(self.R)
        if kind == "oracle":
            return R
        if "static" in self.requests_dataset or "normal" in self.requests_dataset:
            R.fill(1)
            return R

        mean_of_previous_years = self.R_raw["y"][:-8760].mean()
        floor = mean_of_previous_years * 0.9
        cap = mean_of_previous_years * 1.1

        fc = load_prophet_forecast(self.R_raw, i, cache_key=f"R_hat_{self.requests_dataset}_{i}", forecast_params=dict(floor=floor, cap=cap))
        fc["yhat"] = fc["yhat"] * self.request_scaling_factor
        fc["yhat_lower"] = fc["yhat_lower"] * self.request_scaling_factor
        fc["yhat_upper"] = fc["yhat_upper"] * self.request_scaling_factor

        if self.requests_dataset == "wiki_de":
            floor_hard_cap = np.quantile(self.R_raw["y"][:-8760], 0.01) * self.request_scaling_factor
            fc.loc[fc["yhat"] < floor_hard_cap, "yhat"] = floor_hard_cap
            fc.loc[fc["yhat_lower"] < floor_hard_cap, "yhat_lower"] = floor_hard_cap
            fc.loc[fc["yhat_upper"] < floor_hard_cap, "yhat_upper"] = floor_hard_cap

        R[i:] = np.expand_dims(fc[kind].values, axis=1)

        # TODO implement multiple users
        if self.user_group_weights:
            R = np.hstack([R * weight for weight in self.user_group_weights])
        return R

    def generate_C_hat(self, i: int, kind: Literal["oracle", "yhat"]) -> np.ndarray:
        """Return a carbon intensity outlook starting at step 'i'.

        Parameters
        ----------
        i : int
            Starting interval index.
        kind : Literal["oracle", "yhat"]
            Type of forecast (oracle or prediction).
        """
        C = np.copy(self.C)
        if kind == "oracle":
            
            return C

        
        fc = load_prophet_forecast(
            self.C_raw, i, cache_key=f"C_hat_{self.region}_{i}", forecast_params=dict(flat=True)
        )
        
        # print("shape of fc[yhat]:", fc[kind])
        C[i:] = fc[kind] / 1_000_000 # Converting units
        
        max_index = min(i + 1, len(DT_INDEX))
        return self._apply_bootstrapped_forecast(C, fc, i, max_index, kind)

    def _apply_bootstrapped_forecast(
        self, C: np.ndarray, fc: Any, i: int, max_index: int, kind: str
    ) -> np.ndarray:
        """Helper to compute and apply bootstrapped forecast adjustments."""
        # Calculate historical errors
        history_end = max(0, i)
        history_forecast = fc[kind][:history_end] / 1000000
        history_truth = self.C[:history_end]
        denom = np.where(np.abs(history_forecast) > 1e-9, history_forecast, 1e-9)
        historical_errors = np.where(
            denom != 0, (history_truth - history_forecast) / denom, 0
        )
        historical_errors = historical_errors[np.isfinite(historical_errors)]

        # Apply bootstrapping to near-term forecast
        C[i:max_index] = generate_bootstrapped_forecast(
            forecast=C[i:max_index], historical_errors=historical_errors, rng=self.rng
        )
        return C


class Machine:
    """A model of a deployable hardware configuration, including performance, power usage, and carbon impact."""

    def __init__(
        self,
        name: str,
        performance: dict[str, float],
        embedded_carbon: float,
        request_scaling_factor: float,  # Numerical stability
        power_usage: Optional[float] = None,
        idle_power_usage: Optional[float] = None,
        max_power_usage: Optional[list[float]] = None,
        pue: float = 1.0,  # Power Usage Effectiveness, typically >1.0
        quality_to_index: Optional[dict[str, int]] = None,
        quality_count: Optional[int] = None,
    ) -> None:
        # Validate consistency: if quality_to_index is provided, quality_count must be too, and max_power_usage length must match.
        if quality_to_index is not None:
            if quality_count is None:
                raise ValueError(f"Machine '{name}': 'quality_count' must be provided when 'quality_to_index' is used.")
            if max_power_usage is not None and len(max_power_usage) != quality_count:
                raise ValueError(f"Machine '{name}': 'max_power_usage' length ({len(max_power_usage)}) must match 'quality_count' ({quality_count}).")

        # Throughput is normalized so all optimization inputs are scaled the same way.
        # When 'quality_to_index' is provided, we produce a list that can be indexed directly by a quality integer.
        # Added consistency checks in _normalize_performance to ensure valid quality mappings.
        self.name = name
        self.performance = self._normalize_performance(
            name,
            performance,
            request_scaling_factor,
            quality_to_index,
            quality_count,
        )
        # Convert to tonnes of CO₂ equivalent to align with other parts of the model which operate on that unit.
        self.embedded_carbon = embedded_carbon #/ 1000000  # gCO₂eq to tCO₂eq

        # Power models can either be load-independent or load-dependent; both
        # representations are supported simultaneously and consumers can choose
        # which accessors to use.
        self._power_usage = power_usage
        self._idle_power_usage = idle_power_usage
        self._max_power_usage = max_power_usage
        self._pue = pue

    def load_independent_power_usage(self) -> float:
        """Return the constant power draw when the simple model is used."""
        return self._power_usage

    def load_dependent_power_usage(self, q, util: float, n: float = 1) -> float:
        """Return the dynamic power draw for a given utilisation.

        Parameters
        ----------
        q:
            Index of the model quality to look up in ``max_power_usage``.
        util:
            Current utilisation ratio, typically between 0 and 1.
        n:
            Exponent applied to the utilisation value. A value of ``1`` keeps
            the relationship linear while larger values bias the curve towards
            high utilisation.
        """

        return self._pue * (self._idle_power_usage + (self._max_power_usage[q] - self._idle_power_usage) * util ** n)

    @staticmethod
    def _normalize_performance(
        machine_name: str,
        performance: dict[str, float],
        request_scaling_factor: float,
        quality_to_index: Optional[dict[str, int]],
        quality_count: Optional[int]
    ) -> list[float]:
        """Map performance dictionaries to consistent list representations.

        This ensures compatibility with indexing by quality integer (when applicable)
        and validates that specified qualities exist in the provided mapping.
        """
        if quality_to_index is None or quality_count is None:
            return [v * request_scaling_factor for v in performance.values()]

        normalized = [0.0] * quality_count
        for quality, throughput in performance.items():
            if quality not in quality_to_index:
                raise ValueError(
                    f"Performance specified for unknown quality '{quality}' on machine '{machine_name}'. "
                    f"Ensure '{quality}' is a key in the provided 'quality_to_index' mapping."
                )
            normalized[quality_to_index[quality]] = throughput * request_scaling_factor
        return normalized


def load_requests(
    dataset: str,
    weights: Optional[list[float]] = None,
    split: str = "train"  # kept for compatibility, ignored for 2024
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """
    Load request traces with train/test split logic:
      - 2022 & 2023 → request_traces/train/
      - 2024       → request_traces/test/

    Parameters
    ----------
    dataset : str
        Base name without year, e.g. "normal_requests", "static_requests", "synthetic_requests"
    weights : list[float] | None
        If given, duplicate the 2024 trace scaled by each weight.
    split : str
        Ignored (kept for API compatibility). 2024 always comes from 'test'.

    Returns
    -------
    R : np.ndarray
        Scaled 2024 trace(s), shape (8784, n_users) if weights given.
    R_raw : pd.DataFrame
        Full 2022-2023-2024 series with columns `ds` and `y`.
    request_scaling_factor : float
        1 / mean(y) over full 2022-2024.
    """
    base_dir = Path(_DATA_DIR) / "request_traces"
    train_dir = base_dir / "train"
    test_dir  = base_dir / "test"

    # ------------------------------------------------------------------ #
    # 1. Load 2022 & 2023 from train/
    # ------------------------------------------------------------------ #
    dfs = []
    for year in [2022, 2023]:
        path = train_dir / f"{dataset}_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing train file: {path}")
        df = pd.read_csv(path, parse_dates=True)
        dfs.append(df)

    # ------------------------------------------------------------------ #
    # 2. Append 2024 from test/
    # ------------------------------------------------------------------ #
    path_2024 = test_dir / f"{dataset}_2024.csv"
    if not path_2024.is_file():
        raise FileNotFoundError(f"Missing test file (2024): {path_2024}")
    dfs.append(pd.read_csv(path_2024, parse_dates=True))

    # ------------------------------------------------------------------ #
    # 3. Concatenate → R_raw
    # ------------------------------------------------------------------ #
    R_raw = pd.concat(dfs, ignore_index=True)
    R_raw = R_raw.rename(
            columns={"timestamp": "ds",
                    "requests": "y",
            }
        )  # ensure column name
    if "y" not in R_raw.columns:
        raise ValueError("CSV must contain a 'y' column with request counts")

    # ------------------------------------------------------------------ #
    # 4. Scaling factor over FULL 2022-2024
    # ------------------------------------------------------------------ #
    request_scaling_factor = 1.0 / R_raw["y"].mean()

    # ------------------------------------------------------------------ #
    # 5. Slice last 8784 rows (2024) → R
    # ------------------------------------------------------------------ #
    # y_2024 = R_raw["y"].values[-len(DT_INDEX):] * request_scaling_factor
    # R = np.expand_dims(y_2024, axis=1)  # shape (8784, 1)
    R = np.expand_dims(R_raw["y"].values, axis=1)[-len(DT_INDEX):] * request_scaling_factor
    # ------------------------------------------------------------------ #
    # 6. Apply user weights (multiple traces)
    # ------------------------------------------------------------------ #
    if weights:
        R = np.hstack([R * w for w in weights])

    return R, R_raw, request_scaling_factor


def load_carbon_intensity(region: str):
    """
    Load carbon-intensity data for a region.
    - 2022 & 2023 are always taken from the *train* directory
    - 2024 is always taken from the *test* directory
    - The function works regardless of the `split` argument.

    Parameters
    ----------
    region : str
        e.g. "DE", "US-NY-NYISO"
    split : str, default "train"
        Kept for API compatibility – only the *test* folder is used for 2024.

    Returns
    -------
    C : np.ndarray
        2024 carbon intensity (8 784 values) in tCO₂eq/kWh.
    raw : pd.DataFrame
        Full 2022-2024 series with columns `ds` (UTC) and `y` (gCO₂eq/kWh, f-filled).
    """
    base_dir = Path(_DATA_DIR) / "electricitymaps"

    # ------------------------------------------------------------------ #
    # 1. Load 2022 & 2023 from *train*
    # ------------------------------------------------------------------ #
    train_dir = base_dir / "train"
    dfs = []
    for year in [2022, 2023]:
        p = train_dir / f"{region}_{year}_hourly.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing train file: {p}")
        dfs.append(
            pd.read_csv(p, index_col=0, parse_dates=True)
        )

    # ------------------------------------------------------------------ #
    # 2. Append 2024 from *test*
    # ------------------------------------------------------------------ #
    test_dir = base_dir / "test"
    p2024 = test_dir / f"{region}_2024_hourly.csv"
    if not p2024.is_file():
        raise FileNotFoundError(f"Missing test file (2024): {p2024}")

    dfs.append(pd.read_csv(p2024, index_col=0, parse_dates=True))

    # ------------------------------------------------------------------ #
    # 3. Concatenate everything
    # ------------------------------------------------------------------ #
    ci = pd.concat(dfs, axis=0)

    # ------------------------------------------------------------------ #
    # 4. Build `raw` (full series)
    # ------------------------------------------------------------------ #
    raw = ci["Carbon intensity gCO₂eq/kWh (Life cycle)"].reset_index()
    raw = raw.rename(
        columns={
            "Datetime (UTC)": "ds",
            "Carbon intensity gCO₂eq/kWh (Life cycle)": "y",
        }
    )
    raw["y"] = raw["y"].ffill()          # forward-fill missing values

    # ------------------------------------------------------------------ #
    # 5. Slice the last 8 784 rows → C  (g → t conversion)
    # ------------------------------------------------------------------ #
    
    C = raw["y"].values[-len(DT_INDEX):] #/ 1_000_000

    return C, raw
