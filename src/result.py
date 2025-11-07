import os
import pickle
from typing import Optional, List, Dict

import numpy as np

from src.scenario import Scenario
from src.util import get_validity_periods

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def _machine_cost(scenario: Scenario, m_idx: int, i: int, C_hat: np.ndarray) -> float:
    """Emission cost of one machine m at interval i."""
    machine = scenario.machines[scenario.M[m_idx]]
    power = machine.load_independent_power_usage()
    return power * C_hat[i] + machine.embedded_carbon


def interval_power_per_machine(i: int, q_idx: int, m_idx: int, d: np.ndarray, 
                               scenario: Scenario, load_dependent: bool = False, 
                               a_: Optional[np.ndarray] = None) -> float:
    """Calculate power per machine at interval i for tier q and machine type m."""
    machine = scenario.machines[scenario.M[m_idx]]
    base_power = machine.load_independent_power_usage()
    
    if not load_dependent or a_ is None:
        return base_power
    
    # Calculate load-dependent power
    total_requests = a_[i, :, q_idx].sum()
    machine_count = d[i, q_idx, m_idx]
    
    if machine_count == 0:
        return base_power
    
    performance = machine.performance[scenario.Q[q_idx]]
    if performance == 0:
        return base_power
    
    utilization = total_requests / (machine_count * performance)
    utilization = min(1.0, max(0.0, utilization))
    
    # Assuming linear power scaling between idle and peak
    return base_power * (0.5 + 0.5 * utilization)


def interval_emissions(i: int, q_idx: int, m_idx: int, d: np.ndarray, 
                       scenario: Scenario, C_hat: np.ndarray, 
                       load_dependent: bool = False, 
                       a_: Optional[np.ndarray] = None) -> float:
    """Calculate emissions at interval i for tier q and machine type m."""
    machine = scenario.machines[scenario.M[m_idx]]
    power = interval_power_per_machine(i, q_idx, m_idx, d, scenario, load_dependent, a_)
    machine_count = d[i, q_idx, m_idx]
    
    operational_emissions = power * C_hat[i] * machine_count
    embodied_emissions = machine.embedded_carbon * machine_count
    
    return operational_emissions + embodied_emissions


class Result:
    def __init__(self,
                 a: np.ndarray,
                 d: np.ndarray,
                 optimizer: Optional[object] = None,
                 scenario: Optional[Scenario] = None,
                 metrics: Optional[Dict] = None,
                 info: Optional[Dict] = None):
        """
        Result container for optimization solutions.
        
        Args:
            a: Allocation array (n_intervals, n_users, n_tiers)
            d: Deployment array (n_intervals, n_tiers, n_machines)
            optimizer: Optimizer instance that produced this result
            scenario: Scenario instance
            metrics: Dictionary of optimization metrics
            info: Additional information (reflected in result name)
        """
        self.a = a.astype(np.float64)  # (t, u, q)
        self.d = d.astype(np.float64)  # (t, q, m)
        self.optimizer = optimizer
        self.scenario = scenario
        self.metrics = metrics if metrics is not None else {}
        self.info = info if info is not None else {}

        # Cache
        self._qor_list = None

    @property
    def name(self) -> str:
        """Generate a unique name for this result."""
        return get_result_name(self.scenario, self.optimizer, self.info)

    def emissions_over_time(self, C_hat: Optional[np.ndarray] = None, 
                           load_dependent: bool = False) -> np.ndarray:
        """
        Calculate emissions for each time interval.
        
        Args:
            C_hat: Carbon intensity array (if None, uses scenario.C)
            load_dependent: Whether to include load-dependent power consumption
            
        Returns:
            Array of emissions per interval (n_intervals,)
        """
        C = C_hat if C_hat is not None else self.scenario.C
        return np.array([
            sum(
                interval_emissions(i, q_idx, m_idx, self.d, self.scenario, C, 
                                 load_dependent=load_dependent, a_=self.a)
                for q_idx in range(len(self.scenario.Q))
                for m_idx in range(len(self.scenario.M))
            )
            for i in self.scenario.I
        ])

    def energy_over_time(self, load_dependent: bool = False) -> np.ndarray:
        """
        Calculate energy consumption for each time interval.
        
        Args:
            load_dependent: Whether to include load-dependent power consumption
            
        Returns:
            Array of energy per interval (n_intervals,)
        """
        return np.array([
            sum(
                self.d[i, q_idx, m_idx] * 
                interval_power_per_machine(i, q_idx, m_idx, self.d, self.scenario, 
                                          load_dependent=load_dependent, a_=self.a)
                for q_idx in range(len(self.scenario.Q))
                for m_idx in range(len(self.scenario.M))
            )
            for i in self.scenario.I
        ])

    def emissions(self, i: Optional[int] = None, C_hat: Optional[np.ndarray] = None, 
                 load_dependent: bool = False) -> float:
        """
        Calculate total emissions or emissions at a specific interval.
        
        Args:
            i: Interval index (if None, returns total)
            C_hat: Carbon intensity array
            load_dependent: Whether to include load-dependent power
            
        Returns:
            Emissions in gCO2e
        """
        z = self.emissions_over_time(C_hat, load_dependent=load_dependent)
        if i is not None:
            return z[i]
        return z.sum()

    def qor_list(self) -> List[float]:
        """
        Calculate QoR for all validity periods.
        
        Returns:
            List of QoR values for each validity period
        """
        if self._qor_list is None:
            vps = get_validity_periods(self.scenario.I, self.scenario.vp)
            self._qor_list = [self._qor(vp, self.scenario.R) for vp in vps]
        return self._qor_list

    def qor_quantile(self, quantile: float) -> float:
        """Calculate QoR at a specific quantile."""
        return np.quantile(self.qor_list(), quantile)

    def qor_target(self) -> float:
        """Get the minimum QoR across all validity periods."""
        qor_vals = self.qor_list()
        return np.min(qor_vals) if len(qor_vals) > 0 else 0.0

    def qor_target_forecast(self, R_hat: np.ndarray) -> float:
        """
        Calculate minimum QoR using forecasted demand.
        
        Args:
            R_hat: Forecasted demand array
            
        Returns:
            Minimum QoR value
        """
        vps = get_validity_periods(self.scenario.I, self.scenario.vp)
        qor_vals = [self._qor(vp, R_hat) for vp in vps]
        return np.min(qor_vals) if len(qor_vals) > 0 else 0.0

    def _qor(self, validity_period: List[int], R_hat: Optional[np.ndarray] = None) -> float:
        """
        Calculate QoR for a specific validity period.
        
        QoR = 1 - max_error, where error = |SLI - target| / |lower - upper|
        
        Args:
            validity_period: List of interval indices
            R_hat: Request array (if None, uses scenario.R)
            
        Returns:
            QoR value between 0 and 1
        """
        if R_hat is None:
            R_hat = self.scenario.R

        # Sum over time intervals in the validity period
        # a is (n_intervals, n_users, n_tiers)
        # R_hat is (n_intervals, n_users)
        total_allocation_per_user_tier = self.a[validity_period, :, :].sum(axis=0)  # (n_users, n_tiers)
        total_requests_per_user = R_hat[validity_period, :].sum(axis=0)  # (n_users,)
        
        # If no requests in this validity period, QoR is perfect
        if total_requests_per_user.sum() < 1e-9:
            return 1.0

        # Calculate service level indicators (SLI) for each (user, tier)
        # SLI = allocation / total_requests for each user-tier pair
        service_levels = np.divide(
            total_allocation_per_user_tier,
            total_requests_per_user[:, np.newaxis],  # Broadcast to (n_users, 1)
            out=np.zeros_like(total_allocation_per_user_tier),
            where=total_requests_per_user[:, np.newaxis] > 1e-9
        )

        # Calculate errors: |SLI - upper| / |lower - upper|
        nominator = np.abs(service_levels - self.scenario.slo_upper)
        denominator = np.abs(self.scenario.slo_lower - self.scenario.slo_upper)
        
        # Only compute errors where:
        # 1. Denominator is non-zero (there's an actual SLO range)
        # 2. There are requests for this user
        valid_mask = (denominator > 1e-9) & (total_requests_per_user[:, np.newaxis] > 1e-9)
        
        if not valid_mask.any():
            return 1.0  # No valid constraints to check
        
        errors = np.divide(
            nominator,
            denominator,
            out=np.zeros_like(nominator),
            where=valid_mask
        )
        
        highest_error = np.max(errors[valid_mask])
        
        # Clamp to valid range [0, 1]
        highest_error = max(0.0, min(1.0, highest_error))
        
        return round(1 - highest_error, 3)

    def print_stats(self) -> None:
        """Print summary statistics of the result."""
        qor_list = self.qor_list()
        if len(qor_list) == 0:
            print("No QoR values calculated (no validity periods)")
            return
            
        print(f"Actual QoR: min={np.min(qor_list):.3f}, "
              f"5th={self.qor_quantile(0.05):.3f}, "
              f"mean={np.mean(qor_list):.3f}±{np.std(qor_list):.3f} "
              f"at {self.emissions() / 1e6:.2f} tCO2e")
        
        if self.metrics:
            print("\nOptimization Metrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    def save(self, result_dir: str = "results") -> None:
        """
        Save result to disk.
        
        Args:
            result_dir: Directory to save results
        """
        os.makedirs(f"{_BASE_DIR}/{result_dir}", exist_ok=True)
        path = f'{_BASE_DIR}/{result_dir}/{self.name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        print(f"Saved result to {path}")

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "a": self.a,
            "d": self.d,
            "optimizer": self.optimizer,
            "metrics": self.metrics,
            "info": self.info,
            "scenario_name": self.scenario.name,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Result":
        """Load result from dictionary."""
        return cls(
            a=d["a"], 
            d=d["d"], 
            optimizer=d.get("optimizer"),
            metrics=d.get("metrics"),
            info=d.get("info"),
            scenario=Scenario.from_name(d["scenario_name"])
        )


def load_result(name: str, result_dir: str = "results") -> Result:
    """
    Load a result from disk.
    
    Args:
        name: Result name or filename
        result_dir: Directory containing results
        
    Returns:
        Result instance
    """
    if ".pkl" not in name:
        name = f'{name}.pkl'
    path = f'{_BASE_DIR}/{result_dir}/{name}'
    with open(path, 'rb') as f:
        result = Result.from_dict(pickle.load(f))
    return result


def load_results(result_dir: str = "results") -> List[Result]:
    """
    Load all results from a directory.
    
    Args:
        result_dir: Directory containing results
        
    Returns:
        List of Result instances
    """
    results = []
    result_path = f"{_BASE_DIR}/{result_dir}"
    
    if not os.path.exists(result_path):
        return results
    
    for filename in os.listdir(result_path):
        if filename.endswith(".pkl"):
            try:
                results.append(load_result(filename, result_dir=result_dir))
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return results


def get_result_name(scenario: Scenario, optimizer: Optional[object], info: Dict) -> str:
    """
    Generate a unique name for a result.
    
    Args:
        scenario: Scenario instance
        optimizer: Optimizer instance
        info: Additional info dictionary
        
    Returns:
        Formatted result name
    """
    name_parts = [scenario.name]
    
    for k, v in sorted(info.items()):
        if isinstance(v, float):
            name_parts.append(f"{k}={v:.2f}")
        else:
            name_parts.append(f"{k}={v}")
    
    if optimizer is not None:
        optimizer_name = getattr(optimizer, 'name', optimizer.__class__.__name__)
        name_parts.append(optimizer_name)
    
    name_parts.append(f"seed={scenario.seed}")
    
    return ",".join(name_parts)


def result_exists(result_name: str, result_dir: str = "results") -> bool:
    """
    Check if a result file exists.
    
    Args:
        result_name: Name of the result
        result_dir: Directory to check
        
    Returns:
        True if result exists
    """
    path = f"{_BASE_DIR}/{result_dir}/{result_name}.pkl"
    return os.path.exists(path)


# ======================================================================
# Demo / Testing
# ======================================================================
if __name__ == "__main__":
    from hydra import compose, initialize
    from src.qt_model import QtModel
    
    print("\n" + "=" * 70)
    print("RESULT CLASS DEMONSTRATION")
    print("=" * 70)
    
    # Initialize scenario
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    
    scenario = Scenario.from_config(cfg)
    R_hat = scenario.R * 1000
    C_hat = scenario.C * 1_000_000
    
    # Create optimizer and solve
    print("\n1. Running optimization...")
    print("-" * 70)
    solver = QtModel(scenario)
    window = scenario.I[:1]  # First 24 intervals
    
    print(f"Window size: {len(window)} intervals")
    print(f"Total requests in window: {R_hat[window].sum():,.0f}")
    print(f"Number of users: {len(scenario.U)}")
    print(f"Number of tiers: {len(scenario.Q)}")
    print(f"Number of machine types: {len(scenario.M)}")
    
    metrics = solver.minimize_emissions(
        qor_target=0.6,
        window=window,
        R_hat=R_hat,
        C_hat=C_hat
    )
    
    print(f"\nOptimization completed:")
    print(f"  Status: {metrics.get('lp_status', 'Unknown')}")
    print(f"  Total allocation: {solver.a_.sum():,.0f}")
    print(f"  Total machines: {solver.d_.sum():,.0f}")
    
    # Create Result object
    print("\n2. Creating Result object...")
    print("-" * 70)
    result = Result(
        a=solver.a_,
        d=solver.d_,
        optimizer=solver,
        scenario=scenario,
        metrics=metrics,
        info={"qor_target": 0.6, "method": "min_emissions"}
    )
    
    # Display statistics
    print("\n3. Result Statistics:")
    print("-" * 70)
    result.print_stats()
    
    # Emissions analysis
    print("\n4. Emissions Analysis:")
    print("-" * 70)
    total_emissions = result.emissions()
    print(f"Total emissions: {total_emissions:,.0f} gCO2e ({total_emissions/1e6:.3f} tCO2e)")
    
    emissions_time = result.emissions_over_time()
    print(f"Peak interval emissions: {emissions_time.max():,.0f} gCO2e")
    print(f"Mean interval emissions: {emissions_time.mean():,.0f} gCO2e")
    
    # Energy analysis
    print("\n5. Energy Analysis:")
    print("-" * 70)
    energy_time = result.energy_over_time()
    total_energy = energy_time.sum()
    print(f"Total energy: {total_energy:,.0f} Wh ({total_energy/1e6:.3f} MWh)")
    print(f"Peak interval energy: {energy_time.max():,.0f} Wh")
    
    # QoR analysis
    print("\n6. QoR Analysis:")
    print("-" * 70)
    qor_list = result.qor_list()
    if len(qor_list) > 0:
        print(f"QoR statistics:")
        print(f"  Min:    {np.min(qor_list):.4f}")
        print(f"  5th %:  {result.qor_quantile(0.05):.4f}")
        print(f"  Median: {result.qor_quantile(0.50):.4f}")
        print(f"  95th %: {result.qor_quantile(0.95):.4f}")
        print(f"  Max:    {np.max(qor_list):.4f}")
        print(f"  Mean:   {np.mean(qor_list):.4f} ± {np.std(qor_list):.4f}")
    else:
        print("No validity periods found")
    
    # Deployment summary
    print("\n7. Deployment Summary:")
    print("-" * 70)
    total_machines_per_tier = result.d.sum(axis=(0, 2))  # Sum over time and machine types
    for q_idx, q in enumerate(scenario.Q):
        print(f"  Tier {q}: {total_machines_per_tier[q_idx]:.0f} machine-intervals")
    
    # Save and load test
    print("\n8. Save/Load Test:")
    print("-" * 70)
    print(f"Result name: {result.name}")
    
    result_dir = "test_results"
    result.save(result_dir=result_dir)
    
    loaded_result = load_result(result.name, result_dir=result_dir)
    print(f"Successfully loaded result: {loaded_result.name}")
    print(f"Allocations match: {np.allclose(result.a, loaded_result.a)}")
    print(f"Deployments match: {np.allclose(result.d, loaded_result.d)}")
    
    # List all results
    all_results = load_results(result_dir=result_dir)
    print(f"\nFound {len(all_results)} result(s) in '{result_dir}/'")
    
    # Cleanup
    import shutil
    if os.path.exists(f"{_BASE_DIR}/{result_dir}"):
        shutil.rmtree(f"{_BASE_DIR}/{result_dir}")
        print(f"Cleaned up test directory: {result_dir}/")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)