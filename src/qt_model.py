# Import necessary libraries and modules
import numpy as np  # For numerical computations and array handling
from time import time  # For measuring execution time
from typing import List, Optional, Dict  # For type hints
import pulp as pl  # For linear programming (LP) optimization
from src.scenario import Scenario  # Custom module defining the scenario
from src.util import get_validity_periods  # Utility function to get validity periods

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _print_deployment(d: np.ndarray, scenario: Scenario, window: List[int]) -> None:
    """
    Print a summary of deployed machines across the specified time window.
    Aggregates deployments by tier, machine type, and total, then displays 
    the counts in a formatted output.
    """
    # Initialize counters for total deployments, per-tier totals, and per-tier per-machine totals
    total = 0.0
    per_tier = {q: 0.0 for q in scenario.Q}  # Key: tier, value: total machines
    per_tier_per_m = {q: {m: 0.0 for m in scenario.M} for q in scenario.Q}  # Nested dict: tier -> machine -> count
    
    # Iterate over each time interval in the window
    for i in window:
        # Iterate over each tier (quality level)
        for q in scenario.Q:
            q_idx = scenario.Q.index(q)  # Get index of tier q
            # Iterate over each machine type
            for m in scenario.M:
                m_idx = scenario.M.index(m)  # Get index of machine m
                cnt = d[i, q_idx, m_idx]  # Number of machines deployed at interval i, tier q_idx, machine m_idx
                # Accumulate counts
                per_tier_per_m[q][m] += cnt
                per_tier[q] += cnt
                total += cnt
    
    # Print the summary header and totals
    print(f"\nDEPLOYED MACHINES (over {len(window)} intervals)")
    print("-" * 50)
    print(f"{'Total machines':<20} : {int(total):>6}")
    # Print per-tier totals and their machine breakdowns
    for q in scenario.Q:
        print(f"Tier-{q}: {int(per_tier[q]):>6} machines")
        for m in scenario.M:
            cnt = per_tier_per_m[q][m]
            if cnt > 0:  # Only print if there are deployed machines
                name = getattr(scenario.machines[m], "name", f"machine-{m}")  # Get machine name or default
                print(f" - {name:<18} : {int(cnt):>6}")

# ======================================================================
# QtModel - PuLP Implementation with New Objective
# ======================================================================

class QtModel:
    """
    A class to model and solve optimization problems for machine deployment 
    in a multi-tier service scenario using linear programming with PuLP.
    The objective focuses on minimizing emissions while meeting QoR targets 
    or maximizing QoR under emission budgets. It incorporates energy 
    consumption, embedded carbon, and requests handled per machine.
    """
    
    def __init__(self, scenario: Scenario):
        """
        Initialize the QtModel with the given scenario.
        Sets up storage arrays for allocation (a), deployment (d), and 
        requests handled per machine (y).
        """
        self.scenario = scenario
        # a: allocation of requests to users and tiers over intervals
        # Shape: (n_intervals, n_users, n_tiers)
        self.a_ = np.zeros((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=float)
        # d: number of machines deployed per tier and machine type over intervals
        # Shape: (n_intervals, n_tiers, n_machines)
        self.d_ = np.zeros((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=float)
        # y: requests handled per machine type, tier, and interval
        # Shape: (n_intervals, n_tiers, n_machines)
        self.y_ = np.zeros((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=float)

    # ------------------------------------------------------------------
    # Exact QoR computation
    # ------------------------------------------------------------------
    
    def _safe_qor(self, a: np.ndarray, R_hat: np.ndarray, vps: List[List[int]]) -> float:
        """
        Compute the Quality of Service/Reliability (QoR) metric as 1 minus 
        the maximum normalized error over all validity periods, users, and tiers.
        QoR measures how well service level objectives (SLOs) are met.
        The error is normalized by the difference between upper and lower SLO bounds.
        Returns a value between 0 and 1, where 1 is perfect SLO adherence.
        """
        S = self.scenario  # Alias for convenience
        errors = []
        # Iterate over each validity period (vp)
        for vp in vps:
            for u_idx, u in enumerate(S.U):  # For each user
                # Total requests for this user over the validity period
                total_requests = sum(R_hat[i, u_idx] for i in vp)
                if total_requests <= 0:
                    continue  # Skip if no requests
                for q_idx, q in enumerate(S.Q):  # For each tier
                    # Sum of allocated requests for this user and tier over vp
                    sum_alloc = sum(a[i, u_idx, q_idx] for i in vp)
                    sli = sum_alloc / total_requests  # Service Level Indicator (fraction of requests allocated)
                    lo = S.slo_lower[u_idx, q_idx]  # Lower SLO bound
                    hi = S.slo_upper[u_idx, q_idx]  # Upper SLO bound
                    den = abs(lo - hi)  # Denominator for normalization
                    if den <= 0:
                        continue  # Skip invalid SLO bounds
                    # Normalized difference from upper bound
                    diff = (sli - hi) / den
                    errors.append(abs(diff))  # Store absolute error
        # QoR = 1 - max(error), or 1 if no errors
        return 1.0 - (max(errors) if errors else 0.0)

    # ------------------------------------------------------------------
    # Calculate energy consumption
    # ------------------------------------------------------------------
    
    def _calculate_energy(
        self,
        a: np.ndarray,
        d: np.ndarray,
        y: np.ndarray,
        emissions_window: List[int],
        delta_t: float = 1.0,
    ) -> float:
        """
        Calculate total energy consumption in kWh over the emissions window.
        Formula: Energy = Σ_i Δt_i * Σ_{m,q} [p^idle_{m,q} * n_{m,q,i} + β_{m,q} * y_{m,q,i}]
        where β_{m,q} = (p^peak_{m,q} - p^idle_{m,q}) / θ_{m,q} (power per request),
        n_{m,q,i} is the number of machines, y_{m,q,i} is requests handled,
        and θ_{m,q} is machine performance (requests per hour).
        """
        S = self.scenario  # Alias
        n_tiers = len(S.Q)
        n_machines = len(S.M)
        total_energy = 0.0
        # Iterate over intervals in emissions window
        for i in emissions_window:
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    machine = S.machines[S.M[m_idx]]  # Machine object
                    q = S.Q[q_idx]
                    p_idle = machine._idle_power_usage  # Idle power in kW
                    p_peak = machine._power_usage  # Peak power in kW
                    theta = machine.performance[q]  # Requests per hour at tier q
                    if theta > 0:
                        beta = (p_peak - p_idle) / theta  # Power increase per request (kW/req)
                    else:
                        beta = 0.0  # Fallback if no performance data
                    n_mqi = d[i, q_idx, m_idx]  # Number of machines deployed
                    y_handled = y[i, q_idx, m_idx]  # Requests handled by these machines
                    # Energy for this interval: Δt * (idle power * count + β * requests)
                    energy = delta_t * (p_idle * n_mqi + beta * y_handled)
                    total_energy += energy
        return total_energy

    # ------------------------------------------------------------------
    # Build and solve LP with PuLP (NEW OBJECTIVE)
    # ------------------------------------------------------------------
    
    def _solve_lp(
        self,
        window: List[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        vps: List[List[int]],
        emissions_window: Optional[List[int]] = None,
        mode: str = "min_emissions",
        qor_target: Optional[float] = None,
        budget: Optional[float] = None,
        delta_t: float = 1.0,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Solve the optimization LP using PuLP with a new objective incorporating
        operational and embedded carbon emissions.
        Modes: "min_emissions" (minimize emissions subject to QoR target) or
        "max_qor" (maximize QoR subject to emissions budget).
        Returns: (total emissions, total energy, allocation a, deployment d, requests y, metrics dict)
        """
        # Set default emissions window to optimization window
        if emissions_window is None:
            emissions_window = window
        S = self.scenario  # Alias
        n_intervals = len(S.I)
        n_users = len(S.U)
        n_tiers = len(S.Q)
        n_machines = len(S.M)
        
        # Create LP problem: Minimize or Maximize based on mode
        if mode == "min_emissions":
            prob = pl.LpProblem("MinEmissions_New", pl.LpMinimize)
        else:
            prob = pl.LpProblem("MaxQoR_New", pl.LpMaximize)
        
        # Decision variables
        # a_vars: allocation of requests to users and tiers at each interval (continuous)
        a_vars = {}
        for i in range(n_intervals):
            for u_idx in range(n_users):
                for q_idx in range(n_tiers):
                    a_vars[i, u_idx, q_idx] = pl.LpVariable(
                        f"a_{i}_{u_idx}_{q_idx}", lowBound=0, cat='Continuous'
                    )
        # d_vars: number of machines deployed per tier and machine at each interval (integer)
        d_vars = {}
        for i in range(n_intervals):
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    d_vars[i, q_idx, m_idx] = pl.LpVariable(
                        f"d_{i}_{q_idx}_{m_idx}", lowBound=0, cat='Integer'
                    )
        # y_vars: requests handled per machine type at each interval (continuous)
        # Only for intervals in the window where performance data exists
        y_vars = {}
        for i in window:
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    theta = S.machines[S.M[m_idx]].performance[S.Q[q_idx]]
                    if theta > 0:
                        y_vars[i, q_idx, m_idx] = pl.LpVariable(
                            f"y_{i}_{q_idx}_{m_idx}", lowBound=0, cat='Continuous'
                        )
        
        # Helper variable for QoR in max_qor mode: max_error represents bounded SLO errors
        if mode == "max_qor":
            max_error = pl.LpVariable("max_error", lowBound=0, upBound=1, cat='Continuous')
            prob += 1.0 - max_error  # Objective: Maximize QoR = 1 - max_error
        
        # NEW OBJECTIVE FUNCTION: Minimize total carbon emissions (operational + embedded)
        if mode == "min_emissions":
            emissions_expr = 0
            for i in emissions_window:
                for q_idx in range(n_tiers):
                    for m_idx in range(n_machines):
                        machine = S.machines[S.M[m_idx]]
                        q = S.Q[q_idx]
                        p_idle = machine._idle_power_usage
                        p_peak = machine._power_usage
                        theta = machine.performance[q]
                        C_emb = machine.embedded_carbon  # Embedded carbon per machine hour (gCO2e/h)
                        if theta > 0 and (i, q_idx, m_idx) in y_vars:
                            beta = (p_peak - p_idle) / theta  # Power per request
                        else:
                            beta = 0.0
                        C_i = C_hat[i]  # Carbon intensity at interval i (gCO2e/kWh)
                        n_mqi = d_vars[i, q_idx, m_idx]
                        y_handled = y_vars.get((i, q_idx, m_idx), 0)
                        # Operational emissions: Δt * C_i * (idle power * count + β * requests)
                        operational = delta_t * C_i * (p_idle * n_mqi + beta * y_handled)
                        # Embedded emissions: C_emb * count (assuming Δt-normalized)
                        embedded = C_emb * n_mqi
                        emissions_expr += operational + embedded
            prob += emissions_expr  # Set as minimization objective
        
        # Constraints
        # 1. Demand satisfaction: Allocated requests must equal demand per user per interval
        for i in window:
            for u_idx in range(n_users):
                prob += (
                    pl.lpSum([a_vars[i, u_idx, q_idx] for q_idx in range(n_tiers)])
                    == R_hat[i, u_idx],
                    f"demand_{i}_{u_idx}"
                )
        
        # 2. Capacity constraints: Total requests handled must not exceed total machine capacity
        for i in window:
            for q_idx in range(n_tiers):
                # Total requests allocated to this tier at interval i
                total_requests = pl.lpSum([a_vars[i, u_idx, q_idx] for u_idx in range(n_users)])
                # Total capacity from deployed machines: Σ (machines * performance)
                capacity = pl.lpSum([
                    d_vars[i, q_idx, m_idx] * S.machines[S.M[m_idx]].performance[S.Q[q_idx]]
                    for m_idx in range(n_machines)
                ])
                prob += total_requests <= capacity, f"capacity_{i}_{q_idx}"
                
                # NEW: Balance y to total requests per tier
                y_sum = pl.lpSum([
                    y_vars.get((i, q_idx, m_idx), 0) for m_idx in range(n_machines)
                ])
                prob += y_sum == total_requests, f"balance_{i}_{q_idx}"
                
                # NEW: Each machine's handled requests <= its capacity
                for m_idx in range(n_machines):
                    if (i, q_idx, m_idx) in y_vars:
                        prob += y_vars[i, q_idx, m_idx] <= d_vars[i, q_idx, m_idx] * S.machines[S.M[m_idx]].performance[S.Q[q_idx]], f"machine_cap_{i}_{q_idx}_{m_idx}"
        
        # 3. QoR/SLO constraints: Ensure SLO bounds are met (with error tolerance in constraints)
        for vp_idx, vp in enumerate(vps):
            for u_idx in range(n_users):
                total_requests = sum(R_hat[i, u_idx] for i in vp)
                if total_requests <= 1e-9:
                    continue
                for q_idx in range(n_tiers):
                    sum_alloc = pl.lpSum([a_vars[i, u_idx, q_idx] for i in vp])
                    sli = sum_alloc / total_requests  # Fraction allocated
                    lo = S.slo_lower[u_idx, q_idx]
                    hi = S.slo_upper[u_idx, q_idx]
                    den = abs(lo - hi)
                    if den <= 1e-9:
                        continue
                    if mode == "max_qor":
                        # Bound SLO errors by max_error
                        prob += sli - hi <= max_error * den, f"qor_upper_{vp_idx}_{u_idx}_{q_idx}"
                        prob += hi - sli <= max_error * den, f"qor_lower_{vp_idx}_{u_idx}_{q_idx}"
                    else:  # min_emissions: Errors bounded by target QoR
                        err_max = 1.0 - qor_target  # Max allowed error
                        prob += sli - hi <= err_max * den, f"qor_upper_{vp_idx}_{u_idx}_{q_idx}"
                        prob += hi - sli <= err_max * den, f"qor_lower_{vp_idx}_{u_idx}_{q_idx}"
        
        # 4. Budget constraint for max_qor mode: Emissions <= budget
        if mode == "max_qor" and budget is not None:
            emissions_expr = 0  # Same calculation as objective
            for i in emissions_window:
                for q_idx in range(n_tiers):
                    for m_idx in range(n_machines):
                        machine = S.machines[S.M[m_idx]]
                        q = S.Q[q_idx]
                        p_idle = machine._idle_power_usage
                        p_peak = machine._power_usage
                        theta = machine.performance[q]
                        C_emb = machine.embedded_carbon
                        if theta > 0 and (i, q_idx, m_idx) in y_vars:
                            beta = (p_peak - p_idle) / theta
                        else:
                            beta = 0.0
                        C_i = C_hat[i]
                        n_mqi = d_vars[i, q_idx, m_idx]
                        y_handled = y_vars.get((i, q_idx, m_idx), 0)
                        operational = delta_t * C_i * (p_idle * n_mqi + beta * y_handled)
                        embedded = C_emb * n_mqi
                        emissions_expr += operational + embedded
            prob += emissions_expr <= budget, "budget_constraint"
        
        # Solve the LP with a timeout
        solver = pl.PULP_CBC_CMD(msg=0, timeLimit=300)  # Quiet, 5-minute timeout
        status = prob.solve(solver)
        
        # Handle non-optimal solutions
        if status not in [pl.LpStatusOptimal, pl.LpStatusNotSolved]:
            print(f"WARNING: PuLP solver status: {pl.LpStatus[status]}")
            return 0.0, 0.0, np.zeros_like(self.a_), np.zeros_like(self.d_), np.zeros_like(self.y_), {
                "status": pl.LpStatus[status],
                "objective": 0.0
            }
        
        # Extract results into numpy arrays
        a = np.zeros((n_intervals, n_users, n_tiers))
        for (i, u_idx, q_idx), var in a_vars.items():
            a[i, u_idx, q_idx] = var.varValue if var.varValue is not None else 0.0
        
        d = np.zeros((n_intervals, n_tiers, n_machines))
        for (i, q_idx, m_idx), var in d_vars.items():
            d[i, q_idx, m_idx] = var.varValue if var.varValue is not None else 0.0
        
        y = np.zeros((n_intervals, n_tiers, n_machines))
        for (i, q_idx, m_idx), var in y_vars.items():
            y[i, q_idx, m_idx] = var.varValue if var.varValue is not None else 0.0
        
        # Calculate total emissions and energy
        emissions = 0.0
        energy = self._calculate_energy(a, d, y, emissions_window, delta_t)
        for i in emissions_window:
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    if (i, q_idx, m_idx) in y_vars:
                        machine = S.machines[S.M[m_idx]]
                        q = S.Q[q_idx]
                        p_idle = machine._idle_power_usage
                        p_peak = machine._power_usage
                        theta = machine.performance[q]
                        C_emb = machine.embedded_carbon
                        beta = (p_peak - p_idle) / theta if theta > 0 else 0.0
                        C_i = C_hat[i]
                        n_mqi = d[i, q_idx, m_idx]
                        y_handled = y[i, q_idx, m_idx]
                        operational = delta_t * C_i * (p_idle * n_mqi + beta * y_handled)
                        embedded = C_emb * n_mqi
                        emissions += operational + embedded
        
        # Metrics dict
        metrics = {
            "status": pl.LpStatus[status],
            "objective": pl.value(prob.objective) if prob.objective is not None else 0.0
        }
        return emissions, energy, a, d, y, metrics

    # ------------------------------------------------------------------
    # Minimize emissions s.t. QoR >= target
    # ------------------------------------------------------------------
    
    def minimize_emissions(
        self,
        qor_target: float,
        window: List[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        past_vps: bool = True,
        future_vps: bool = True,
        delta_t: float = 1.0,
    ) -> Dict[str, float]:
        """
        Minimize total carbon emissions while ensuring QoR >= qor_target.
        Use the _solve_lp method in "min_emissions" mode, then compute achieved QoR.
        Returns metrics including emissions, energy, QoR, runtime, etc.
        Updates internal arrays a_, d_, y_ with results.
        """
        t0 = time()  # Start timer
        # Get validity periods for QoR calculation
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)
        # Solve the LP
        em, energy, a, d, y, lp_metrics = self._solve_lp(
            window, R_hat, C_hat, vps,
            emissions_window=window,
            mode="min_emissions",
            qor_target=qor_target,
            delta_t=delta_t
        )
        # Compute achieved QoR
        achieved_qor = self._safe_qor(a, R_hat, vps)
        # Warn if target not met (within tolerance)
        if achieved_qor < qor_target - 1e-4:
            print(f"WARNING: QoR target {qor_target:.4f} not met (achieved {achieved_qor:.4f})")
        # Collect metrics
        metrics = {
            "qor_target": qor_target,
            "qor_achieved": achieved_qor,
            "emissions": em,  # Total emissions in gCO2e
            "energy": energy,  # Total energy in kWh
            "runtime": time() - t0,  # Seconds
            "work": 0.0,  # Placeholder
            "mip_gap": 0.0,  # Placeholder
            "lp_status": lp_metrics["status"],
            "objective": lp_metrics["objective"]
        }
        # Store results in instance arrays
        self.a_[:] = a
        self.d_[:] = d
        self.y_[:] = y
        return metrics

    # ------------------------------------------------------------------
    # Maximize QoR s.t. emissions <= budget
    # ------------------------------------------------------------------
    
    def maximize_qor(
        self,
        budget: float,
        window: List[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        past_vps: bool = True,
        future_vps: bool = True,
        emissions_window: Optional[List[int]] = None,
        delta_t: float = 1.0,
    ) -> Dict[str, float]:
        """
        Maximize QoR (minimize max SLO error) subject to total emissions <= budget.
        Use the _solve_lp method in "max_qor" mode.
        Returns metrics including achieved QoR, emissions, energy, runtime, etc.
        Default emissions_window is the full scenario interval set.
        Updates internal arrays a_, d_, y_ with results.
        """
        # Default emissions window to all intervals if not specified
        if emissions_window is None:
            emissions_window = self.scenario.I
        t0 = time()  # Start timer
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)
        # Solve the LP
        em, energy, a, d, y, lp_metrics = self._solve_lp(
            window, R_hat, C_hat, vps,
            emissions_window=emissions_window,
            mode="max_qor",
            budget=budget,
            delta_t=delta_t
        )
        # Compute achieved QoR
        achieved_qor = self._safe_qor(a, R_hat, vps)
        # Collect metrics
        metrics = {
            "qor_target": achieved_qor,  # Since we maximize, this is the achieved QoR
            "emissions": em,  # Total emissions in gCO2e
            "energy": energy,  # Total energy in kWh
            "runtime": time() - t0,  # Seconds
            "work": 0.0,  # Placeholder
            "mip_gap": 0.0,  # Placeholder
            "lp_status": lp_metrics["status"],
            "objective": lp_metrics["objective"]
        }
        # Store results in instance arrays
        self.a_[:] = a
        self.d_[:] = d
        self.y_[:] = y
        return metrics

# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    # Load configuration using Hydra
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    # Initialize scenario from config
    scenario = Scenario.from_config(cfg)
    R_hat = scenario.R  # Demand matrix (intervals x users)
    C_hat = scenario.C  # Carbon intensity vector (intervals)
    # Print sample data
    print(f"R_hat[0]: {R_hat[0]}")
    print(f"C_hat[0]: {C_hat[0]}")
    
    # Create solver instance
    solver = QtModel(scenario)
    window = scenario.I[1002:1003]  # Single interval window for demo (I is list of intervals)
    
    # Header for demo
    print("\n" + "=" * 60)
    print("PULP SOLVER")
    print("=" * 60)
    
    # Print machine parameters for reference
    print("\nMachine Parameters:")
    print("-" * 60)
    for m_idx, m in enumerate(scenario.M):
        machine = scenario.machines[m]  # Machine object
        print(f"\n{machine.name}:")
    
    # Run minimization demo
    print("\n" + "=" * 60)
    print("1. MINIMIZE EMISSIONS")
    print("-" * 50)
    min_res = solver.minimize_emissions(qor_target=0.1, window=window, R_hat=R_hat, C_hat=C_hat)
    print(f" Runtime : {min_res['runtime']:.3f} s")
    print(f" LP Status : {min_res['lp_status']}")
    print(f" Emissions : {min_res['emissions']:_.0f} gCO₂e")
    print(f" Energy : {min_res['energy']:_.2f} kWh")
    print(f" Achieved QoR : {min_res['qor_achieved']:.4f}")
    _print_deployment(solver.d_, scenario, window)  # Print deployment summary
    
    # Commented out maximize QoR demo
    # budget = min_res["emissions"] * 1.05  # 5% above min emissions
    # print("\n2. MAXIMIZE QoR (budget = +5%)")
    # print("-" * 50)
    # max_res = solver.maximize_qor(budget=budget, window=window, R_hat=R_hat, C_hat=C_hat)
    # print(f" Runtime : {max_res['runtime']:.3f} s")
    # print(f" LP Status : {max_res['lp_status']}")
    # print(f" Emissions : {max_res['emissions']:_.0f} gCO₂e")
    # print(f" Energy : {max_res['energy']:_.2f} kWh")
    # print(f" Achieved QoR: {max_res['qor_target']:.4f}")
    # _print_deployment(solver.d_, scenario, window)
    
    print("\nDone!")