import numpy as np
from time import time
from typing import List, Optional, Dict
import pulp as pl
from src.scenario import Scenario
from src.util import get_validity_periods


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _print_deployment(d: np.ndarray, scenario: Scenario, window: List[int]) -> None:
    """Print summary of deployed machines."""
    total = 0.0
    per_tier = {q: 0.0 for q in scenario.Q}
    per_tier_per_m = {q: {m: 0.0 for m in scenario.M} for q in scenario.Q}
    for i in window:
        for q in scenario.Q:
            q_idx = scenario.Q.index(q)
            for m in scenario.M:
                m_idx = scenario.M.index(m)
                cnt = d[i, q_idx, m_idx]
                per_tier_per_m[q][m] += cnt
                per_tier[q] += cnt
                total += cnt
    print(f"\nDEPLOYED MACHINES (over {len(window)} intervals)")
    print("-" * 50)
    print(f"{'Total machines':<20} : {int(total):>6}")
    for q in scenario.Q:
        print(f"Tier-{q}: {int(per_tier[q]):>6} machines")
        for m in scenario.M:
            cnt = per_tier_per_m[q][m]
            if cnt > 0:
                name = getattr(scenario.machines[m], "name", f"machine-{m}")
                print(f" - {name:<18} : {int(cnt):>6}")


# ======================================================================
# QtModel - PuLP Implementation with New Objective
# ======================================================================
class QtModel:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.a_ = np.zeros((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=float)
        self.d_ = np.zeros((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=float)

    # ------------------------------------------------------------------
    # Exact QoR computation
    # ------------------------------------------------------------------
    def _safe_qor(self, a: np.ndarray, R_hat: np.ndarray, vps: List[List[int]]) -> float:
        """Compute QoR as 1 - max_error over all validity periods, users, and tiers."""
        S = self.scenario
        errors = []
        for vp in vps:
            for u_idx, u in enumerate(S.U):
                total_requests = sum(R_hat[i, u_idx] for i in vp)
                if total_requests <= 0:
                    continue
                for q_idx, q in enumerate(S.Q):
                    sum_alloc = sum(a[i, u_idx, q_idx] for i in vp)
                    sli = sum_alloc / total_requests
                    lo = S.slo_lower[u_idx, q_idx]
                    hi = S.slo_upper[u_idx, q_idx]
                    den = abs(lo - hi)
                    if den <= 0:
                        continue
                    diff = (sli - hi) / den
                    errors.append(abs(diff))
        return 1.0 - (max(errors) if errors else 0.0)

    # ------------------------------------------------------------------
    # Calculate energy consumption
    # ------------------------------------------------------------------
    def _calculate_energy(
        self,
        a: np.ndarray,
        d: np.ndarray,
        emissions_window: List[int],
        delta_t: float = 1.0,
    ) -> float:
        """
        Calculate total energy consumption in kWh.
        
        Energy = Σ_i Δt_i * Σ_{m,q} [p^idle_{m,q} * n_{m,q,i} + β_{m,q} * Σ_u y_{u,m,q,i}]
        
        where β_{m,q} = (p^peak_{m,q} - p^idle_{m,q}) / θ_{m,q}
        """
        S = self.scenario
        n_tiers = len(S.Q)
        n_machines = len(S.M)
        n_users = len(S.U)
        
        total_energy = 0.0
        
        for i in emissions_window:
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    machine = S.machines[S.M[m_idx]]
                    q = S.Q[q_idx]
                    
                    p_idle = machine._idle_power_usage  # kW
                    p_peak = machine._power_usage  # kW
                    theta = machine.performance[q]  # Requests per hour
                    
                    # Calculate β_{m,q}
                    if theta > 0:
                        beta = (p_peak - p_idle) / theta  # kW per request
                    else:
                        beta = 0.0
                    
                    n_mqi = d[i, q_idx, m_idx]  # Number of machines
                    sum_requests = sum(a[i, u_idx, q_idx] for u_idx in range(n_users))
                    
                    # Energy in kWh: Δt * (p^idle * n + β * requests)
                    energy = delta_t * (p_idle * n_mqi + beta * sum_requests)
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
    ) -> tuple[float, float, np.ndarray, np.ndarray, Dict]:
        """
        Solve the optimization problem using PuLP with new objective function.
        
        Returns: (emissions, energy, a, d, metrics)
        """
        if emissions_window is None:
            emissions_window = window
        
        S = self.scenario
        n_intervals = len(S.I)
        n_users = len(S.U)
        n_tiers = len(S.Q)
        n_machines = len(S.M)

        # Create the LP problem
        if mode == "min_emissions":
            prob = pl.LpProblem("MinEmissions_New", pl.LpMinimize)
        else:
            prob = pl.LpProblem("MaxQoR_New", pl.LpMaximize)

        # Decision variables
        a_vars = {}
        for i in range(n_intervals):
            for u_idx in range(n_users):
                for q_idx in range(n_tiers):
                    a_vars[i, u_idx, q_idx] = pl.LpVariable(
                        f"a_{i}_{u_idx}_{q_idx}", lowBound=0, cat='Continuous'
                    )

        d_vars = {}
        for i in range(n_intervals):
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    d_vars[i, q_idx, m_idx] = pl.LpVariable(
                        f"d_{i}_{q_idx}_{m_idx}", lowBound=0, cat='Integer'
                    )

        # Helper variable for QoR (max error)
        if mode == "max_qor":
            max_error = pl.LpVariable("max_error", lowBound=0, upBound=1, cat='Continuous')
            prob += 1.0 - max_error  # Maximize QoR = 1 - max_error

        # NEW OBJECTIVE FUNCTION
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
                        C_emb = machine.embedded_carbon
                        
                        if theta > 0:
                            beta = (p_peak - p_idle) / theta
                        else:
                            beta = 0.0
                        
                        C_i = C_hat[i]
                        n_mqi = d_vars[i, q_idx, m_idx]
                        sum_requests = pl.lpSum([a_vars[i, u_idx, q_idx] for u_idx in range(n_users)])
                        
                        operational = delta_t * C_i * (p_idle * n_mqi + beta * sum_requests)
                        embedded = C_emb * n_mqi
                        emissions_expr += operational + embedded
            
            prob += emissions_expr

        # Constraints
        
        # 1. Demand satisfaction
        for i in window:
            for u_idx in range(n_users):
                prob += (
                    pl.lpSum([a_vars[i, u_idx, q_idx] for q_idx in range(n_tiers)]) 
                    == R_hat[i, u_idx],
                    f"demand_{i}_{u_idx}"
                )

        # 2. Capacity constraints
        for i in window:
            for q_idx in range(n_tiers):
                total_requests = pl.lpSum([a_vars[i, u_idx, q_idx] for u_idx in range(n_users)])
                capacity = pl.lpSum([
                    d_vars[i, q_idx, m_idx] * S.machines[S.M[m_idx]].performance[S.Q[q_idx]]
                    for m_idx in range(n_machines)
                    if S.machines[S.M[m_idx]].performance[S.Q[q_idx]] > 0
                ])
                prob += total_requests <= capacity, f"capacity_{i}_{q_idx}"

        # 3. QoR/SLO constraints
        for vp_idx, vp in enumerate(vps):
            for u_idx in range(n_users):
                total_requests = sum(R_hat[i, u_idx] for i in vp)
                if total_requests <= 1e-9:
                    continue
                
                for q_idx in range(n_tiers):
                    sum_alloc = pl.lpSum([a_vars[i, u_idx, q_idx] for i in vp])
                    sli = sum_alloc / total_requests
                    
                    lo = S.slo_lower[u_idx, q_idx]
                    hi = S.slo_upper[u_idx, q_idx]
                    den = abs(lo - hi)
                    
                    if den <= 1e-9:
                        continue
                    
                    if mode == "max_qor":
                        prob += sli - hi <= max_error * den, f"qor_upper_{vp_idx}_{u_idx}_{q_idx}"
                        prob += hi - sli <= max_error * den, f"qor_lower_{vp_idx}_{u_idx}_{q_idx}"
                    else:  # min_emissions
                        err_max = 1.0 - qor_target
                        prob += sli - hi <= err_max * den, f"qor_upper_{vp_idx}_{u_idx}_{q_idx}"
                        prob += hi - sli <= err_max * den, f"qor_lower_{vp_idx}_{u_idx}_{q_idx}"

        # 4. Budget constraint (for max_qor mode)
        if mode == "max_qor" and budget is not None:
            emissions_expr = 0
            for i in emissions_window:
                for q_idx in range(n_tiers):
                    for m_idx in range(n_machines):
                        machine = S.machines[S.M[m_idx]]
                        q = S.Q[q_idx]
                        
                        p_idle = machine._idle_power_usage
                        p_peak = machine._power_usage
                        theta = machine.performance[q]
                        C_emb = machine.embedded_carbon
                        
                        if theta > 0:
                            beta = (p_peak - p_idle) / theta
                        else:
                            beta = 0.0
                        
                        C_i = C_hat[i]
                        n_mqi = d_vars[i, q_idx, m_idx]
                        sum_requests = pl.lpSum([a_vars[i, u_idx, q_idx] for u_idx in range(n_users)])
                        
                        operational = delta_t * C_i * (p_idle * n_mqi + beta * sum_requests)
                        embedded = C_emb * n_mqi
                        emissions_expr += operational + embedded
            
            prob += emissions_expr <= budget, "budget_constraint"

        # Solve the problem
        solver = pl.PULP_CBC_CMD(msg=0, timeLimit=300)
        status = prob.solve(solver)

        # Extract results
        if status not in [pl.LpStatusOptimal, pl.LpStatusNotSolved]:
            print(f"WARNING: PuLP solver status: {pl.LpStatus[status]}")
            return 0.0, 0.0, np.zeros_like(self.a_), np.zeros_like(self.d_), {
                "status": pl.LpStatus[status],
                "objective": 0.0
            }

        # Extract assignment variables
        a = np.zeros((n_intervals, n_users, n_tiers))
        for (i, u_idx, q_idx), var in a_vars.items():
            a[i, u_idx, q_idx] = var.varValue if var.varValue is not None else 0.0

        # Extract deployment variables
        d = np.zeros((n_intervals, n_tiers, n_machines))
        for (i, q_idx, m_idx), var in d_vars.items():
            d[i, q_idx, m_idx] = var.varValue if var.varValue is not None else 0.0

        # Calculate emissions and energy
        emissions = 0.0
        energy = 0.0
        
        for i in emissions_window:
            for q_idx in range(n_tiers):
                for m_idx in range(n_machines):
                    machine = S.machines[S.M[m_idx]]
                    q = S.Q[q_idx]
                    
                    p_idle = machine._idle_power_usage
                    p_peak = machine._power_usage
                    theta = machine.performance[q]
                    C_emb = machine.embedded_carbon
                    
                    if theta > 0:
                        beta = (p_peak - p_idle) / theta
                    else:
                        beta = 0.0
                    
                    C_i = C_hat[i]
                    n_mqi = d[i, q_idx, m_idx]
                    sum_requests = sum(a[i, u_idx, q_idx] for u_idx in range(n_users))
                    
                    # Energy consumption (kWh)
                    energy_component = delta_t * (p_idle * n_mqi + beta * sum_requests)
                    energy += energy_component
                    
                    # Emissions (gCO2e)
                    operational = delta_t * C_i * (p_idle * n_mqi + beta * sum_requests)
                    embedded = C_emb * n_mqi
                    emissions += operational + embedded

        metrics = {
            "status": pl.LpStatus[status],
            "objective": pl.value(prob.objective) if prob.objective is not None else 0.0
        }

        return emissions, energy, a, d, metrics

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
        t0 = time()
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)
        
        em, energy, a, d, lp_metrics = self._solve_lp(
            window, R_hat, C_hat, vps,
            emissions_window=window,
            mode="min_emissions",
            qor_target=qor_target,
            delta_t=delta_t
        )
        
        achieved_qor = self._safe_qor(a, R_hat, vps)
        
        if achieved_qor < qor_target - 1e-4:
            print(f"WARNING: QoR target {qor_target:.4f} not met (achieved {achieved_qor:.4f})")

        metrics = {
            "qor_target": qor_target,
            "qor_achieved": achieved_qor,
            "emissions": em,
            "energy": energy,  # NEW: Total energy consumption in kWh
            "runtime": time() - t0,
            "work": 0.0,
            "mip_gap": 0.0,
            "lp_status": lp_metrics["status"],
            "objective": lp_metrics["objective"]
        }
        
        self.a_[:] = a
        self.d_[:] = d
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
        if emissions_window is None:
            emissions_window = self.scenario.I

        t0 = time()
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        em, energy, a, d, lp_metrics = self._solve_lp(
            window, R_hat, C_hat, vps,
            emissions_window=emissions_window,
            mode="max_qor",
            budget=budget,
            delta_t=delta_t
        )
        
        achieved_qor = self._safe_qor(a, R_hat, vps)

        metrics = {
            "qor_target": achieved_qor,
            "emissions": em,
            "energy": energy,  # NEW: Total energy consumption in kWh
            "runtime": time() - t0,
            "work": 0.0,
            "mip_gap": 0.0,
            "lp_status": lp_metrics["status"],
            "objective": lp_metrics["objective"]
        }
        
        self.a_[:] = a
        self.d_[:] = d
        return metrics


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    scenario = Scenario.from_config(cfg)
    R_hat = scenario.R * 1_000_000
    C_hat = scenario.C * 1_000_000
    print(C_hat.shape)
    print(C_hat[0])
    solver = QtModel(scenario)
    window = scenario.I[:1]
    
    print("\n" + "=" * 60)
    print("PULP SOLVER")
    print("=" * 60)
    
    # Print machine parameters
    print("\nMachine Parameters:")
    print("-" * 60)
    for m_idx, m in enumerate(scenario.M):
        machine = scenario.machines[m]
        print(f"\n{machine.name}:")
        print(f"  Idle Power  : {machine._idle_power_usage:.4f} kW")
        print(f"  Peak Power  : {machine._power_usage:.4f} kW")
        print(f"  Emb. Carbon : {machine.embedded_carbon:.2f} gCO2e/h")
        print(f"  Performance (rps):")
        for q in scenario.Q:
            theta = machine.performance[q]
            if theta > 0:
                beta = (machine._power_usage - machine._idle_power_usage) / theta
                print(f"    {q}: {theta:>6.0f} req/hour, β = {beta*1e6:.4f} mW/req")
    
    # Minimize emissions
    print("\n" + "=" * 60)
    print("1. MINIMIZE EMISSIONS")
    print("-" * 50)
    min_res = solver.minimize_emissions(qor_target=0.1, window=window, R_hat=R_hat, C_hat=C_hat)
    print(f" Runtime       : {min_res['runtime']:.3f} s")
    print(f" LP Status     : {min_res['lp_status']}")
    print(f" Emissions     : {min_res['emissions']:_.0f} gCO₂e")
    print(f" Energy        : {min_res['energy']:_.2f} kWh")  # NEW
    print(f" Achieved QoR  : {min_res['qor_achieved']:.4f}")
    _print_deployment(solver.d_, scenario, window)
    
    print("\nDone!")