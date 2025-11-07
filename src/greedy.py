import numpy as np
from time import time
from typing import List, Optional, Dict
from src.scenario import Scenario
from src.util import get_validity_periods


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _cheapest_machine(scenario: Scenario, i: int, q_idx: int, C_hat: np.ndarray) -> int:
    """Find the machine with lowest operational emission cost per request for tier q at interval i."""
    best_m, best_cost = None, np.inf
    for m_idx, m in enumerate(scenario.M):
        perf = scenario.machines[m].performance[scenario.Q[q_idx]]
        if perf <= 0:
            continue
        power = scenario.machines[m].load_independent_power_usage()
        # Only operational emissions
        emissions = power * C_hat[i]
        cost = emissions / perf
        
        if cost < best_cost:
            best_cost, best_m = cost, m_idx
    if best_m is None:
        raise ValueError(f"No valid machine for tier {q_idx} at interval {i}")
    return best_m

def _machine_cost(scenario: Scenario, m_idx: int, i: int, C_hat: np.ndarray) -> float:
    """Emission cost of one machine m at interval i."""
    machine = scenario.machines[scenario.M[m_idx]]
    power = machine.load_independent_power_usage()
    return power * C_hat[i] + machine.embedded_carbon


def _total_emissions(d: np.ndarray, scenario: Scenario, window: List[int], C_hat: np.ndarray) -> float:
    """Total emissions over the window."""
    total = 0.0
    for i in window:
        for q_idx in range(len(scenario.Q)):
            for m_idx in range(len(scenario.M)):
                total += d[i, q_idx, m_idx] * _machine_cost(scenario, m_idx, i, C_hat)
    return total


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
# QtModel - Greedy Implementation (FIXED)
# ======================================================================
class QtModel:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.a_ = np.zeros((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=float)
        self.d_ = np.zeros((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=float)

    # ------------------------------------------------------------------
    # Exact QoR (matches Gurobi)
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
                    
        return 1.0 - (max(errors))

    # ------------------------------------------------------------------
    # Greedy assignment for a QoR target (FIXED)
    # ------------------------------------------------------------------
    def _compute_assignment(
        self,
        qor_target: float,
        window: List[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        vps: List[List[int]],
        emissions_window: Optional[List[int]] = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Greedily assign requests and deploy machines to approx meet QoR target."""
        if emissions_window is None:
            emissions_window = window
        S = self.scenario
        a = np.zeros_like(self.a_)
        d = np.zeros_like(self.d_)
        err_max = 1.0 - qor_target

        # Per validity period
        for vp in vps:
            # 1. Compute weighted average cost per tier across VP intervals
            # Weight by carbon intensity to make cost representative
            cost_per_q = np.zeros(len(S.Q))
            total_carbon = sum(C_hat[i] for i in vp)
            
            for q_idx in range(len(S.Q)):
                weighted_cost = 0.0
                for i in vp:
                    m_idx = _cheapest_machine(S, i, q_idx, C_hat)
                    
                    perf = S.machines[S.M[m_idx]].performance[S.Q[q_idx]]
                    if perf > 0:
                        machine = S.machines[S.M[m_idx]]
                        power = machine.load_independent_power_usage()
                        emissions = power * C_hat[i] + machine.embedded_carbon
                        cost = emissions / perf
                        weighted_cost += cost * C_hat[i]
                cost_per_q[q_idx] = weighted_cost / max(total_carbon, 1e-9)

            # 2. Per-user fraction allocation (greedy: fill cheapest within bounds)
            for u_idx in range(len(S.U)):
                total_R = sum(R_hat[i, u_idx] for i in vp)
                if total_R <= 1e-9:
                    continue

                lo = np.array([S.slo_lower[u_idx, q_idx] for q_idx in range(len(S.Q))])
                hi = np.array([S.slo_upper[u_idx, q_idx] for q_idx in range(len(S.Q))])
                
                # Expand bounds by err_max
                den = np.abs(lo - hi)

                allowed_lo = np.maximum(0.0, hi - err_max * den)
                allowed_hi = np.minimum(1.0, hi + err_max * den)
                # Start with lower bounds
                fracs = allowed_lo.copy()
                remaining = 1.0 - fracs.sum()
                
                if remaining < -1e-9:
                    # Infeasible: sum(lo) > 1 → scale down
                    print(f"WARNING: Infeasible SLO lowers for user {u_idx} in VP {vp}; scaling down.")
                    scale = 1.0 / (fracs.sum() + 1e-12)
                    fracs *= scale
                    remaining = 0.0
                else:
                    # Greedy fill: prioritize CHEAPEST tiers
                    order = np.argsort(cost_per_q)
                    for q_idx in order:
                        if remaining <= 1e-9:
                            break
                        add = min(remaining, allowed_hi[q_idx] - fracs[q_idx])
                        fracs[q_idx] += add
                        remaining -= add

                    if remaining > 1e-6:
                        print(f"WARNING: Infeasible SLO uppers for user {u_idx} in VP {vp}; adding excess to cheapest.")
                        fracs[order[0]] += remaining

                # 3. Distribute fractions over intervals proportionally to demand
                for i in vp:
                    if R_hat[i, u_idx] > 0:
                        a[i, u_idx, :] = fracs * R_hat[i, u_idx]
                        

        # 4. Deploy machines for each interval in window - FIXED LOGIC
        # Use interval-specific cheapest machine, not VP average
        for i in window:
            for q_idx in range(len(S.Q)):
                total_q = a[i, :, q_idx].sum()
                if total_q > 1e-9:
                    # Get the actual cheapest machine for THIS interval
                    m_idx = _cheapest_machine(S, i, q_idx, C_hat)
                    perf = S.machines[S.M[m_idx]].performance[S.Q[q_idx]]
                    if perf > 0:
                        d[i, q_idx, m_idx] = np.ceil(total_q / perf)

        emissions = _total_emissions(d, S, emissions_window, C_hat)
        return emissions, a, d

    # ------------------------------------------------------------------
    # Minimize emissions s.t. QoR >= target (approx)
    # ------------------------------------------------------------------
    def minimize_emissions(
        self,
        qor_target: float,
        window: List[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        past_vps: bool = True,
        future_vps: bool = True,
    ) -> Dict[str, float]:
        t0 = time()
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)
        em, a, d = self._compute_assignment(
            qor_target, window, R_hat, C_hat, vps, emissions_window=window
        )
        achieved_qor = self._safe_qor(a, R_hat, vps)
        if achieved_qor < qor_target - 1e-4:
            print(f"WARNING: QoR target {qor_target:.4f} not met (achieved {achieved_qor:.4f})")

        metrics = {
            "qor_target": qor_target,
            "qor_achieved": achieved_qor,
            "emissions": em,
            "runtime": time() - t0,
            "work": 0.0,
            "mip_gap": 0.0,
        }
        self.a_[:] = a
        self.d_[:] = d
        return metrics

    # ------------------------------------------------------------------
    # Maximize QoR s.t. emissions <= budget (binary search)
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
    ) -> Dict[str, float]:
        if emissions_window is None:
            emissions_window = self.scenario.I

        t0 = time()
        vps = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        lo, hi = 0.0, 1.0
        best_a, best_d = None, None
        best_qor, best_em = 0.0, np.inf
        
        for _ in range(30):  # Binary search for max QoR
            mid = (lo + hi) / 2.0
            em, a, d = self._compute_assignment(
                mid, window, R_hat, C_hat, vps, emissions_window=emissions_window
            )
            if em <= budget:
                lo = mid
                best_qor = self._safe_qor(a, R_hat, vps)
                best_em = em
                best_a, best_d = a.copy(), d.copy()
            else:
                hi = mid

        # Handle case where no solution found
        if best_a is None:
            print("WARNING: No feasible solution found within budget")
            best_a = np.zeros_like(self.a_)
            best_d = np.zeros_like(self.d_)
            best_qor = 0.0
            best_em = 0.0

        metrics = {
            "qor_target": best_qor,
            "emissions": best_em,
            "runtime": time() - t0,
            "work": 0.0,
            "mip_gap": 0.0,
        }
        self.a_[:] = best_a
        self.d_[:] = best_d
        return metrics


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    scenario = Scenario.from_config(cfg)
    R_hat = scenario.R * 10_000
    C_hat = scenario.C * 1_000_000
    solver = QtModel(scenario)
    window = scenario.I[:1]
    print("\n" + "=" * 60)
    print("GREEDY SOLVER")
    print("=" * 60)
    # ---- Minimize emissions (QoR >= 0.5) ----
    print("\n1. MINIMIZE EMISSIONS (QoR >= 0.5)")
    print("-" * 50)
    min_res = solver.minimize_emissions(qor_target=0.6, window=window, R_hat=R_hat, C_hat=C_hat)
    print(f" Runtime : {min_res['runtime']:.3f} s")
    print(f" Emissions : {min_res['emissions']:_.0f} gCO₂e")
    print(f" Achieved QoR : {min_res['qor_achieved']:.4f}")
    _print_deployment(solver.d_, scenario, window)
    # ---- Maximize QoR under +5% budget ----
    budget = min_res["emissions"] * 1.05
    print("\n2. MAXIMIZE QoR (budget = +5%)")
    print("-" * 50)
    max_res = solver.maximize_qor(budget=budget, window=window, R_hat=R_hat, C_hat=C_hat)
    print(f" Runtime : {max_res['runtime']:.3f} s")
    print(f" Emissions : {max_res['emissions']:_.0f} gCO₂e")
    print(f" Achieved QoR : {max_res['qor_target']:.4f}")
    _print_deployment(solver.d_, scenario, window)
    print("\nDone!")