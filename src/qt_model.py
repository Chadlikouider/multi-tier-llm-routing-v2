from time import time
from typing import Optional

import numpy as np
import pulp as pl

from src.scenario import Scenario
from src.util import get_validity_periods


class QtModel:

    def __init__(self, scenario: Scenario, silent: bool = False):
        self.scenario = scenario

        self.a_ = np.empty((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=object)
        self.d_ = np.empty((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=object)

    def minimize_emissions(self,
                           qor_target,
                           window,
                           R_hat,
                           C_hat,
                           past_vps: bool = True,  # for outlooks
                           future_vps: bool = True,
                           relax: bool = False):
        """Optimizes for the lowest budget that satisfies the QoR constraint."""
        t0 = time()
        self.model = pl.LpProblem("QtModel", sense=pl.LpMinimize)

        _init_variables(self.model, self.a_, self.d_, self.scenario, R_hat, window=window, relax=relax)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set QoR constraint
        qor, _ = _qor(self.scenario.slo_lower, self.scenario.slo_upper, validity_periods, self.a_, self.scenario, self.model, R=R_hat)
        self.model += qor >= qor_target, "qor_constr"

        # Configure model for budget optimization
        emissions = _emissions(self.scenario, self.d_, window=window, C_hat=C_hat)
        self.model.setObjective(emissions)

        # Optimize
        self._optimize()
        if self.model.status != pl.LpStatusOptimal:
            # We are always expecting a solution
            raise RuntimeError("This is bad")

        metrics = self._collect_metrics()
        metrics["qor_target"] = qor.value()
        metrics["emissions"] = emissions.value()
        metrics["runtime"] = time() - t0
        return metrics

    def maximize_qor(self,
                     budget,
                     window,
                     R_hat,
                     C_hat,
                     past_vps: bool = True,
                     future_vps: bool = True,
                     relax: bool = False,
                     emissions_window: list[int] = None) -> dict:
        """Maximised the lowest QoR among the provided validity_periods under a budget."""
        if emissions_window is None:
            emissions_window = self.scenario.I

        t0 = time()
        self.model = pl.LpProblem("QtModel", sense=pl.LpMinimize)  # We minimize highest_error to maximize qor

        _init_variables(self.model, self.a_, self.d_, self.scenario, R_hat, window=window, relax=relax)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set budget constraint
        emissions = _emissions(self.scenario, self.d_, window=emissions_window, C_hat=C_hat)
        self.model += emissions <= budget, "budget_constraint"

        # Configure model for QoR optimization (by minimizing highest_error)
        _, highest_error = _qor(self.scenario.slo_lower, self.scenario.slo_upper, validity_periods, self.a_, self.scenario, self.model, R=R_hat)
        self.model.setObjective(highest_error)

        # Optimize
        self._optimize()
        if self.model.status != pl.LpStatusOptimal:
            # We are always expecting a solution
            raise RuntimeError("This is bad")

        qor = 1 - highest_error.value() if highest_error else 1  # Note: 1 - highest_error if errors else 1 to match original logic
        metrics = self._collect_metrics()
        metrics["qor_target"] = qor
        metrics["emissions"] = emissions.value()
        metrics["runtime"] = time() - t0
        return metrics

    def _optimize(self):
        # PuLP does not support callbacks directly, so ignoring callback
        self.model.solve()

    def _collect_metrics(self):
        metrics = {
            # PuLP doesn't have direct equivalents to these, so approximating
            "work": None,  # PuLP doesn't track this
            "mip_gap": None,  # PuLP may not have this directly, but could calculate if needed
        }
        return metrics


def _init_variables(model, a_, d_, scenario, R_hat, window: Optional[list[int]] = None, relax: bool = False, add_constraints=True):
    if window is None:
        window = scenario.I
    for i in window:
        for u in scenario.U:
            for q in scenario.Q:
                var = pl.LpVariable(f"a_{i}_{u}_{q}", lowBound=0, upBound=R_hat[i, u], cat='Continuous')
                a_[i, u, q] = var
        for q in scenario.Q:
            for m in scenario.M:
                if scenario.machines[m].performance[q] > 0:  # only introduce variable if level/machine combination is valid
                    cat = 'Continuous' if relax else 'Integer'
                    var = pl.LpVariable(f"d_{i}_{q}_{m}", lowBound=0, cat=cat)
                    d_[i, q, m] = var
                else:
                    d_[i, q, m] = None
        if add_constraints:
            # Add request allocation constraints
            request_constraints = _constraint_request_allocation(scenario, a_, i, R=R_hat)
            for constr in request_constraints:  # Iterate over the constraints
                model += constr, f"request_allocation_{i}_{constr}"
            # Add sufficient resources constraints
            sufficient_constraints = _constraint_sufficient_resources(scenario, a_, d_, i)
            for constr in sufficient_constraints:  # Iterate over the constraints
                model += constr, f"sufficient_resources_{i}_{constr}"


def _emissions(scenario: Scenario, d_, window, C_hat):
    """Returns the emissions in gCO2e"""
    return sum(interval_emissions(i, q, m, d_, scenario, C_hat) for i in window for q in scenario.Q for m in scenario.M)


def interval_emissions(i, q, m, d_, scenario: Scenario, C_hat, load_dependent=False, a_=None):
    if d_[i, q, m] is None:
        return 0
    p = interval_power_per_machine(i, q, m, d_, scenario, load_dependent=load_dependent, a_=a_)
    return d_[i, q, m] * (p * C_hat[i] + scenario.machines[m].embedded_carbon)


def interval_power_per_machine(i, q, m, d_, scenario: Scenario, load_dependent=False, a_=None) -> float:
    if d_[i, q, m] is None:
        return 0
    if load_dependent:
        # Note: sum over scenario.U, assume a_ is provided
        allocated = sum(a_[i, u, q] for u in scenario.U)
        total_capacity = sum(d_[i, q, m_] * scenario.K[q, m_] for m_ in scenario.M if d_[i, q, m_] is not None)
        util_ = allocated / total_capacity
        return scenario.machines[m].load_dependent_power_usage(q, util_)
    else:
        return scenario.machines[m].load_independent_power_usage()


def _qor(slo_lower, slo_upper, validity_periods, a_, scenario, model, R):
    errors = []

    for vp_i, validity_period in enumerate(validity_periods):
        for u in scenario.U:
            if R[validity_period, u].sum() == 0:
                continue
            for q in scenario.Q:
                # SLI
                lb = min(slo_upper[u][q], slo_lower[u][q])
                ub = max(slo_upper[u][q], slo_lower[u][q])
                sli = pl.LpVariable(f"sli_{vp_i}_{u}_{q}", lowBound=lb, upBound=ub)
                model += sli == sum(a_[j, u, q] for j in validity_period) / R[validity_period, u].sum()

                # Errors
                denominator = abs(slo_lower[u][q] - slo_upper[u][q])
                if denominator == 0:
                    continue
                diff = pl.LpVariable(f"diff_{vp_i}_{u}_{q}", lowBound=-1, upBound=1)
                model += diff == (sli - slo_upper[u][q]) / denominator

                error = pl.LpVariable(f"error_{vp_i}_{u}_{q}", lowBound=0, upBound=1)
                errors.append(error)
                model += diff - error <= 0  # error >= diff
                model += -diff - error <= 0  # error >= -diff

    if len(errors) > 0:
        highest_error = pl.LpVariable("highest_error", lowBound=0, upBound=1)
        for error in errors:
            model += error - highest_error <= 0  # highest_error >= error for each
        qor = 1 - highest_error
    else:
        highest_error = None
        qor = 1
    return qor, highest_error


def _constraint_sufficient_resources(scenario: Scenario, a_, d_, i: int):
    """Ensure deployment can serve all requests for each level"""
    constr = []
    for q in scenario.Q:
        allocated_requests = sum(a_[i, u, q] for u in scenario.U)
        servable_requests = sum(d_[i, q, m] * scenario.machines[m].performance[q] for m in scenario.M if d_[i, q, m] is not None)
        constr.append(allocated_requests <= servable_requests)
    return constr  # Return flat list of constraints


def _constraint_no_wasted_resources(scenario: Scenario, model, a_, d_, i: int):
    # TODO assumes only one machine
    constr = []
    for q in scenario.Q:
        allocated_requests = sum(a_[i, u, q] for u in scenario.U)
        servable_requests = sum(d_[i, q, m] * scenario.machines[m].performance[q] for m in scenario.M if d_[i, q, m] is not None)
        model += servable_requests - allocated_requests <= scenario.machines[0].performance[q], f"constraint_no_wasted_resources_{i}_{q}"
    return constr


def _constraint_request_allocation(scenario, a_, i: int, R):
    """Ensure all requests are allocated to a level"""
    constr = []
    for u in scenario.U:
        constr.append(sum(a_[i, u, q] for q in scenario.Q) == R[i, u])
    return constr  # Return flat list of constraints

