# Multi-Tier LLM Routing

## Overview
Multi-Tier LLM Routing explores how to dynamically assign large language model (LLM) requests across a hierarchy of model qualities and hardware platforms while managing energy usage and greenhouse-gas emissions. The project evaluates year-long operating scenarios for cloud-scale inference clusters, balancing user-specific service level objectives (SLOs) against the carbon intensity of electricity grids. It is designed to accompany research on sustainable, quality-aware routing for LLM services.

## Methodology
The study combines three data streams—request demand traces, carbon-intensity signals, and hardware performance catalogs—to build synthetic but reproducible deployment scenarios. Request traces are rescaled per user cohort, and regional carbon intensities are forecast using bootstrapped historical error distributions to capture uncertainty. Each scenario defines user cohorts with quality-of-response (QoR) bounds, available model tiers, and deployable machine types.

For a given scenario, the optimization layer formulates hourly mixed-integer linear programs that minimize operational plus embedded emissions subject to demand satisfaction, QoR guarantees, capacity constraints, and machine availability. QoR is computed by aggregating allocations over configurable validity periods that mirror the study’s SLO definitions. The solver also tracks energy consumption implied by the deployment schedule. Year-long sweeps are executed across multiple QoR targets to trace the Pareto frontier between service quality and environmental impact.

## Architecture
The repository is organized around four conceptual subsystems:

1. **Configuration and Data Layer** – Hydra-driven configuration artifacts describe the scenario seed, user cohorts, machine catalog, and baseline QoR targets, enabling reproducible experiments and variant generation. The bundled data assets provide request traces, carbon signals, and auxiliary metadata consumed during scenario construction.
2. **Scenario Builder** – Scenario objects assemble all numerical inputs, normalize units for numerical stability, and expose convenience sets for downstream optimization. Machine abstractions encapsulate throughput, idle/peak power, and embedded carbon models that inform both energy and emissions calculations.
3. **Optimization Engine** – A linear-programming model encoded in PuLP produces allocation, deployment, and utilization decisions. It supports objectives that minimize emissions or maximize QoR and reports detailed metrics—including QoR achieved, emissions, and energy use—for each hour solved.
4. **Experiment Orchestrator and Analytics** – The orchestration layer distributes hourly optimizations across CPU workers, checkpoints intermediate results, and aggregates deployments, emissions, and QoR statistics. Post-processing routines persist structured datasets (JSON, CSV) and generate summary visualizations for analysis and publication.

These components communicate through lightweight NumPy arrays and pandas data frames, allowing the research workflow to scale to year-long horizons while remaining transparent for auditing and replication.

## Usage
1. **Environment setup** – Create a Python 3.10+ environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure experiments** – Adjust the Hydra configuration (see the `config/` directory) or override parameters via command-line arguments to explore different regions, request datasets, or machine catalogs. Custom QoR sweeps can be provided to the optimization driver defined in `src/main.py`.
3. **Run optimizations** – Execute the annual optimization sweep from the repository root:
   ```bash
   python src/main.py
   ```
   The script initializes the configured scenario, distributes hourly solves across available CPU cores, and saves results under `results/`.
4. **Analyze outputs** – Inspect the generated JSON/CSV summaries and visualization assets to compare emissions, energy consumption, and QoR trade-offs across QoR targets. These artifacts are suitable for inclusion in research figures or further statistical analysis.

This documentation targets researchers who need to understand the high-level workflow, replicate experiments, or extend the methodology for alternative routing policies and sustainability studies.
