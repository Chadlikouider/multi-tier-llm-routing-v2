import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json
from hydra import compose, initialize
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle
from src.scenario import Scenario
from qt_model import QtModel  # Replace with actual module name


def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def optimize_single_hour(args):
    """
    Optimize a single hour. This function will be run in parallel.
    Args:
        args: Tuple of (hour_idx, scenario_pickle, R_hat, C_hat, qor_target, machine_capacities)
    """
    hour_idx, scenario_pickle, R_hat, C_hat, qor_target, machine_capacities = args
    
    # Unpickle scenario in each worker process
    scenario = pickle.loads(scenario_pickle)
    solver = QtModel(scenario)
    
    year_start = datetime(2024, 1, 1, 0, 0, 0)
    # Convert numpy.int64 to Python int
    timestamp = year_start + timedelta(hours=int(hour_idx))
    window = [int(hour_idx)]
    
    try:
        # Minimize emissions
        metrics = solver.minimize_emissions(
            qor_target=qor_target,
            window=window,
            R_hat=R_hat,
            C_hat=C_hat,
            past_vps=True,
            future_vps=True
        )
        
        # Extract machine deployment information per tier and type
        machines_per_tier_type = {}
        total_machines = int(np.sum(solver.d_[int(hour_idx), :, :]))
        
        for q_idx, q in enumerate(scenario.Q):
            for m_idx, m in enumerate(scenario.M):
                machine_count = int(solver.d_[int(hour_idx), q_idx, m_idx])
                tier_key = f"tier_{q}"
                machine_type = f"machine_{m}"
                
                if machine_type not in machines_per_tier_type:
                    machines_per_tier_type[machine_type] = {}
                
                machines_per_tier_type[machine_type][tier_key] = machine_count
        
        return {
            'success': True,
            'hour_idx': int(hour_idx),
            'timestamp': timestamp.isoformat(),
            'total_machines': total_machines,
            'machines_per_tier_type': machines_per_tier_type
        }
        
    except Exception as e:
        return {
            'success': False,
            'hour_idx': int(hour_idx),
            'error': str(e),
            'timestamp': timestamp.isoformat()
        }


def run_sampled_optimization_parallel(
    output_dir: str = "results/sampled_optimization",
    qor_target: float = 0.6,
    n_samples: int = 2000,
    n_workers: int = None,
    random_seed: int = 42
):
    """
    Run optimization in parallel for randomly sampled hours from 2024.
    
    Args:
        output_dir: Directory to save results
        qor_target: Target QoR for minimize_emissions mode
        n_samples: Number of random samples to take (default: 2000)
        n_workers: Number of parallel workers (None = use all CPUs)
        random_seed: Random seed for reproducibility
    """
    # Initialize scenario
    print("Initializing scenario...")
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    
    scenario = Scenario.from_config(cfg)
    R_hat = scenario.R * 1_000_000
    C_hat = scenario.C * 1_000_000
    
    # Get machine capacities for utilization calculation
    machine_capacities = {}
    for m_idx, m in enumerate(scenario.M):
        machine_capacities[f"machine_{m}"] = float(scenario.C[m_idx])
    
    # Pickle scenario for worker processes
    scenario_pickle = pickle.dumps(scenario)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_hours = len(scenario.I)
    
    # Randomly sample hours
    np.random.seed(random_seed)
    n_samples = min(n_samples, total_hours)  # Don't sample more than available
    sampled_hours = np.random.choice(total_hours, size=n_samples, replace=False)
    sampled_hours = sorted(sampled_hours)  # Sort for easier tracking
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave 1 CPU free
    
    print(f"Total hours in 2024: {total_hours}")
    print(f"Sampled hours: {n_samples}")
    print(f"QoR Target: {qor_target}")
    print(f"Parallel workers: {n_workers}")
    print(f"Random seed: {random_seed}")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    
    all_results = []
    start_time = datetime.now()
    
    # Prepare arguments for parallel processing
    # Convert numpy integers to Python int
    args_list = [
        (int(hour_idx), scenario_pickle, R_hat, C_hat, qor_target, machine_capacities)
        for hour_idx in sampled_hours
    ]
    
    # Run optimization in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(optimize_single_hour, args): args[0]
                   for args in args_list}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result['success']:
                all_results.append(result)
                print(f" ✓ Hour {result['hour_idx']:4d} | "
                      f"Total machines: {result['total_machines']:5d} | "
                      f"[{completed}/{n_samples}]")
            else:
                print(f" ✗ Hour {result['hour_idx']:4d} | ERROR: {result['error']}")
            
            # Print progress every 100 samples
            if completed % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / completed
                remaining = n_samples - completed
                eta = timedelta(seconds=int(avg_time * remaining))
                print(f"\n Progress: {completed}/{n_samples} ({100*completed/n_samples:.1f}%)")
                print(f" ETA: {eta}\n")
    
    # Save and analyze results
    print("\n" + "=" * 80)
    print("Optimization complete! Computing metrics...")
    
    # Compute and display metrics
    metrics = compute_machine_utilization_metrics(all_results, scenario, machine_capacities)
    
    # Save results
    save_results(output_path, all_results, metrics, scenario, sampled_hours, random_seed)
    
    # Print summary
    print_summary(metrics)
    
    return metrics


def compute_machine_utilization_metrics(results: List[Dict], scenario: Scenario, 
                                       machine_capacities: Dict) -> Dict:
    """
    Compute percentage utilization of each machine type per tier and average active machines.
    
    Utilization % = (Average machines of this type / Average total machines) * 100
    This shows what percentage of the fleet each machine type represents.
    """
    valid_results = [r for r in results if r.get('success', False)]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    # Initialize counters for each machine type and tier
    machine_tier_counts = {}
    total_machines_per_hour = []
    total_machines_per_tier_per_hour = {}  # Track total per tier
    
    # Get all unique machine types and tiers
    machine_types = set()
    tiers = set()
    
    for result in valid_results:
        for machine_type in result['machines_per_tier_type']:
            machine_types.add(machine_type)
            for tier_key in result['machines_per_tier_type'][machine_type]:
                tiers.add(tier_key)
    
    # Initialize nested dictionary
    for machine_type in machine_types:
        machine_tier_counts[machine_type] = {}
        for tier in tiers:
            machine_tier_counts[machine_type][tier] = []
    
    for tier in tiers:
        total_machines_per_tier_per_hour[tier] = []
    
    # Aggregate across all samples
    for result in valid_results:
        total_machines_per_hour.append(result['total_machines'])
        
        # Calculate total machines per tier for this hour
        tier_totals = {tier: 0 for tier in tiers}
        
        for machine_type in machine_types:
            for tier in tiers:
                count = 0
                if machine_type in result['machines_per_tier_type']:
                    if tier in result['machines_per_tier_type'][machine_type]:
                        count = result['machines_per_tier_type'][machine_type][tier]
                machine_tier_counts[machine_type][tier].append(count)
                tier_totals[tier] += count
        
        for tier in tiers:
            total_machines_per_tier_per_hour[tier].append(tier_totals[tier])
    
    # Compute average machines per tier and type
    avg_machines_per_tier_type = {}
    avg_total_per_tier = {}
    
    for tier in tiers:
        avg_total_per_tier[tier] = float(np.mean(total_machines_per_tier_per_hour[tier]))
    
    for machine_type in machine_types:
        avg_machines_per_tier_type[machine_type] = {}
        for tier in tiers:
            avg_machines_per_tier_type[machine_type][tier] = float(np.mean(
                machine_tier_counts[machine_type][tier]
            ))
    
    # Calculate utilization percentage: (avg machines of this type / avg total in tier) * 100
    utilization_pct_per_tier_type = {}
    for machine_type in machine_types:
        utilization_pct_per_tier_type[machine_type] = {}
        for tier in tiers:
            avg = avg_machines_per_tier_type[machine_type][tier]
            total = avg_total_per_tier[tier]
            
            if total > 0:
                pct = (avg / total) * 100
            else:
                pct = 0.0
            
            utilization_pct_per_tier_type[machine_type][tier] = float(pct)
    
    # Compute average utilization across all tiers for each machine type
    avg_utilization_per_machine = {}
    for machine_type in machine_types:
        tier_utils = [utilization_pct_per_tier_type[machine_type][tier] 
                      for tier in tiers]
        avg_utilization_per_machine[machine_type] = float(np.mean(tier_utils))
    
    # Overall statistics
    avg_active_machines = float(np.mean(total_machines_per_hour))
    std_active_machines = float(np.std(total_machines_per_hour))
    min_active_machines = int(np.min(total_machines_per_hour))
    max_active_machines = int(np.max(total_machines_per_hour))
    
    return {
        'n_samples': int(len(valid_results)),
        'avg_machines_per_tier_type': avg_machines_per_tier_type,
        'avg_total_per_tier': avg_total_per_tier,
        'utilization_percentage_per_tier': utilization_pct_per_tier_type,
        'avg_utilization_per_machine_type': avg_utilization_per_machine,
        'overall_active_machines': {
            'average': avg_active_machines,
            'std': std_active_machines,
            'min': min_active_machines,
            'max': max_active_machines,
            'median': float(np.median(total_machines_per_hour))
        }
    }


def save_results(output_path: Path, results: List[Dict], metrics: Dict,
                 scenario: Scenario, sampled_hours: np.ndarray, random_seed: int):
    """Save results and metrics."""
    
    # Convert metrics to native Python types for JSON serialization
    metrics_native = convert_to_native_types(metrics)
    
    # Save metrics as JSON
    with open(output_path / "machine_utilization_metrics.json", 'w') as f:
        json.dump(metrics_native, f, indent=2)
    
    # Save full results
    with open(output_path / "detailed_results.json", 'w') as f:
        json.dump({
            'results': results,
            'sampled_hours': [int(h) for h in sampled_hours],
            'random_seed': random_seed,
            'scenario_info': {
                'users': convert_to_native_types(scenario.U),
                'tiers': convert_to_native_types(scenario.Q),
                'machines': convert_to_native_types(scenario.M),
                'total_intervals': len(scenario.I)
            }
        }, f, indent=2)
    
    # Save as CSV for easy viewing
    df_results = pd.DataFrame([r for r in results if r.get('success', False)])
    if len(df_results) > 0:
        df_results.to_csv(output_path / "hourly_results.csv", index=False)
    
    # Create utilization table similar to the paper's Table 4
    if 'utilization_percentage_per_tier' in metrics:
        # Prepare data for table
        table_data = []
        
        for machine_type in sorted(metrics['utilization_percentage_per_tier'].keys()):
            row = {'Machine_Type': machine_type}
            
            # Add tier columns
            tier_data = metrics['utilization_percentage_per_tier'][machine_type]
            for tier in sorted(tier_data.keys()):
                row[tier] = tier_data[tier]
            
            # Add average utilization
            row['Avg_Utilization_Pct'] = metrics['avg_utilization_per_machine_type'][machine_type]
            
            table_data.append(row)
        
        df_util = pd.DataFrame(table_data)
        df_util = df_util.round(0)  # Round to nearest integer like in the paper
        df_util.to_csv(output_path / "utilization_table.csv", index=False)
        
        # Also save average machines table
        avg_table_data = []
        for machine_type in sorted(metrics['avg_machines_per_tier_type'].keys()):
            row = {'Machine_Type': machine_type}
            
            tier_data = metrics['avg_machines_per_tier_type'][machine_type]
            for tier in sorted(tier_data.keys()):
                row[tier] = tier_data[tier]
            
            avg_table_data.append(row)
        
        df_avg = pd.DataFrame(avg_table_data)
        df_avg = df_avg.round(1)
        df_avg.to_csv(output_path / "avg_machines_table.csv", index=False)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(metrics: Dict):
    """Print summary metrics to console in a format similar to Table 4."""
    
    if 'error' in metrics:
        print(f"\nERROR: {metrics['error']}")
        return
    
    print("\n" + "=" * 100)
    print("MACHINE UTILIZATION SUMMARY")
    print("=" * 100)
    
    print(f"\nNumber of samples: {metrics['n_samples']}")
    
    print(f"\n{'='*100}")
    print("OVERALL ACTIVE MACHINES:")
    print(f"{'='*100}")
    overall = metrics['overall_active_machines']
    print(f"  Average: {overall['average']:.2f}")
    print(f"  Std Dev: {overall['std']:.2f}")
    print(f"  Median:  {overall['median']:.2f}")
    print(f"  Range:   [{overall['min']}, {overall['max']}]")
    
    print(f"\n{'='*100}")
    print("UTILIZATION PERCENTAGE PER MACHINE TYPE AND TIER:")
    print(f"{'='*100}")
    
    # Get all tiers
    if 'utilization_percentage_per_tier' in metrics and metrics['utilization_percentage_per_tier']:
        first_machine = list(metrics['utilization_percentage_per_tier'].keys())[0]
        tiers = sorted(metrics['utilization_percentage_per_tier'][first_machine].keys())
        
        # Print header
        header = f"{'Machine Type':<25}"
        for tier in tiers:
            header += f" {tier:>12}"
        header += f" {'Avg (%)':>12} {'Avg. Active':>15}"
        print(header)
        print("-" * 100)
        
        # Print rows
        for machine_type in sorted(metrics['utilization_percentage_per_tier'].keys()):
            row = f"{machine_type:<25}"
            
            # Tier utilizations
            for tier in tiers:
                util = metrics['utilization_percentage_per_tier'][machine_type][tier]
                row += f" {util:>11.0f}%"
            
            # Average utilization
            avg_util = metrics['avg_utilization_per_machine_type'][machine_type]
            row += f" {avg_util:>11.0f}%"
            
            # Average active machines (average across tiers)
            avg_machines = np.mean([
                metrics['avg_machines_per_tier_type'][machine_type][tier]
                for tier in tiers
            ])
            row += f" {avg_machines:>15.1f}"
            
            print(row)
    
    print("=" * 100)


if __name__ == "__main__":
    metrics = run_sampled_optimization_parallel(
        output_dir="results/sampled_500_hours",
        qor_target=0.9,
        n_samples=100,
        n_workers=None,  # Use all available CPUs
        random_seed=42
    )