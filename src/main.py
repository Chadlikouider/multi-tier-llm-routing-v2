import numpy as np

from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from hydra import compose, initialize
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle

from src.scenario import Scenario
from qt_model import QtModel  # Replace with actual module name
from src.visual_functions import save_checkpoint, save_final_results, print_summary_statistics, create_visualizations, create_comparison_visualizations

def optimize_single_hour(args):
    """
    Optimize a single hour. This function will be run in parallel.
    
    Args:
        args: Tuple of (hour_idx, scenario_pickle, R_hat, C_hat, qor_target)
    """
    hour_idx, scenario_pickle, R_hat, C_hat, qor_target = args
    
    # Unpickle scenario in each worker process
    scenario = pickle.loads(scenario_pickle)
    solver = QtModel(scenario)
    
    year_start = datetime(2024, 1, 1, 0, 0, 0)
    timestamp = year_start + timedelta(hours=hour_idx)
    window = [hour_idx]
    
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
        
        # Calculate total energy consumption (kWh)
        # Energy = sum over all machines of (power * deployment)
        total_energy_kwh = metrics['energy']
        
        # Add metadata
        metrics['timestamp'] = timestamp.isoformat()
        metrics['hour_of_year'] = hour_idx + 1
        metrics['day_of_year'] = timestamp.timetuple().tm_yday
        metrics['hour_of_day'] = timestamp.hour
        metrics['day_of_week'] = timestamp.strftime('%A')
        metrics['month'] = timestamp.month
        metrics['qor_target'] = qor_target
        metrics['energy_kwh'] = float(total_energy_kwh)  # Add energy consumption
        
        # Deployment summary
        deployment_summary = {
            'timestamp': timestamp.isoformat(),
            'hour_of_year': hour_idx + 1,
            'qor_target': qor_target,
            'total_machines': int(np.sum(solver.d_[hour_idx, :, :])),
            'energy_kwh': float(total_energy_kwh),
            'machines_per_tier': {
                f'tier_{q}': int(np.sum(solver.d_[hour_idx, q_idx, :]))
                for q_idx, q in enumerate(scenario.Q)
            },
            'machines_per_type': {
                str(m): int(np.sum(solver.d_[hour_idx, :, m_idx]))
                for m_idx, m in enumerate(scenario.M)
            }
        }
        
        return {
            'success': True,
            'hour_idx': hour_idx,
            'qor_target': qor_target,
            'metrics': metrics,
            'deployment': deployment_summary
        }
        
    except Exception as e:
        return {
            'success': False,
            'hour_idx': hour_idx,
            'qor_target': qor_target,
            'error': str(e),
            'timestamp': timestamp.isoformat()
        }


def run_yearly_optimization_parallel(
    output_dir: str = "results/yearly_optimization",
    qor_targets: List[float] = None,
    n_workers: int = None,
    batch_size: int = 168,  # 1 week at a time
):
    """
    Run optimization in parallel for each hour of 2024 across multiple QoR targets.
    
    Args:
        output_dir: Directory to save results
        qor_targets: List of target QoR values (default: [0.0, 0.1, ..., 1.0])
        n_workers: Number of parallel workers (None = use all CPUs)
        batch_size: How many hours to process before saving checkpoint
    """
    # Default QoR targets from 0 to 1 in steps of 0.1
    if qor_targets is None:
        qor_targets = np.arange(0.0, 1.1, 0.1).tolist()
    
    # Initialize scenario
    print("Initializing scenario...")
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    scenario = Scenario.from_config(cfg)
    
    R_hat = scenario.R
    C_hat = scenario.C 

    # Pickle scenario for worker processes
    scenario_pickle = pickle.dumps(scenario)
    
    # Create base output directory
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    total_hours = len(scenario.I)
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave 1 CPU free
    
    print(f"Total hours in 2024: {total_hours}")
    print(f"QoR Targets: {qor_targets}")
    print(f"Total optimizations: {total_hours * len(qor_targets)}")
    print(f"Parallel workers: {n_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {base_output_path}")
    print("=" * 80)
    
    overall_start_time = datetime.now()
    
    # Process each QoR target
    for qor_idx, qor_target in enumerate(qor_targets):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING QoR TARGET {qor_idx + 1}/{len(qor_targets)}: {qor_target:.2f}")
        print(f"{'=' * 80}")
        
        # Create output directory for this QoR target
        output_path = base_output_path / f"qor_{qor_target:.2f}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        deployments = []
        start_time = datetime.now()
        
        # Process in batches
        for batch_start in range(0, total_hours, batch_size):
            batch_end = min(batch_start + batch_size, total_hours)
            batch_indices = range(batch_start, batch_end)
            
            print(f"\nProcessing hours {batch_start}-{batch_end-1} ({len(batch_indices)} hours)")
            
            # Prepare arguments for parallel processing
            args_list = [
                (hour_idx, scenario_pickle, R_hat, C_hat, qor_target)
                for hour_idx in batch_indices
            ]
            
            # Run optimization in parallel
            batch_results = []
            batch_deployments = []
            
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
                        batch_results.append(result['metrics'])
                        batch_deployments.append(result['deployment'])
                        print(f"  ✓ Hour {result['hour_idx']:4d} | "
                              f"Emissions: {result['metrics']['emissions']:10,.0f} gCO₂e | "
                              f"Energy: {result['metrics']['energy_kwh']:8,.1f} kWh | "
                              f"QoR: {result['metrics']['qor_achieved']:.4f} | "
                              f"Time: {result['metrics']['runtime']:.2f}s | "
                              f"[{completed}/{len(batch_indices)}]")
                    else:
                        # Log error
                        error_metrics = {
                            'timestamp': result['timestamp'],
                            'hour_of_year': result['hour_idx'] + 1,
                            'qor_target': qor_target,
                            'error': result['error'],
                            'emissions': None,
                            'energy_kwh': None,
                            'qor_achieved': None
                        }
                        batch_results.append(error_metrics)
                        batch_deployments.append({
                            'timestamp': result['timestamp'],
                            'hour_of_year': result['hour_idx'] + 1,
                            'qor_target': qor_target,
                            'total_machines': 0,
                            'energy_kwh': 0,
                            'error': result['error']
                        })
                        print(f"  ✗ Hour {result['hour_idx']:4d} | ERROR: {result['error']}")
            
            # Sort batch results by hour
            batch_results.sort(key=lambda x: x.get('hour_of_year', 0))
            batch_deployments.sort(key=lambda x: x.get('hour_of_year', 0))
            
            all_results.extend(batch_results)
            deployments.extend(batch_deployments)
            
            # Only save checkpoint if we have reached the final hour (8760)
            if batch_end == total_hours:
                print(f"Saving specific checkpoint: checkpoint_hour_{batch_end}.json")
                save_checkpoint(output_path, all_results, deployments, batch_end)
            
            # Print progress
            elapsed = (datetime.now() - start_time).total_seconds()
            hours_completed = len(all_results)
            avg_time_per_hour = elapsed / hours_completed
            remaining_hours = total_hours - hours_completed
            eta_seconds = avg_time_per_hour * remaining_hours
            eta = timedelta(seconds=int(eta_seconds))
            
            print(f"\n  Batch complete!")
            print(f"  Progress: {hours_completed}/{total_hours} ({100*hours_completed/total_hours:.1f}%)")
            print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"  Avg time/hour: {avg_time_per_hour:.2f}s")
            print(f"  ETA: {eta}")
        
        # Save final results for this QoR target
        print(f"\n{'=' * 80}")
        print(f"QoR Target {qor_target:.2f} complete! Saving results...")
        save_final_results(output_path, all_results, deployments, scenario, qor_target)
        
        # Print summary statistics
        print_summary_statistics(all_results, qor_target)
        
        # Generate visualizations
        create_visualizations(output_path, all_results, deployments, qor_target)
        
        qor_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nQoR {qor_target:.2f} total time: {timedelta(seconds=int(qor_elapsed))}")
    
    # Create comparison visualizations across all QoR targets
    print(f"\n{'=' * 80}")
    print("Creating comparison visualizations across all QoR targets...")
    create_comparison_visualizations(base_output_path, qor_targets)
    
    total_elapsed = (datetime.now() - overall_start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print("ALL QoR TARGETS COMPLETE!")
    print(f"Total runtime: {timedelta(seconds=int(total_elapsed))}")
    print(f"Results saved to: {base_output_path}")
    print(f"{'=' * 80}")



if __name__ == "__main__":
    # Define QoR targets from 0 to 1
    #qor_targets = np.arange(0.0, 1.1, 0.1).tolist()  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # Alternative: Custom list of QoR targets
    #qor_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    qor_targets = [0.5]
    # Alternative: Finer granularity
    # qor_targets = np.arange(0.0, 1.05, 0.05).tolist()  # [0.0, 0.05, 0.1, ..., 1.0]
    
    run_yearly_optimization_parallel(
        output_dir="results/2024_qor_sweep",
        qor_targets=qor_targets,
        n_workers=None,  # Use all available CPUs
        batch_size=168   # Process 1 week at a time
    )