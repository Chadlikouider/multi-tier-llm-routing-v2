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
        
        # Add metadata
        metrics['timestamp'] = timestamp.isoformat()
        metrics['hour_of_year'] = hour_idx + 1
        metrics['day_of_year'] = timestamp.timetuple().tm_yday
        metrics['hour_of_day'] = timestamp.hour
        metrics['day_of_week'] = timestamp.strftime('%A')
        metrics['month'] = timestamp.month
        
        # Deployment summary
        deployment_summary = {
            'timestamp': timestamp.isoformat(),
            'hour_of_year': hour_idx + 1,
            'total_machines': int(np.sum(solver.d_[hour_idx, :, :])),
            'machines_per_tier': {
                f'tier_{q}': int(np.sum(solver.d_[hour_idx, q_idx, :]))
                for q_idx, q in enumerate(scenario.Q)
            }
        }
        
        return {
            'success': True,
            'hour_idx': hour_idx,
            'metrics': metrics,
            'deployment': deployment_summary
        }
        
    except Exception as e:
        return {
            'success': False,
            'hour_idx': hour_idx,
            'error': str(e),
            'timestamp': timestamp.isoformat()
        }


def run_yearly_optimization_parallel(
    output_dir: str = "results/yearly_optimization",
    qor_target: float = 0.6,
    n_workers: int = None,
    batch_size: int = 168,  # 1 week at a time
):
    """
    Run optimization in parallel for each hour of 2024.
    
    Args:
        output_dir: Directory to save results
        qor_target: Target QoR for minimize_emissions mode
        n_workers: Number of parallel workers (None = use all CPUs)
        batch_size: How many hours to process before saving checkpoint
    """
    # Initialize scenario
    print("Initializing scenario...")
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    scenario = Scenario.from_config(cfg)
    
    R_hat = scenario.R * 1000
    C_hat = scenario.C * 1_000_000
    
    # Pickle scenario for worker processes
    scenario_pickle = pickle.dumps(scenario)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_hours = len(scenario.I)
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave 1 CPU free
    
    print(f"Total hours in 2024: {total_hours}")
    print(f"QoR Target: {qor_target}")
    print(f"Parallel workers: {n_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    
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
                          f"QoR: {result['metrics']['qor_achieved']:.4f} | "
                          f"Time: {result['metrics']['runtime']:.2f}s | "
                          f"[{completed}/{len(batch_indices)}]")
                else:
                    # Log error
                    error_metrics = {
                        'timestamp': result['timestamp'],
                        'hour_of_year': result['hour_idx'] + 1,
                        'error': result['error'],
                        'emissions': None,
                        'qor_achieved': None
                    }
                    batch_results.append(error_metrics)
                    batch_deployments.append({
                        'timestamp': result['timestamp'],
                        'hour_of_year': result['hour_idx'] + 1,
                        'total_machines': 0,
                        'error': result['error']
                    })
                    print(f"  ✗ Hour {result['hour_idx']:4d} | ERROR: {result['error']}")
        
        # Sort batch results by hour
        batch_results.sort(key=lambda x: x.get('hour_of_year', 0))
        batch_deployments.sort(key=lambda x: x.get('hour_of_year', 0))
        
        all_results.extend(batch_results)
        deployments.extend(batch_deployments)
        
        # Save checkpoint after each batch
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
    
    # Save final results
    print("\n" + "=" * 80)
    print("Optimization complete! Saving final results...")
    save_final_results(output_path, all_results, deployments, scenario)
    
    # Print summary statistics
    print_summary_statistics(all_results)
    
    # Generate visualizations
    create_visualizations(output_path, all_results, deployments)


def save_checkpoint(output_path: Path, results: List[Dict], 
                    deployments: List[Dict], hour_num: int):
    """Save intermediate checkpoint."""
    checkpoint_file = output_path / f"checkpoint_hour_{hour_num}.json"
    
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'results': results,
            'deployments': deployments,
            'last_hour': hour_num
        }, f, indent=2)
    
    print(f"  Checkpoint saved: {checkpoint_file}")


def save_final_results(output_path: Path, results: List[Dict], 
                       deployments: List[Dict], scenario: Scenario):
    """Save final results in multiple formats."""
    
    # 1. Save as JSON
    with open(output_path / "yearly_results.json", 'w') as f:
        json.dump({
            'results': results,
            'deployments': deployments,
            'scenario_info': {
                'users': scenario.U,
                'tiers': scenario.Q,
                'machines': scenario.M,
                'total_intervals': len(scenario.I)
            }
        }, f, indent=2)
    
    # 2. Save as CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path / "yearly_results.csv", index=False)
    
    df_deployments = pd.DataFrame(deployments)
    df_deployments.to_csv(output_path / "yearly_deployments.csv", index=False)
    
    # 3. Save aggregated statistics
    stats = compute_statistics(results)
    with open(output_path / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 4. Save aggregates
    df_hourly = df_results.groupby('hour_of_day').agg({
        'emissions': ['mean', 'std', 'min', 'max'],
        'qor_achieved': ['mean', 'std', 'min', 'max'],
        'runtime': ['mean', 'max']
    }).round(2)
    df_hourly.to_csv(output_path / "hourly_aggregates.csv")
    
    df_daily = df_results.groupby('day_of_year').agg({
        'emissions': ['sum', 'mean'],
        'qor_achieved': ['mean', 'min'],
        'runtime': 'sum'
    }).round(2)
    df_daily.to_csv(output_path / "daily_aggregates.csv")
    
    df_monthly = df_results.groupby('month').agg({
        'emissions': ['sum', 'mean'],
        'qor_achieved': ['mean', 'min'],
        'runtime': 'sum'
    }).round(2)
    df_monthly.to_csv(output_path / "monthly_aggregates.csv")
    
    print(f"\nResults saved to: {output_path}")


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics."""
    valid_results = [r for r in results if r.get('emissions') is not None]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    emissions = [r['emissions'] for r in valid_results]
    qor_values = [r['qor_achieved'] for r in valid_results]
    runtimes = [r['runtime'] for r in valid_results]
    
    return {
        'total_hours': len(results),
        'successful_optimizations': len(valid_results),
        'failed_optimizations': len(results) - len(valid_results),
        'emissions': {
            'total_gCO2e': sum(emissions),
            'total_kgCO2e': sum(emissions) / 1000,
            'total_tCO2e': sum(emissions) / 1_000_000,
            'mean': np.mean(emissions),
            'median': np.median(emissions),
            'std': np.std(emissions),
            'min': min(emissions),
            'max': max(emissions),
            'percentiles': {
                'p25': np.percentile(emissions, 25),
                'p75': np.percentile(emissions, 75),
                'p90': np.percentile(emissions, 90),
                'p95': np.percentile(emissions, 95),
                'p99': np.percentile(emissions, 99)
            }
        },
        'qor': {
            'mean': np.mean(qor_values),
            'median': np.median(qor_values),
            'std': np.std(qor_values),
            'min': min(qor_values),
            'max': max(qor_values),
            'percentiles': {
                'p5': np.percentile(qor_values, 5),
                'p10': np.percentile(qor_values, 10),
                'p25': np.percentile(qor_values, 25),
                'p75': np.percentile(qor_values, 75)
            }
        },
        'runtime': {
            'total_seconds': sum(runtimes),
            'total_hours': sum(runtimes) / 3600,
            'mean_seconds': np.mean(runtimes),
            'median_seconds': np.median(runtimes),
            'max_seconds': max(runtimes),
            'min_seconds': min(runtimes)
        }
    }


def print_summary_statistics(results: List[Dict]):
    """Print summary statistics to console."""
    stats = compute_statistics(results)
    
    if 'error' in stats:
        print(f"\nERROR: {stats['error']}")
        return
    
    print("\n" + "=" * 80)
    print("YEARLY OPTIMIZATION SUMMARY (2024)")
    print("=" * 80)
    print(f"\nTotal hours processed: {stats['total_hours']}")
    print(f"Successful: {stats['successful_optimizations']}")
    print(f"Failed: {stats['failed_optimizations']}")
    
    print(f"\nEMISSIONS:")
    print(f"  Total:  {stats['emissions']['total_gCO2e']:,.0f} gCO₂e")
    print(f"         ({stats['emissions']['total_kgCO2e']:,.2f} kgCO₂e)")
    print(f"         ({stats['emissions']['total_tCO2e']:,.4f} tCO₂e)")
    print(f"  Mean:   {stats['emissions']['mean']:,.0f} gCO₂e/hour")
    print(f"  Median: {stats['emissions']['median']:,.0f} gCO₂e/hour")
    print(f"  Std:    {stats['emissions']['std']:,.0f} gCO₂e")
    print(f"  Range:  [{stats['emissions']['min']:,.0f}, {stats['emissions']['max']:,.0f}] gCO₂e")
    
    print(f"\nQoR:")
    print(f"  Mean:   {stats['qor']['mean']:.4f}")
    print(f"  Median: {stats['qor']['median']:.4f}")
    print(f"  Range:  [{stats['qor']['min']:.4f}, {stats['qor']['max']:.4f}]")
    
    print(f"\nRUNTIME:")
    print(f"  Total:  {timedelta(seconds=int(stats['runtime']['total_seconds']))}")
    print(f"  Mean:   {stats['runtime']['mean_seconds']:.2f}s per hour")
    print("=" * 80)


def create_visualizations(output_path: Path, results: List[Dict], 
                         deployments: List[Dict]):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_valid = df[df['emissions'].notna()].copy()
        
        if len(df_valid) == 0:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        axes[0].plot(df_valid['timestamp'], df_valid['emissions'], linewidth=0.5)
        axes[0].set_title('Hourly Emissions (2024)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Emissions (gCO₂e)')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        axes[1].plot(df_valid['timestamp'], df_valid['qor_achieved'], 
                     linewidth=0.5, color='green')
        axes[1].set_title('QoR Achievement (2024)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('QoR')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        df_deploy = pd.DataFrame(deployments)
        df_deploy['timestamp'] = pd.to_datetime(df_deploy['timestamp'])
        df_deploy_valid = df_deploy[df_deploy['total_machines'] > 0]
        
        axes[2].plot(df_deploy_valid['timestamp'], 
                     df_deploy_valid['total_machines'], 
                     linewidth=0.5, color='purple')
        axes[2].set_title('Total Deployed Machines (2024)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Number of Machines')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        plt.tight_layout()
        plt.savefig(output_path / 'yearly_overview.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_path / 'yearly_overview.png'}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    run_yearly_optimization_parallel(
        output_dir="results/2024_hourly_parallel",
        qor_target=0.6,
        n_workers=None,  # Use all available CPUs
        batch_size=168   # Process 1 week at a time
    )