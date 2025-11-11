
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import Dict, List
import json
from src.scenario import Scenario



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
                       deployments: List[Dict], scenario: Scenario,
                       qor_target: float):
    """Save final results in multiple formats."""
    
    # 1. Save as JSON
    with open(output_path / "yearly_results.json", 'w') as f:
        json.dump({
            'qor_target': qor_target,
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
    stats['qor_target'] = qor_target
    with open(output_path / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 4. Save aggregates
    if 'emissions' in df_results.columns and df_results['emissions'].notna().any():
        agg_dict = {
            'emissions': ['mean', 'std', 'min', 'max'],
            'energy_kwh': ['mean', 'std', 'min', 'max', 'sum'],
            'qor_achieved': ['mean', 'std', 'min', 'max'],
            'runtime': ['mean', 'max']
        }
        
        df_hourly = df_results.groupby('hour_of_day').agg(agg_dict).round(2)
        df_hourly.to_csv(output_path / "hourly_aggregates.csv")
        
        df_daily = df_results.groupby('day_of_year').agg({
            'emissions': ['sum', 'mean'],
            'energy_kwh': ['sum', 'mean'],
            'qor_achieved': ['mean', 'min'],
            'runtime': 'sum'
        }).round(2)
        df_daily.to_csv(output_path / "daily_aggregates.csv")
        
        df_monthly = df_results.groupby('month').agg({
            'emissions': ['sum', 'mean'],
            'energy_kwh': ['sum', 'mean'],
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
    energy_kwh = [r['energy_kwh'] for r in valid_results]
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
        'energy': {
            'total_kwh': sum(energy_kwh),
            'total_mwh': sum(energy_kwh) / 1000,
            'total_gwh': sum(energy_kwh) / 1_000_000,
            'mean_kwh': np.mean(energy_kwh),
            'median_kwh': np.median(energy_kwh),
            'std_kwh': np.std(energy_kwh),
            'min_kwh': min(energy_kwh),
            'max_kwh': max(energy_kwh),
            'percentiles': {
                'p25': np.percentile(energy_kwh, 25),
                'p75': np.percentile(energy_kwh, 75),
                'p90': np.percentile(energy_kwh, 90),
                'p95': np.percentile(energy_kwh, 95),
                'p99': np.percentile(energy_kwh, 99)
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


def print_summary_statistics(results: List[Dict], qor_target: float):
    """Print summary statistics to console."""
    stats = compute_statistics(results)
    
    if 'error' in stats:
        print(f"\nERROR: {stats['error']}")
        return
    
    print("\n" + "=" * 80)
    print(f"SUMMARY FOR QoR TARGET = {qor_target:.2f}")
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
    
    print(f"\nENERGY:")
    print(f"  Total:  {stats['energy']['total_kwh']:,.0f} kWh")
    print(f"         ({stats['energy']['total_mwh']:,.2f} MWh)")
    print(f"         ({stats['energy']['total_gwh']:,.4f} GWh)")
    print(f"  Mean:   {stats['energy']['mean_kwh']:,.0f} kWh/hour")
    print(f"  Median: {stats['energy']['median_kwh']:,.0f} kWh/hour")
    print(f"  Std:    {stats['energy']['std_kwh']:,.0f} kWh")
    print(f"  Range:  [{stats['energy']['min_kwh']:,.0f}, {stats['energy']['max_kwh']:,.0f}] kWh")
    
    print(f"\nQoR:")
    print(f"  Mean:   {stats['qor']['mean']:.4f}")
    print(f"  Median: {stats['qor']['median']:.4f}")
    print(f"  Range:  [{stats['qor']['min']:.4f}, {stats['qor']['max']:.4f}]")
    
    print(f"\nRUNTIME:")
    print(f"  Total:  {timedelta(seconds=int(stats['runtime']['total_seconds']))}")
    print(f"  Mean:   {stats['runtime']['mean_seconds']:.2f}s per hour")
    print("=" * 80)


def create_visualizations(output_path: Path, results: List[Dict], 
                         deployments: List[Dict], qor_target: float):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib
        matplotlib.use('Agg')
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_valid = df[df['emissions'].notna()].copy()
        
        if len(df_valid) == 0:
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Plot 1: Emissions
        axes[0].plot(df_valid['timestamp'], df_valid['emissions'], linewidth=0.5)
        axes[0].set_title(f'Hourly Emissions (2024) - QoR Target = {qor_target:.2f}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Emissions (gCO₂e)')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        # Plot 2: Energy
        axes[1].plot(df_valid['timestamp'], df_valid['energy_kwh'], 
                     linewidth=0.5, color='orange')
        axes[1].set_title(f'Hourly Energy Consumption (2024) - QoR Target = {qor_target:.2f}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Energy (kWh)')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        # Plot 3: QoR
        axes[2].plot(df_valid['timestamp'], df_valid['qor_achieved'], 
                     linewidth=0.5, color='green')
        axes[2].axhline(y=qor_target, color='red', linestyle='--', 
                       label=f'Target = {qor_target:.2f}', linewidth=1)
        axes[2].set_title(f'QoR Achievement (2024) - QoR Target = {qor_target:.2f}', 
                         fontsize=14, fontweight='bold')
        axes[2].set_ylabel('QoR')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        # Plot 4: Machines
        df_deploy = pd.DataFrame(deployments)
        df_deploy['timestamp'] = pd.to_datetime(df_deploy['timestamp'])
        df_deploy_valid = df_deploy[df_deploy['total_machines'] > 0]
        
        axes[3].plot(df_deploy_valid['timestamp'], 
                     df_deploy_valid['total_machines'], 
                     linewidth=0.5, color='purple')
        axes[3].set_title(f'Total Deployed Machines (2024) - QoR Target = {qor_target:.2f}', 
                         fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Number of Machines')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        plt.tight_layout()
        plt.savefig(output_path / 'yearly_overview.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_path / 'yearly_overview.png'}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


def create_comparison_visualizations(base_output_path: Path, qor_targets: List[float]):
    """Create comparison visualizations across all QoR targets."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        # Collect statistics for all QoR targets
        comparison_data = []
        
        for qor_target in qor_targets:
            qor_path = base_output_path / f"qor_{qor_target:.2f}"
            stats_file = qor_path / "statistics.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    if 'error' not in stats:
                        comparison_data.append({
                            'qor_target': qor_target,
                            'total_emissions_tCO2e': stats['emissions']['total_tCO2e'],
                            'mean_emissions': stats['emissions']['mean'],
                            'total_energy_gwh': stats['energy']['total_gwh'],
                            'total_energy_mwh': stats['energy']['total_mwh'],
                            'mean_energy_kwh': stats['energy']['mean_kwh'],
                            'mean_qor': stats['qor']['mean'],
                            'min_qor': stats['qor']['min'],
                            'max_qor': stats['qor']['max'],
                            'total_runtime_hours': stats['runtime']['total_hours']
                        })
        
        if not comparison_data:
            print("No data available for comparison visualizations")
            return
        
        df_comp = pd.DataFrame(comparison_data)
        
        # Create comparison plots (3x2 grid)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot 1: Total emissions vs QoR target
        axes[0, 0].plot(df_comp['qor_target'], df_comp['total_emissions_tCO2e'], 
                       marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('QoR Target', fontsize=12)
        axes[0, 0].set_ylabel('Total Annual Emissions (tCO₂e)', fontsize=12)
        axes[0, 0].set_title('Total Emissions vs QoR Target', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total energy vs QoR target
        axes[0, 1].plot(df_comp['qor_target'], df_comp['total_energy_mwh'], 
                       marker='o', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('QoR Target', fontsize=12)
        axes[0, 1].set_ylabel('Total Annual Energy (MWh)', fontsize=12)
        axes[0, 1].set_title('Total Energy vs QoR Target', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mean hourly emissions vs QoR target
        axes[1, 0].plot(df_comp['qor_target'], df_comp['mean_emissions'], 
                       marker='o', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('QoR Target', fontsize=12)
        axes[1, 0].set_ylabel('Mean Hourly Emissions (gCO₂e)', fontsize=12)
        axes[1, 0].set_title('Mean Hourly Emissions vs QoR Target', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Mean hourly energy vs QoR target
        axes[1, 1].plot(df_comp['qor_target'], df_comp['mean_energy_kwh'], 
                       marker='o', linewidth=2, markersize=8, color='darkorange')
        axes[1, 1].set_xlabel('QoR Target', fontsize=12)
        axes[1, 1].set_ylabel('Mean Hourly Energy (kWh)', fontsize=12)
        axes[1, 1].set_title('Mean Hourly Energy vs QoR Target', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: QoR achievement vs target
        axes[2, 0].plot(df_comp['qor_target'], df_comp['mean_qor'], 
                       marker='o', linewidth=2, markersize=8, label='Mean QoR', color='purple')
        axes[2, 0].fill_between(df_comp['qor_target'], 
                                df_comp['min_qor'], 
                                df_comp['max_qor'], 
                                alpha=0.3, label='Min-Max Range')
        axes[2, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Achievement', linewidth=1)
        axes[2, 0].set_xlabel('QoR Target', fontsize=12)
        axes[2, 0].set_ylabel('QoR Achieved', fontsize=12)
        axes[2, 0].set_title('QoR Achievement vs Target', 
                            fontsize=14, fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Runtime vs QoR target
        axes[2, 1].plot(df_comp['qor_target'], df_comp['total_runtime_hours'], 
                       marker='o', linewidth=2, markersize=8, color='red')
        axes[2, 1].set_xlabel('QoR Target', fontsize=12)
        axes[2, 1].set_ylabel('Total Runtime (hours)', fontsize=12)
        axes[2, 1].set_title('Computational Cost vs QoR Target', 
                            fontsize=14, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_file = base_output_path / 'qor_comparison.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved: {comparison_file}")
        plt.close()
        
        # Create additional plot: Emissions vs Energy (Pareto front)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_comp['total_energy_mwh'], 
                           df_comp['total_emissions_tCO2e'],
                           c=df_comp['qor_target'], 
                           cmap='viridis', 
                           s=200, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=1.5)
        
        # Add labels for each point
        for idx, row in df_comp.iterrows():
            ax.annotate(f"{row['qor_target']:.1f}", 
                       (row['total_energy_mwh'], row['total_emissions_tCO2e']),
                       fontsize=9, ha='center', va='center')
        
        ax.set_xlabel('Total Annual Energy (MWh)', fontsize=12)
        ax.set_ylabel('Total Annual Emissions (tCO₂e)', fontsize=12)
        ax.set_title('Emissions vs Energy Trade-off (QoR Target labeled)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('QoR Target', fontsize=12)
        
        plt.tight_layout()
        pareto_file = base_output_path / 'emissions_vs_energy.png'
        plt.savefig(pareto_file, dpi=300, bbox_inches='tight')
        print(f"Pareto front visualization saved: {pareto_file}")
        plt.close()
        
        # Save comparison data as CSV
        df_comp.to_csv(base_output_path / 'qor_comparison.csv', index=False)
        print(f"Comparison data saved: {base_output_path / 'qor_comparison.csv'}")
        
    except ImportError:
        print("Matplotlib not available, skipping comparison visualizations")
    except Exception as e:
        print(f"Error creating comparison visualizations: {e}")