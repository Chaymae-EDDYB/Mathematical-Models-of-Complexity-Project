import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
import warnings

from rumor_model import (
    RumorModel, 
    AgentState,
    run_single_simulation,
    run_simulation_with_history
)

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


class AnalysisSuite:
    def __init__(
        self, 
        n_cores: int = 20,
        grid_size: Tuple[int, int] = (40, 40),
        default_iterations: int = 500
    ):
        self.n_cores = n_cores
        self.width, self.height = grid_size
        self.default_iterations = default_iterations
        self.population = self.width * self.height
    
    def run_stochastic_analysis(
        self,
        stifle_values: List[float] = [0, 25, 50, 75, 100],
        transmission: float = 100.0,
        iterations: int = None
    ) -> pd.DataFrame:
        """
        Run stochastic analysis varying stifle chance.
        
        Computes mean, std, confidence intervals, and distribution statistics.
        """
        iterations = iterations or self.default_iterations
        results = []
        raw_data = {}
        
        for stifle in stifle_values:
            print(f"  Stifle {stifle}%: Running {iterations} simulations...")
            
            sim_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(run_single_simulation)(
                    transmission, stifle, self.width, self.height
                ) for _ in range(iterations)
            )
            
            reaches = [r[0] for r in sim_results]
            steps = [r[1] for r in sim_results]
            fractions = [r[2] for r in sim_results]
            
            raw_data[stifle] = {
                'reaches': reaches,
                'steps': steps,
                'fractions': fractions
            }
            
            # Calculate statistics
            mean_reach = np.mean(reaches)
            std_reach = np.std(reaches, ddof=1)
            sem = std_reach / np.sqrt(iterations)
            ci_95 = 1.96 * sem
            
            results.append({
                'stifle_chance': stifle,
                'transmission_chance': transmission,
                'mean_reach': mean_reach,
                'std_reach': std_reach,
                'sem': sem,
                'ci_95_lower': mean_reach - ci_95,
                'ci_95_upper': mean_reach + ci_95,
                'median_reach': np.median(reaches),
                'min_reach': np.min(reaches),
                'max_reach': np.max(reaches),
                'mean_steps': np.mean(steps),
                'std_steps': np.std(steps, ddof=1),
                'reach_fraction': np.mean(fractions),
                'iterations': iterations
            })
        
        df = pd.DataFrame(results)
        self._raw_stochastic_data = raw_data  
        return df
    
    def plot_stochastic_results(
        self, 
        df: pd.DataFrame, 
        save_prefix: str = "stochastic"
    ) -> None:
        """Generate all stochastic analysis plots."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = df['stifle_chance'].astype(str)
        y = df['mean_reach']
        yerr = df['ci_95_upper'] - df['mean_reach']
        
        bars = ax.bar(x, y, yerr=yerr, capsize=8, color='steelblue', 
                      edgecolor='black', alpha=0.8, error_kw={'linewidth': 2})
        
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Mean Reach (Number Informed)', fontsize=12)
        ax.set_title('Stochastic Analysis: Rumor Reach vs Stifling Probability\n(95% Confidence Intervals)', fontsize=14)
        
        for bar, val, std in zip(bars, y, df['std_reach']):
            ax.annotate(f'{val:.0f}±{std:.0f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_reach_ci.png", dpi=300)
        plt.close()
        
        if hasattr(self, '_raw_stochastic_data'):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            plot_data = []
            for stifle, data in self._raw_stochastic_data.items():
                for reach in data['reaches']:
                    plot_data.append({'Stifle Chance (%)': str(stifle), 'Reach': reach})
            
            plot_df = pd.DataFrame(plot_data)
            sns.violinplot(data=plot_df, x='Stifle Chance (%)', y='Reach', 
                          palette='Blues', ax=ax)
            
            ax.set_title('Distribution of Rumor Reach Across Simulations', fontsize=14)
            ax.set_ylabel('Final Reach', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{save_prefix}_distribution.png", dpi=300)
            plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['stifle_chance'], df['reach_fraction'] * 100, 
                marker='o', markersize=10, linewidth=2, color='darkgreen')
        ax.fill_between(df['stifle_chance'], 
                       (df['ci_95_lower'] / self.population) * 100,
                       (df['ci_95_upper'] / self.population) * 100,
                       alpha=0.3, color='green')
        
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Population Reached (%)', fontsize=12)
        ax.set_title('Rumor Penetration Rate', fontsize=14)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_penetration.png", dpi=300)
        plt.close()
    
    def run_speed_analysis(
        self,
        stifle_values: List[float] = [0, 10, 25, 50, 75, 90, 100],
        transmission: float = 100.0,
        iterations: int = None
    ) -> pd.DataFrame:
        """Analyze rumor lifespan vs stifling."""
        iterations = iterations or self.default_iterations
        results = []
        
        for stifle in stifle_values:
            print(f"  Speed analysis - Stifle {stifle}%...")
            
            sim_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                delayed(run_single_simulation)(
                    transmission, stifle, self.width, self.height
                ) for _ in range(iterations)
            )
            
            steps = [r[1] for r in sim_results]
            
            results.append({
                'stifle_chance': stifle,
                'mean_steps': np.mean(steps),
                'std_steps': np.std(steps, ddof=1),
                'median_steps': np.median(steps),
                'min_steps': np.min(steps),
                'max_steps': np.max(steps)
            })
        
        return pd.DataFrame(results)
    
    def plot_speed_results(self, df: pd.DataFrame, save_prefix: str = "speed") -> None:        
        # Linear scale
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear
        ax = axes[0]
        ax.errorbar(df['stifle_chance'], df['mean_steps'], 
                   yerr=df['std_steps'], marker='o', markersize=8,
                   linewidth=2, capsize=5, color='crimson')
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Mean Steps to Extinction', fontsize=12)
        ax.set_title('Rumor Lifespan (Linear Scale)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Log scale
        ax = axes[1]
        ax.plot(df['stifle_chance'], df['mean_steps'], 
               marker='o', markersize=8, linewidth=2, color='crimson')
        ax.set_yscale('log')
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Mean Steps (Log Scale)', fontsize=12)
        ax.set_title('Rumor Lifespan (Logarithmic Scale)', fontsize=14)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        
        # Add annotations
        for x, y in zip(df['stifle_chance'], df['mean_steps']):
            ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_lifespan.png", dpi=300)
        plt.close()
    
    def run_sensitivity_analysis(
        self,
        transmission_values: List[float] = [10, 25, 50, 75, 100],
        stifle_values: List[float] = [10, 25, 50, 75, 100],
        iterations: int = 100
    ) -> pd.DataFrame:
        results = []
        
        total = len(transmission_values) * len(stifle_values)
        current = 0
        
        for t in transmission_values:
            for s in stifle_values:
                current += 1
                print(f"  [{current}/{total}] T={t}%, S={s}%...")
                
                sim_results = Parallel(n_jobs=self.n_cores, backend="loky")(
                    delayed(run_single_simulation)(
                        t, s, self.width, self.height
                    ) for _ in range(iterations)
                )
                
                reaches = [r[0] for r in sim_results]
                steps = [r[1] for r in sim_results]
                
                results.append({
                    'transmission': t,
                    'stifle': s,
                    'mean_reach': np.mean(reaches),
                    'std_reach': np.std(reaches),
                    'mean_steps': np.mean(steps)
                })
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_heatmap(
        self, 
        df: pd.DataFrame, 
        save_prefix: str = "sensitivity"
    ) -> None:
        """Generate sensitivity analysis heatmaps."""
        
        # Reach heatmap
        pivot_reach = df.pivot(index='transmission', columns='stifle', values='mean_reach')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Mean Reach
        ax = axes[0]
        sns.heatmap(pivot_reach, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   ax=ax, cbar_kws={'label': 'Mean Reach'})
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Transmission Chance (%)', fontsize=12)
        ax.set_title('Phase Diagram: Mean Rumor Reach', fontsize=14)
        
        # Mean Steps
        pivot_steps = df.pivot(index='transmission', columns='stifle', values='mean_steps')
        ax = axes[1]
        sns.heatmap(pivot_steps, annot=True, fmt='.0f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Mean Steps'})
        ax.set_xlabel('Stifle Chance (%)', fontsize=12)
        ax.set_ylabel('Transmission Chance (%)', fontsize=12)
        ax.set_title('Phase Diagram: Rumor Lifespan', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_heatmap.png", dpi=300)
        plt.close()
    
    def generate_spatial_comparison(
        self,
        stifle_values: List[float] = [0, 25, 50, 75, 100],
        transmission: float = 100.0,
        seed: int = 42
    ) -> None:
        """Generate spatial pattern visualizations for different stifle values."""
        
        n_plots = len(stifle_values)
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green'])
        
        for ax, stifle in zip(axes, stifle_values):
            model = RumorModel(
                width=self.width,
                height=self.height,
                transmission_chance=transmission,
                stifle_chance=stifle,
                seed=seed
            )
            model.run()
            
            matrix = model.get_grid_state_matrix()
            
            im = ax.imshow(matrix.T, cmap=cmap, vmin=0, vmax=2, origin='lower')
            ax.set_title(f'Stifle: {stifle}%\nReach: {model.get_final_reach()}/{self.population}',
                        fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Ignorant'),
            Patch(facecolor='red', label='Spreader'),
            Patch(facecolor='green', label='Stifler')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
        
        plt.suptitle('Spatial Distribution of Final States', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "spatial_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, stifle in zip(axes, [25, 75]):
            model = RumorModel(
                width=self.width,
                height=self.height,
                transmission_chance=transmission,
                stifle_chance=stifle,
                seed=seed
            )
            model.run()
            
            times_matrix = model.get_times_heard_matrix()
            
            im = ax.imshow(times_matrix.T, cmap='hot', origin='lower')
            ax.set_title(f'Times Heard (Stifle: {stifle}%)', fontsize=12)
            plt.colorbar(im, ax=ax, label='Times Heard')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "spatial_times_heard.png", dpi=300)
        plt.close()
    
    def analyze_temporal_dynamics(
        self,
        stifle_values: List[float] = [0, 25, 50, 75, 100],
        transmission: float = 100.0,
        seed: int = 42
    ) -> Dict[float, pd.DataFrame]:
        """Analyze how states evolve over time."""
        
        dynamics = {}
        
        for stifle in stifle_values:
            history = run_simulation_with_history(
                transmission, stifle, self.width, self.height, seed=seed
            )
            
            df = pd.DataFrame([{
                'step': h.step,
                'ignorant': h.n_ignorant,
                'spreader': h.n_spreader,
                'stifler': h.n_stifler,
                'reach': h.reach
            } for h in history])
            
            dynamics[stifle] = df
        
        return dynamics
    
    def plot_temporal_dynamics(
        self, 
        dynamics: Dict[float, pd.DataFrame],
        save_prefix: str = "temporal"
    ) -> None:        
        n_plots = len(dynamics)
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
        if n_plots == 1:
            axes = [axes]
        
        for ax, (stifle, df) in zip(axes, dynamics.items()):
            ax.fill_between(df['step'], 0, df['ignorant'], alpha=0.7, 
                           label='Ignorant', color='lightgray')
            ax.fill_between(df['step'], df['ignorant'], 
                           df['ignorant'] + df['spreader'], alpha=0.7,
                           label='Spreader', color='red')
            ax.fill_between(df['step'], df['ignorant'] + df['spreader'],
                           self.population, alpha=0.7,
                           label='Stifler', color='green')
            
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_title(f'Stifle: {stifle}%', fontsize=11)
            ax.set_xlim(0, df['step'].max())
            ax.set_ylim(0, self.population)
        
        axes[0].set_ylabel('Number of Agents', fontsize=10)
        axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.suptitle('Temporal Evolution of Agent States', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(dynamics)))
        
        for (stifle, df), color in zip(dynamics.items(), colors):
            ax.plot(df['step'], df['spreader'], label=f'Stifle {stifle}%',
                   linewidth=2, color=color)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Number of Active Spreaders', fontsize=12)
        ax.set_title('Spreader Population Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{save_prefix}_spreaders.png", dpi=300)
        plt.close()
    
    def run_statistical_tests(self) -> pd.DataFrame:        
        if not hasattr(self, '_raw_stochastic_data'):
            print("Run stochastic analysis first!")
            return None
        
        results = []
        stifle_values = list(self._raw_stochastic_data.keys())
        
        for i, s1 in enumerate(stifle_values):
            for s2 in stifle_values[i+1:]:
                data1 = self._raw_stochastic_data[s1]['reaches']
                data2 = self._raw_stochastic_data[s2]['reaches']
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                    (np.std(data1)**2 + np.std(data2)**2) / 2
                )  
                
                results.append({
                    'comparison': f'{s1}% vs {s2}%',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'cohens_d': effect_size
                })
        
        all_groups = [self._raw_stochastic_data[s]['reaches'] for s in stifle_values]
        f_stat, anova_p = stats.f_oneway(*all_groups)
        
        print(f"\nANOVA Results: F={f_stat:.2f}, p={anova_p:.2e}")
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running analysis...")
    
    suite = AnalysisSuite(n_cores=20, grid_size=(40, 40), default_iterations=500)
    
    print("\n[1/5] Stochastic Analysis...")
    stochastic_df = suite.run_stochastic_analysis()
    stochastic_df.to_csv(OUTPUT_DIR / "stochastic_results.csv", index=False)
    suite.plot_stochastic_results(stochastic_df)
    print(stochastic_df[['stifle_chance', 'mean_reach', 'std_reach', 'reach_fraction']])
    
    # 2. Speed Analysis
    print("\n[2/5] Running Speed Analysis...")
    speed_df = suite.run_speed_analysis()
    speed_df.to_csv(OUTPUT_DIR / "speed_results.csv", index=False)
    suite.plot_speed_results(speed_df)
    
    # 3. Sensitivity Analysis
    print("\n[3/5] Running Sensitivity Analysis...")
    sensitivity_df = suite.run_sensitivity_analysis(iterations=100)
    sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False)
    suite.plot_sensitivity_heatmap(sensitivity_df)
    
    # 4. Spatial Analysis
    print("\n[4/5] Generating Spatial Visualizations...")
    suite.generate_spatial_comparison()
    
    # 5. Temporal Dynamics
    print("\n[5/5] Analyzing Temporal Dynamics...")
    dynamics = suite.analyze_temporal_dynamics()
    suite.plot_temporal_dynamics(dynamics)
    
    # Statistical Tests
    print("\n[BONUS] Statistical Tests...")
    stats_df = suite.run_statistical_tests()
    if stats_df is not None:
        stats_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
        print(stats_df)
    
    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE! Results saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 70)
