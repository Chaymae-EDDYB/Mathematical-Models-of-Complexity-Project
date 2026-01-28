import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis import AnalysisSuite

N_CORES = 20
ITERATIONS = 1000 
SENSITIVITY_ITERATIONS = 200 

OUTPUT_DIR = Path("results_1000iter")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 70)
    print(f"FULL ANALYSIS RUN - {ITERATIONS} ITERATIONS PER CONDITION")
    print("=" * 70)
    print(f"Using {N_CORES} CPU cores")
    print(f"Results will be saved to: {OUTPUT_DIR.absolute()}")
    print()
    
    start_time = time.time()
    
    suite = AnalysisSuite(
        n_cores=N_CORES,
        grid_size=(40, 40),
        default_iterations=ITERATIONS
    )
    
    import analysis
    analysis.OUTPUT_DIR = OUTPUT_DIR
    
    # 1. Stochastic Analysis 
    print("\n" + "=" * 50)
    print(f"[1/4] STOCHASTIC ANALYSIS ({ITERATIONS} iterations)")
    print("=" * 50)
    
    stochastic_df = suite.run_stochastic_analysis(
        stifle_values=[0, 10, 25, 50, 75, 90, 100],  
        transmission=100.0,
        iterations=ITERATIONS
    )
    stochastic_df.to_csv(OUTPUT_DIR / "stochastic_results.csv", index=False)
    suite.plot_stochastic_results(stochastic_df, save_prefix="stochastic")
    
    print("\nStochastic Results:")
    print(stochastic_df[['stifle_chance', 'mean_reach', 'std_reach', 'ci_95_lower', 'ci_95_upper', 'reach_fraction']].to_string())
    
    # 2. Speed Analysis
    print("\n" + "=" * 50)
    print(f"[2/4] SPEED ANALYSIS ({ITERATIONS} iterations)")
    print("=" * 50)
    
    speed_df = suite.run_speed_analysis(
        stifle_values=[0, 10, 25, 50, 75, 90, 100],
        iterations=ITERATIONS
    )
    speed_df.to_csv(OUTPUT_DIR / "speed_results.csv", index=False)
    suite.plot_speed_results(speed_df, save_prefix="speed")
    
    print("\nSpeed Results:")
    print(speed_df.to_string())
    
    # 3. Sensitivity Analysis 
    print("\n" + "=" * 50)
    print(f"[3/4] SENSITIVITY ANALYSIS ({SENSITIVITY_ITERATIONS} iterations per cell)")
    print("=" * 50)
    
    sensitivity_df = suite.run_sensitivity_analysis(
        transmission_values=[10, 25, 50, 75, 100],
        stifle_values=[10, 25, 50, 75, 100],
        iterations=SENSITIVITY_ITERATIONS
    )
    sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False)
    suite.plot_sensitivity_heatmap(sensitivity_df, save_prefix="sensitivity")
    
    # 4. Statistical Tests
    print("\n" + "=" * 50)
    print("[4/4] STATISTICAL TESTS")
    print("=" * 50)
    
    stats_df = suite.run_statistical_tests()
    if stats_df is not None:
        stats_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
        print("\nStatistical Tests:")
        print(stats_df.to_string())
    
    print("\n" + "=" * 50)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 50)
    
    suite.generate_spatial_comparison()
    dynamics = suite.analyze_temporal_dynamics()
    suite.plot_temporal_dynamics(dynamics)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total time: {hours}h {minutes}m ({elapsed:.0f} seconds)")
    print(f"Iterations per condition: {ITERATIONS}")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
