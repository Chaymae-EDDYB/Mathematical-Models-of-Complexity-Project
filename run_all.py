import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='Rumor Spreading Analysis Suite')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test with fewer iterations')
    parser.add_argument('--cores', type=int, default=20,
                       help='Number of CPU cores to use')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['all', 'stochastic', 'speed', 'sensitivity', 
                               'spatial', 'temporal', 'theory', 'animation'],
                       help='Which analysis to run')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    
    args = parser.parse_args()
    
    if args.quick:
        iterations = 50
        sensitivity_iters = 20
        print("=" * 70)
        print("QUICK MODE - Reduced iterations for testing")
        print("=" * 70)
    else:
        iterations = 500
        sensitivity_iters = 100
        print("=" * 70)
        print("FULL ANALYSIS MODE")
        print("=" * 70)
    
    start_time = time.time()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if args.analysis in ['all', 'stochastic', 'speed', 'sensitivity', 'spatial', 'temporal']:
        from analysis import AnalysisSuite
        suite = AnalysisSuite(
            n_cores=args.cores,
            grid_size=(40, 40),
            default_iterations=iterations
        )
    
    # 1. Stochastic Analysis
    if args.analysis in ['all', 'stochastic']:
        print("\n" + "=" * 50)
        print("[1] STOCHASTIC ANALYSIS")
        print("=" * 50)
        stochastic_df = suite.run_stochastic_analysis()
        stochastic_df.to_csv(results_dir / "stochastic_results.csv", index=False)
        suite.plot_stochastic_results(stochastic_df)
        print("\nResults:")
        print(stochastic_df[['stifle_chance', 'mean_reach', 'std_reach', 'reach_fraction']].to_string())
    
    # 2. Speed Analysis
    if args.analysis in ['all', 'speed']:
        print("\n" + "=" * 50)
        print("[2] SPEED/LIFESPAN ANALYSIS")
        print("=" * 50)
        speed_df = suite.run_speed_analysis()
        speed_df.to_csv(results_dir / "speed_results.csv", index=False)
        suite.plot_speed_results(speed_df)
        print("\nResults:")
        print(speed_df.to_string())
    
    # 3. Sensitivity Analysis
    if args.analysis in ['all', 'sensitivity']:
        print("\n" + "=" * 50)
        print("[3] SENSITIVITY ANALYSIS (2D Parameter Sweep)")
        print("=" * 50)
        sensitivity_df = suite.run_sensitivity_analysis(iterations=sensitivity_iters)
        sensitivity_df.to_csv(results_dir / "sensitivity_results.csv", index=False)
        suite.plot_sensitivity_heatmap(sensitivity_df)
    
    # 4. Spatial Analysis
    if args.analysis in ['all', 'spatial']:
        print("\n" + "=" * 50)
        print("[4] SPATIAL PATTERN ANALYSIS")
        print("=" * 50)
        suite.generate_spatial_comparison()
    
    # 5. Temporal Dynamics
    if args.analysis in ['all', 'temporal']:
        print("\n" + "=" * 50)
        print("[5] TEMPORAL DYNAMICS ANALYSIS")
        print("=" * 50)
        dynamics = suite.analyze_temporal_dynamics()
        suite.plot_temporal_dynamics(dynamics)
    
    # 6. Theoretical Validation
    if args.analysis in ['all', 'theory']:
        print("\n" + "=" * 50)
        print("[6] THEORETICAL VALIDATION")
        print("=" * 50)
        from theoretical_validation import (
            compare_theory_simulation,
            plot_theory_comparison,
            plot_ode_dynamics,
            theoretical_final_ignorant
        )
        
        x_inf = theoretical_final_ignorant(1.0)
        print(f"Theoretical final ignorant fraction (γ/β=1): {x_inf:.4f}")
        print(f"Theoretical reach: {(1-x_inf)*100:.1f}%")
        
        plot_ode_dynamics()
        comparison_df = compare_theory_simulation(
            n_cores=args.cores, 
            iterations=iterations // 2
        )
        comparison_df.to_csv(results_dir / "theory_comparison.csv", index=False)
        plot_theory_comparison(comparison_df)
        print("\nTheory vs Simulation Comparison:")
        print(comparison_df.to_string())
    
    # 7. Statistical Tests
    if args.analysis in ['all', 'stochastic']:
        print("\n" + "=" * 50)
        print("[7] STATISTICAL TESTS")
        print("=" * 50)
        stats_df = suite.run_statistical_tests()
        if stats_df is not None:
            stats_df.to_csv(results_dir / "statistical_tests.csv", index=False)
            print(stats_df.to_string())
    
    # 8. Animations
    if args.analysis in ['all', 'animation'] and not args.no_animation:
        print("\n" + "=" * 50)
        print("[8] GENERATING ANIMATIONS")
        print("=" * 50)
        try:
            from animations import create_spreading_animation, create_comparison_animation
            
            create_spreading_animation(
                stifle_chance=50,
                transmission_chance=100,
                grid_size=40,
                save_path=results_dir / "rumor_spreading.gif"
            )
            
            create_comparison_animation(
                stifle_values=[25, 75],
                grid_size=30,
                save_path=results_dir / "stifle_comparison.gif"
            )
        except Exception as e:
            print(f"Animation generation failed: {e}")
            print("(This is optional - other results are still valid)")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {results_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(results_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
