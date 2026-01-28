import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from pathlib import Path

from rumor_model import run_single_simulation

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def dk_ode_system(y, t, beta, gamma):
    """
    Daley-Kendall ODE system (mean-field approximation).
    Parameters:
        y: [x, s, r] where x=ignorant, s=spreader, r=stifler (fractions)
        t: time
        beta: transmission rate
        gamma: stifling rate
    Equations:
        dx/dt = -beta * x * s
        ds/dt = beta * x * s - gamma * s * (s + r)
        dr/dt = gamma * s * (s + r)
    """
    x, s, r = y
    
    dxdt = -beta * x * s
    dsdt = beta * x * s - gamma * s * (s + r)
    drdt = gamma * s * (s + r)
    
    return [dxdt, dsdt, drdt]


def solve_dk_ode(beta=1.0, gamma=1.0, x0=0.99, s0=0.01, t_max=50, n_points=1000):
    t = np.linspace(0, t_max, n_points)
    y0 = [x0, s0, 1 - x0 - s0]  # Initial: mostly ignorant, few spreaders
    
    solution = odeint(dk_ode_system, y0, t, args=(beta, gamma))
    
    return pd.DataFrame({
        'time': t,
        'ignorant': solution[:, 0],
        'spreader': solution[:, 1],
        'stifler': solution[:, 2],
        'reached': 1 - solution[:, 0]
    })


def theoretical_final_ignorant(gamma_over_beta=1.0):
    def equation(x_inf):
        return x_inf - np.exp(-2 * (1 - x_inf) * gamma_over_beta)
    
    # Solve numerically
    x_inf = fsolve(equation, 0.2)[0]
    return x_inf


def compare_theory_simulation(
    n_cores: int = 20,
    iterations: int = 500,
    grid_size: int = 40
) -> pd.DataFrame:
    results = []
    
    stifle_values = [25, 50, 75, 100]
    transmission = 100
    
    population = grid_size ** 2
    
    for stifle in stifle_values:
        print(f"  Running simulations for stifle={stifle}%...")
        
        sim_results = Parallel(n_jobs=n_cores, backend="loky")(
            delayed(run_single_simulation)(
                transmission, stifle, grid_size, grid_size
            ) for _ in range(iterations)
        )
        
        reaches = [r[0] for r in sim_results]
        sim_reach_fraction = np.mean(reaches) / population
        
        gamma_over_beta = stifle / transmission
        theory_ignorant = theoretical_final_ignorant(gamma_over_beta)
        theory_reach = 1 - theory_ignorant
        
        results.append({
            'stifle_chance': stifle,
            'gamma_over_beta': gamma_over_beta,
            'theory_reach_fraction': theory_reach,
            'theory_ignorant_fraction': theory_ignorant,
            'sim_reach_fraction': sim_reach_fraction,
            'sim_ignorant_fraction': 1 - sim_reach_fraction,
            'absolute_error': abs(sim_reach_fraction - theory_reach),
            'relative_error': abs(sim_reach_fraction - theory_reach) / theory_reach * 100
        })
    
    return pd.DataFrame(results)


def plot_theory_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    x = df['stifle_chance']
    
    ax.plot(x, df['theory_reach_fraction'] * 100, 'b-', linewidth=2, 
            label='Theory (Mean-Field)', marker='s', markersize=8)
    ax.plot(x, df['sim_reach_fraction'] * 100, 'r--', linewidth=2,
            label='Simulation (Spatial)', marker='o', markersize=8)
    
    ax.set_xlabel('Stifle Chance (%)', fontsize=12)
    ax.set_ylabel('Population Reached (%)', fontsize=12)
    ax.set_title('Theory vs Simulation: Rumor Reach', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    ax = axes[1]
    ax.bar(x.astype(str), df['relative_error'], color='orange', edgecolor='black')
    ax.set_xlabel('Stifle Chance (%)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Deviation from Mean-Field Theory', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "theory_comparison.png", dpi=300)
    plt.close()


def plot_ode_dynamics() -> None:    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ratios = [0.25, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
    
    ax = axes[0]
    for ratio, color in zip(ratios, colors):
        df = solve_dk_ode(beta=1.0, gamma=ratio, t_max=20)
        ax.plot(df['time'], df['reached'], label=f'γ/β = {ratio}',
               linewidth=2, color=color)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Fraction Reached', fontsize=12)
    ax.set_title('Theoretical DK Model: Reach Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    df = solve_dk_ode(beta=1.0, gamma=1.0, t_max=20)
    
    ax.plot(df['time'], df['ignorant'], 'b-', linewidth=2, label='Ignorant')
    ax.plot(df['time'], df['spreader'], 'r-', linewidth=2, label='Spreader')
    ax.plot(df['time'], df['stifler'], 'g-', linewidth=2, label='Stifler')
    
    ax.axhline(y=0.203, color='blue', linestyle='--', alpha=0.5, 
               label='Theoretical x∞ ≈ 0.203')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Fraction of Population', fontsize=12)
    ax.set_title('DK Model Dynamics (γ/β = 1)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "theory_ode_dynamics.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("THEORETICAL VALIDATION")
    print("=" * 70)
    
    x_inf = theoretical_final_ignorant(1.0)
    print(f"\nTheoretical final ignorant fraction (γ/β=1): {x_inf:.4f}")
    print(f"Theoretical reach: {(1-x_inf)*100:.1f}%")
    
    print("\nGenerating theoretical dynamics plots...")
    plot_ode_dynamics()
    
    print("\nComparing theory with simulations...")
    comparison_df = compare_theory_simulation(n_cores=20, iterations=300)
    comparison_df.to_csv(OUTPUT_DIR / "theory_comparison.csv", index=False)
    print(comparison_df)
    
    plot_theory_comparison(comparison_df)
    
    print(f"\nResults saved to {OUTPUT_DIR.absolute()}")
