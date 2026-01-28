import pandas as pd
import matplotlib.pyplot as plt

df_speed = pd.read_csv("speed_results.csv")

plt.figure(figsize=(10, 6))
plt.plot(df_speed["Stifle"], df_speed["Steps"], marker='o', linestyle='-', color='crimson', linewidth=2)

plt.yscale('log')

for x, y in zip(df_speed["Stifle"], df_speed["Steps"]):
    plt.text(x, y * 1.1, f'{y:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Rumor Lifespan: Logarithmic Time to Extinction", fontsize=14)
plt.xlabel("Stifle Chance (%)", fontsize=12)
plt.ylabel("Average Steps (Log Scale)", fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.5)

plt.savefig("rumor_speed_log_plot.png", dpi=300)
