import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from rumor_model import RumorModel, AgentState


def run_custom_sim(t_chance, s_chance):
    model = RumorModel(40, 40, transmission_chance=t_chance, stifle_chance=s_chance)
    while model.running:
        model.step()
    return model.count_by_state()[AgentState.SPREADER] + model.count_by_state()[AgentState.STIFLER]


if __name__ == "__main__":
    t_values = [20, 40, 60, 80, 100]  # Transmission
    s_values = [20, 40, 60, 80, 100]  # Stifling
    heatmap_matrix = np.zeros((len(t_values), len(s_values)))

    for i, t in enumerate(t_values):
        for j, s in enumerate(s_values):
            results = Parallel(n_jobs=20)(delayed(run_custom_sim)(t, s) for _ in range(100))
            heatmap_matrix[i, j] = np.mean(results)

    plt.figure(figsize=(10, 8))
    df_hm = pd.DataFrame(heatmap_matrix, index=t_values, columns=s_values)
    sns.heatmap(df_hm, annot=True, fmt=".0f", cmap="Reds")
    plt.xlabel("Stifle Chance (%)")
    plt.ylabel("Transmission Chance (%)")
    plt.title("Phase Diagram: Mean Rumor Reach")
    plt.savefig("sensitivity_heatmap.png")
    plt.show()
