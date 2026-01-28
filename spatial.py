import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rumor_model import RumorModel


def generate_spatial_plot(s_chance, filename):
    model = RumorModel(40, 40, transmission_chance=100, stifle_chance=s_chance)
    while model.running:
        model.step()
    
    grid_matrix = np.zeros((40, 40))
    for contents, (x, y) in model.grid.coord_iter():
        if contents is not None:
            grid_matrix[x][y] = contents.state

    plt.figure(figsize=(8, 6))
    sns.heatmap(grid_matrix, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)
    plt.title(f"Final Social Landscape (Stifle Chance: {s_chance}%)")
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_spatial_plot(25, "spatial_25.png")
    generate_spatial_plot(75, "spatial_75.png")
