import os
import multiprocessing
import mesa
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class Person(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.state = 0  # 0: Ignorant, 1: Spreader, 2: Stifler

    def step(self):
        if self.state == 1:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            if neighbors:
                other = self.random.choice(neighbors)
                if other.state == 0:
                    if self.random.random() < (self.model.transmission_chance / 100):
                        other.state = 1
                elif other.state > 0:
                    if self.random.random() < (self.model.stifle_chance / 100):
                        self.state = 2

class RumorModel(mesa.Model):
    def __init__(self, width, height, transmission_chance, stifle_chance):
        super().__init__()
        self.grid = mesa.space.SingleGrid(width, height, torus=True)
        self.transmission_chance = transmission_chance
        self.stifle_chance = stifle_chance
        self.running = True
        
        for x in range(width):
            for y in range(height):
                a = Person(self)
                self.grid.place_agent(a, (x, y))
        
        # Start with 5 spreaders
        initial_spreaders = self.random.sample(list(self.agents), 5)
        for a in initial_spreaders:
            a.state = 1

    def count_state(self, state_val):
        return sum(1 for agent in self.agents if agent.state == state_val)

    def step(self):
        self.agents.shuffle_do("step")
        if self.count_state(1) == 0:
            self.running = False


def run_speed_sim(s_val):
    model = RumorModel(40, 40, 100, s_val)
    steps = 0
    while model.running and steps < 2000:
        model.step()
        steps += 1
    return steps

def run_sensitivity_sim(t_val, s_val):
    model = RumorModel(40, 40, t_val, s_val)
    while model.running:
        model.step()
    return (1600 - model.count_state(0))

if __name__ == "__main__":
    CORES = 20
    
    print("Running Phase 1: Speed Analysis...")
    stifle_vals = [0, 25, 50, 75, 100]
    speed_data = []
    for s in stifle_vals:
        res = Parallel(n_jobs=CORES)(delayed(run_speed_sim)(s) for _ in range(500))
        speed_data.append({"Stifle": s, "Steps": np.mean(res)})
    pd.DataFrame(speed_data).to_csv("speed_results.csv")

    print("Running Phase 2: Generating Spatial Maps...")
    for s in [25, 75]:
        model = RumorModel(40, 40, 100, s)
        while model.running: model.step()
        grid_matrix = np.zeros((40, 40))
        for agent, (x, y) in model.grid.coord_iter():
            grid_matrix[x][y] = agent.state
        plt.figure(figsize=(6,4))
        sns.heatmap(grid_matrix, cmap="YlGnBu", cbar=False)
        plt.title(f"Final State Stifle {s}%")
        plt.savefig(f"spatial_{s}.png")

    print("Running Phase 3: Sensitivity Heatmap...")
    t_range = [20, 50, 80, 100]
    s_range = [20, 50, 80, 100]
    hm = np.zeros((4, 4))
    for i, t in enumerate(t_range):
        for j, s in enumerate(s_range):
            res = Parallel(n_jobs=CORES)(delayed(run_sensitivity_sim)(t, s) for _ in range(100))
            hm[i, j] = np.mean(res)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(pd.DataFrame(hm, index=t_range, columns=s_range), annot=True, cmap="Reds")
    plt.savefig("sensitivity_heatmap.png")
    
    print("All analyses complete!")
