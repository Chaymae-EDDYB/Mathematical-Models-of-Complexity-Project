import os
import multiprocessing
import mesa
import numpy as np
import pandas as pd
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
        self.state = 0 # 0: Ignorant, 1: Spreader, 2: Stifler

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
        
        initial_spreaders = self.random.sample(list(self.agents), 5)
        for a in initial_spreaders:
            a.state = 1

    def step(self):
        self.agents.shuffle_do("step")
        if not any(a.state == 1 for a in self.agents):
            self.running = False

def run_sim(t_val, s_val):
    model = RumorModel(40, 40, t_val, s_val)
    steps = 0
    while model.running and steps < 2000:
        model.step()
        steps += 1
    reach = sum(1 for a in model.agents if a.state != 0)
    return reach, steps

if __name__ == "__main__":
    CORES = 20
    STIFLE_VALS = [0, 25, 50, 75, 100]
    ITERATIONS = 500 
    
    stochastic_data = []
    raw_results = {}

    for s in STIFLE_VALS:
        print(f"Simulation Stifle {s}%...")
        results = Parallel(n_jobs=CORES)(delayed(run_sim)(100, s) for _ in range(ITERATIONS))
        
        reaches = [r[0] for r in results]
        steps = [r[1] for r in results]
        
        raw_results[s] = reaches 
        
        stochastic_data.append({
            "Stifle": s,
            "Mean_Reach": np.mean(reaches),
            "Std_Reach": np.std(reaches),
            "Mean_Steps": np.mean(steps)
        })

    df = pd.DataFrame(stochastic_data)
    df.to_csv("stochastic_rumor_results.csv")

    plt.figure(figsize=(10, 6))
    plt.bar(df["Stifle"].astype(str), df["Mean_Reach"], yerr=df["Std_Reach"], 
            capsize=10, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title("Analyse Stochastique : Reach Moyen ± Écart-type")
    plt.ylabel("Nombre de personnes informées")
    plt.xlabel("Chance de Stifling (%)")
    plt.savefig("stochastic_reach.png")

    plt.figure(figsize=(12, 6))
    for s in [50, 75]: 
        sns.kdeplot(raw_results[s], label=f"Stifle {s}%", fill=True)
    plt.title("Distribution de Densité : Variabilité de la rumeur")
    plt.xlabel("Reach Final")
    plt.legend()
    plt.savefig("stochastic_distribution.png")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Stifle"], df["Mean_Steps"], marker='o', color='red')
    plt.yscale('log')
    plt.title("Lifespan de la rumeur (Echelle Logarithmique)")
    plt.grid(True, which="both", ls="--")
    plt.savefig("stochastic_speed_log.png")
    
    print("Analyses stochastiques terminées.")
