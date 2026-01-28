import os
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mesa
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

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
        available_agents = list(self.agents)
        initial_spreaders = self.random.sample(available_agents, 5)
        for a in initial_spreaders:
            a.state = 1

    def count_state(self, state_val):
        return sum(1 for agent in self.agents if agent.state == state_val)

    def step(self):
        self.agents.shuffle_do("step")
        # Stop if no more spreaders
        if self.count_state(1) == 0:
            self.running = False

def run_single_simulation(s_chance, width, height):
    model = RumorModel(width, height, 100, s_chance)
    # Safety timeout for steps to prevent infinite loops
    for _ in range(2000): 
        if not model.running:
            break
        model.step()
    
    total_pop = width * height
    return total_pop - model.count_state(0)

if __name__ == "__main__":
    WIDTH, HEIGHT = 40, 40
    STIFLE_VALUES = [0, 25, 50, 75, 100]
    ITERATIONS = 1000
    CORES = 20 
    
    all_results = []
    start_total = time.time()

    for s_val in STIFLE_VALUES:
        print(f">>> Processing Stifle Chance: {s_val}%")
        
        batch_reaches = Parallel(n_jobs=CORES, backend="loky")(
            delayed(run_single_simulation)(s_val, WIDTH, HEIGHT) for _ in range(ITERATIONS)
        )
        
        avg_reach = np.mean(batch_reaches)
        all_results.append({"Stifle_Chance": s_val, "Mean_Reach": avg_reach})
        print(f"Done. Avg Reach: {avg_reach:.2f}")

    df = pd.DataFrame(all_results)
    df.to_csv("hpc_rumor_results.csv", index=False)
    print(f"\nSUCCESS! Total Time: {time.time() - start_total:.2f}s")
