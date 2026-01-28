import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def run_speed_simulation(s_chance, width, height):
    model = RumorModel(width, height, 100, s_chance) # β=100
    steps = 0
    while model.running and steps < 2000:
        model.step()
        steps += 1
    return steps

STIFLE_VALUES = [0, 25, 50, 75, 100]
speed_results = []

for s_val in STIFLE_VALUES:
    times = Parallel(n_jobs=20)(
        delayed(run_speed_simulation)(s_val, 40, 40) for _ in range(1000)
    )
    speed_results.append({"Stifle_Chance": s_val, "Avg_Steps": np.mean(times)})

pd.DataFrame(speed_results).to_csv("rumor_speed.csv", index=False)
