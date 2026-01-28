import os
import multiprocessing
from enum import IntEnum
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mesa
import numpy as np


class AgentState(IntEnum):
    IGNORANT = 0
    SPREADER = 1
    STIFLER = 2


@dataclass
class SimulationMetrics:
    step: int
    n_ignorant: int
    n_spreader: int
    n_stifler: int
    total_heard: int
    
    @property
    def reach(self) -> int:
        return self.n_spreader + self.n_stifler
    
    @property
    def reach_fraction(self) -> float:
        total = self.n_ignorant + self.n_spreader + self.n_stifler
        return self.reach / total if total > 0 else 0.0


class Person(mesa.Agent):
    def __init__(self, unique_id: int, model: "RumorModel"):
        super().__init__(unique_id, model)
        self.state: AgentState = AgentState.IGNORANT
        self.times_heard: int = 0
        self.step_became_spreader: int = -1
        self.step_became_stifler: int = -1
    
    def step(self) -> None:
        if self.state != AgentState.SPREADER:
            return
        
        # Get neighbors based on model's neighborhood type
        neighbors = self.model.grid.get_neighbors(
            self.pos, 
            moore=self.model.use_moore_neighborhood,
            include_center=False
        )
        
        if not neighbors:
            return
        
        # Spreader attempts to tell rumor to a random neighbor
        other = self.random.choice(neighbors)
        
        if other.state == AgentState.IGNORANT:
            # Spreading: Ignorant may become Spreader
            other.times_heard += 1
            if self.random.random() < (self.model.transmission_chance / 100):
                other.state = AgentState.SPREADER
                other.step_became_spreader = self.model.current_step
                
        elif other.state in (AgentState.SPREADER, AgentState.STIFLER):
            # Stifling: Spreader may become Stifler when meeting someone who knows
            if self.random.random() < (self.model.stifle_chance / 100):
                self.state = AgentState.STIFLER
                self.step_became_stifler = self.model.current_step


class RumorModel(mesa.Model):
    """
    Mesa model for simulating rumor spreading with stifling dynamics.
    
    Parameters:
        width: Grid width
        height: Grid height  
        transmission_chance: Probability (0-100) of successful rumor transmission
        stifle_chance: Probability (0-100) of stifling upon meeting informed person
        initial_spreaders: Number of initial spreaders (default: 5)
        use_moore_neighborhood: If True, use 8 neighbors; if False, use 4 (Von Neumann)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        width: int = 40,
        height: int = 40,
        transmission_chance: float = 100.0,
        stifle_chance: float = 50.0,
        initial_spreaders: int = 5,
        use_moore_neighborhood: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(seed=seed)
        
        # Model parameters
        self.width = width
        self.height = height
        self.transmission_chance = transmission_chance
        self.stifle_chance = stifle_chance
        self.initial_spreaders = initial_spreaders
        self.use_moore_neighborhood = use_moore_neighborhood
        
        # Grid setup (torus topology for periodic boundaries)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)
        
        # State tracking
        self.current_step = 0
        self.running = True
        self.history: List[SimulationMetrics] = []
        self.next_id = 0  # Agent ID counter
        
        # Create all agents
        for x in range(width):
            for y in range(height):
                agent = Person(self.next_id, self)
                self.next_id += 1
                self.grid.place_agent(agent, (x, y))
        
        # Initialize spreaders
        if initial_spreaders > 0:
            available_agents = list(self.agents)
            selected = self.random.sample(
                available_agents, 
                min(initial_spreaders, len(available_agents))
            )
            for agent in selected:
                agent.state = AgentState.SPREADER
                agent.step_became_spreader = 0
                agent.times_heard = 1
        
        # Record initial state
        self._record_metrics()
    
    def count_by_state(self) -> Dict[AgentState, int]:
        counts = {state: 0 for state in AgentState}
        for agent in self.agents:
            counts[agent.state] += 1
        return counts
    
    def _record_metrics(self) -> None:
        counts = self.count_by_state()
        total_heard = sum(1 for a in self.agents if a.times_heard > 0)
        
        metrics = SimulationMetrics(
            step=self.current_step,
            n_ignorant=counts[AgentState.IGNORANT],
            n_spreader=counts[AgentState.SPREADER],
            n_stifler=counts[AgentState.STIFLER],
            total_heard=total_heard
        )
        self.history.append(metrics)
    
    def step(self) -> None:
        self.current_step += 1
        self.agents.shuffle_do("step")
        self._record_metrics()
        
        # Stop if no more active spreaders
        if self.count_by_state()[AgentState.SPREADER] == 0:
            self.running = False
    
    def run(self, max_steps: int = 5000) -> int:
        while self.running and self.current_step < max_steps:
            self.step()
        return self.current_step
    
    def get_final_reach(self) -> int:
        counts = self.count_by_state()
        return counts[AgentState.SPREADER] + counts[AgentState.STIFLER]
    
    def get_final_reach_fraction(self) -> float:
        return self.get_final_reach() / (self.width * self.height)
    
    def get_grid_state_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.width, self.height), dtype=int)
        for agent, (x, y) in self.grid.coord_iter():
            matrix[x, y] = agent.state
        return matrix
    
    def get_times_heard_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.width, self.height), dtype=int)
        for agent, (x, y) in self.grid.coord_iter():
            matrix[x, y] = agent.times_heard
        return matrix


# --- SIMULATION RUNNER FUNCTIONS (for parallel execution) ---

def run_single_simulation(
    transmission_chance: float,
    stifle_chance: float,
    width: int = 40,
    height: int = 40,
    initial_spreaders: int = 5,
    max_steps: int = 5000,
    seed: Optional[int] = None
) -> Tuple[int, int, float]:

    model = RumorModel(
        width=width,
        height=height,
        transmission_chance=transmission_chance,
        stifle_chance=stifle_chance,
        initial_spreaders=initial_spreaders,
        seed=seed
    )
    steps = model.run(max_steps=max_steps)
    
    reach = model.get_final_reach()
    reach_fraction = model.get_final_reach_fraction()
    
    return reach, steps, reach_fraction


def run_simulation_with_history(
    transmission_chance: float,
    stifle_chance: float,
    width: int = 40,
    height: int = 40,
    initial_spreaders: int = 5,
    max_steps: int = 5000,
    seed: Optional[int] = None
) -> List[SimulationMetrics]:
    model = RumorModel(
        width=width,
        height=height,
        transmission_chance=transmission_chance,
        stifle_chance=stifle_chance,
        initial_spreaders=initial_spreaders,
        seed=seed
    )
    model.run(max_steps=max_steps)
    return model.history


if __name__ == "__main__":
    # Quick test
    print("Testing RumorModel...")
    model = RumorModel(
        width=20, 
        height=20, 
        transmission_chance=100, 
        stifle_chance=50,
        seed=42
    )
    steps = model.run()
    print(f"Completed in {steps} steps")
    print(f"Final reach: {model.get_final_reach()} / 400 ({model.get_final_reach_fraction():.1%})")
    print(f"History length: {len(model.history)} records")
