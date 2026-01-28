import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from pathlib import Path

from rumor_model import RumorModel, AgentState

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_spreading_animation(
    stifle_chance: float = 50.0,
    transmission_chance: float = 100.0,
    grid_size: int = 40,
    seed: int = 42,
    max_frames: int = 200,
    interval: int = 100,
    save_path: str = None
) -> None:
    model = RumorModel(
        width=grid_size,
        height=grid_size,
        transmission_chance=transmission_chance,
        stifle_chance=stifle_chance,
        seed=seed
    )
    
    # colormap: White (Ignorant), Red (Spreader), Green (Stifler)
    cmap = ListedColormap(['white', 'red', 'green'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initial grid
    grid_data = model.get_grid_state_matrix()
    im = ax1.imshow(grid_data.T, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax1.set_title('Rumor Spreading Simulation', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Ignorant'),
        Patch(facecolor='red', label='Spreader'),
        Patch(facecolor='green', label='Stifler')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    times = [0]
    ignorants = [model.count_by_state()[AgentState.IGNORANT]]
    spreaders = [model.count_by_state()[AgentState.SPREADER]]
    stiflers = [model.count_by_state()[AgentState.STIFLER]]
    
    line_ig, = ax2.plot(times, ignorants, 'gray', linewidth=2, label='Ignorant')
    line_sp, = ax2.plot(times, spreaders, 'red', linewidth=2, label='Spreader')
    line_st, = ax2.plot(times, stiflers, 'green', linewidth=2, label='Stifler')
    
    ax2.set_xlim(0, max_frames)
    ax2.set_ylim(0, grid_size ** 2)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Number of Agents', fontsize=12)
    ax2.set_title('Population Dynamics', fontsize=12)
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'DK Rumor Model | Transmission: {transmission_chance}%, Stifle: {stifle_chance}%',
                fontsize=14, fontweight='bold')
    
    def animate(frame):
        if model.running:
            model.step()
            
            grid_data = model.get_grid_state_matrix()
            im.set_array(grid_data.T)
            
            counts = model.count_by_state()
            times.append(model.current_step)
            ignorants.append(counts[AgentState.IGNORANT])
            spreaders.append(counts[AgentState.SPREADER])
            stiflers.append(counts[AgentState.STIFLER])
            
            line_ig.set_data(times, ignorants)
            line_sp.set_data(times, spreaders)
            line_st.set_data(times, stiflers)
            
            ax2.set_xlim(0, max(times) + 10)
            
            ax1.set_title(f'Step: {model.current_step} | Reach: {counts[AgentState.SPREADER] + counts[AgentState.STIFLER]}')
        
        return [im, line_ig, line_sp, line_st]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=interval, blit=False
    )
    
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == '.gif':
            anim.save(save_path, writer='pillow', fps=10)
        else:
            anim.save(save_path, writer='ffmpeg', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_animation(
    stifle_values: list = [25, 75],
    transmission_chance: float = 100.0,
    grid_size: int = 30,
    seed: int = 42,
    max_frames: int = 150,
    save_path: str = None
) -> None:
    n_plots = len(stifle_values)
    
    models = []
    for stifle in stifle_values:
        model = RumorModel(
            width=grid_size,
            height=grid_size,
            transmission_chance=transmission_chance,
            stifle_chance=stifle,
            seed=seed
        )
        models.append(model)
    
    cmap = ListedColormap(['white', 'red', 'green'])
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    images = []
    for ax, model, stifle in zip(axes, models, stifle_values):
        grid_data = model.get_grid_state_matrix()
        im = ax.imshow(grid_data.T, cmap=cmap, vmin=0, vmax=2, origin='lower')
        ax.set_title(f'Stifle: {stifle}%', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(im)
    
    fig.suptitle('Comparison of Stifling Effects', fontsize=14, fontweight='bold')
    
    def animate(frame):
        for im, model in zip(images, models):
            if model.running:
                model.step()
            grid_data = model.get_grid_state_matrix()
            im.set_array(grid_data.T)
        
        for ax, model, stifle in zip(axes, models, stifle_values):
            counts = model.count_by_state()
            reach = counts[AgentState.SPREADER] + counts[AgentState.STIFLER]
            ax.set_title(f'Stifle: {stifle}% | Step: {model.current_step} | Reach: {reach}')
        
        return images
    
    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=100, blit=False
    )
    
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == '.gif':
            anim.save(save_path, writer='pillow', fps=10)
        else:
            anim.save(save_path, writer='ffmpeg', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Generating animations...")
    
    create_spreading_animation(
        stifle_chance=50,
        transmission_chance=100,
        grid_size=40,
        save_path=OUTPUT_DIR / "rumor_spreading.gif"
    )
    
    create_comparison_animation(
        stifle_values=[25, 75],
        grid_size=30,
        save_path=OUTPUT_DIR / "stifle_comparison.gif"
    )
    
    print("Done!")
