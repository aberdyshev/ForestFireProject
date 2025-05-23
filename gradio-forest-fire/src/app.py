import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import tempfile
from numba import jit

NEIGHBOURHOOD = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE = 0, 1, 2
colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

@jit(nopython=True)
def iterate_optimized(X_curr, p_growth, f_lightning, ny_dim, nx_dim):
    X_next = np.zeros((ny_dim, nx_dim), dtype=X_curr.dtype)
    for r in range(1, ny_dim - 1):
        for c in range(1, nx_dim - 1):
            current_state = X_curr[r, c]
            if current_state == EMPTY:
                if np.random.random() <= p_growth:
                    X_next[r, c] = TREE
            elif current_state == TREE:
                X_next[r, c] = TREE
                caught_fire_from_neighbor = False
                for dr, dc in NEIGHBOURHOOD:
                    is_diagonal = (abs(dr) == abs(dc))
                    can_spread_from_this_neighbor = True
                    if is_diagonal and np.random.random() < 0.573:
                        can_spread_from_this_neighbor = False
                    if can_spread_from_this_neighbor:
                        if X_curr[r + dr, c + dc] == FIRE:
                            X_next[r, c] = FIRE
                            caught_fire_from_neighbor = True
                            break
                if not caught_fire_from_neighbor:
                    if np.random.random() <= f_lightning:
                        X_next[r, c] = FIRE
    return X_next

def create_forest_animation(seed=0, frames=100, size=200, p=0.02, f=0.0001, forest_fraction=0.2, fps=20):
    np.random.seed(seed)
    ny, nx = size, size
    X = np.zeros((ny, nx), dtype=np.int8)
    interior_mask = np.random.random(size=(ny-2, nx-2)) < forest_fraction
    X[1:ny-1, 1:nx-1] = np.where(interior_mask, TREE, EMPTY).astype(np.int8)

    dpi = 100
    figsize = (size / dpi, size / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    im = ax.imshow(X, cmap=cmap, norm=norm, interpolation='nearest')

    def animate(i):
        nonlocal X
        X = iterate_optimized(X, p, f, ny, nx)
        im.set_data(X)
        return [im]

    writer = FFMpegWriter(fps=fps, bitrate=int(size*size*fps*0.08))
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani = FuncAnimation(fig, animate, frames=frames, interval=1000/fps, blit=True)
    ani.save(tmp_file.name, writer=writer)
    plt.close(fig)
    return tmp_file.name

def gradio_grid(seed, grid_size, size, frames, p, f, forest_fraction):
    video_items = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_seed = seed + i * grid_size + j
            video_path = create_forest_animation(
                seed=cell_seed, frames=frames, size=size, p=p, f=f, forest_fraction=forest_fraction
            )
            label = f"Seed: {cell_seed} ({i},{j})"
            video_items.append((video_path, label))
    return video_items

gr.Interface(
    fn=gradio_grid,
    inputs=[
        gr.Number(label="Random Seed", value=0),
        gr.Slider(1, 10, value=1, step=1, label="Grid Size GR (GRxGR)"),
        gr.Slider(50, 1500, value=100, step=10, label="Forest Size (NxN)"),
        gr.Slider(10, 2000, value=50, step=1, label="Number of Frames"),
        gr.Slider(0.001, 0.1, value=0.02, step=0.001, label="Tree Growth Probability (p)"),
        gr.Slider(0.0, 0.01, value=0.0001, step=0.0001, label="Lightning Probability (f)"),
        gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="Initial Forest Fraction"),
    ],
    outputs=gr.Gallery(label="Forest Fire Animation Grid (MP4)", columns=5, height="auto"),
    title="Forest Fire Cellular Automaton Animation Grid",
    description="Simulates a GRxGR grid of independent forest fire cellular automata. Download the MP4 animations."
).launch()