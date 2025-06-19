import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import tempfile
import os
from numba import jit

# --- Константы и цвета ---
NEIGHBOURHOOD = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE = 0, 1, 2
colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

@jit(nopython=True)
def iterate_optimized(X_curr, p_growth, f_lightning, ny_dim, nx_dim):
    X_next = np.zeros((ny_dim, nx_dim), dtype=X_curr.dtype)
    X_values = np.zeros(3, dtype=np.int64)  # [EMPTY, TREE, FIRE]
    grown = 0  # Количество выросших деревьев за шаг
    lightning_fires = 0  # Количество деревьев, загоревшихся от молнии за шаг
    for r in range(1, ny_dim - 1):
        for c in range(1, nx_dim - 1):
            current_state = X_curr[r, c]
            X_values[current_state] += 1
            if current_state == EMPTY:
                if np.random.random() <= p_growth:
                    X_next[r, c] = TREE
                    grown += 1
            elif current_state == TREE:
                X_next[r, c] = TREE
                caught_fire_from_neighbor = False
                for dr, dc in NEIGHBOURHOOD:
                    if X_curr[r + dr, c + dc] == FIRE:
                        X_next[r, c] = FIRE
                        caught_fire_from_neighbor = True
                        break
                if not caught_fire_from_neighbor:
                    if np.random.random() <= f_lightning:
                        X_next[r, c] = FIRE
                        lightning_fires += 1
    return X_next, X_values, grown, lightning_fires

def create_forest_animation(seed=0, frames=100, size=200, p=0.02, f=0.0001, forest_fraction=0.2, burning_fraction=0.01, fps=20, skip_video=False):
    np.random.seed(seed)
    ny, nx = size, size
    X = np.zeros((ny, nx), dtype=np.int8)
    interior_mask = np.random.random(size=(ny-2, nx-2)) < forest_fraction
    X[1:ny-1, 1:nx-1] = np.where(interior_mask, TREE, EMPTY).astype(np.int8)
    fire_mask = (np.random.random(size=(ny-2, nx-2)) < burning_fraction) & (X[1:ny-1, 1:nx-1] == TREE)
    X[1:ny-1, 1:nx-1][fire_mask] = FIRE

    stats = np.zeros((frames, 3), dtype=np.int64)
    s_arr = np.zeros(frames, dtype=np.float32)

    if skip_video:
        # Просто считаем статистику без рендера видео
        for i in range(frames):
            X, X_values, grown, lightning_fires = iterate_optimized(X, p, f, ny, nx)
            stats[i, :] = X_values
            s_arr[i] = grown / lightning_fires if lightning_fires > 0 else np.nan
        return None, stats, s_arr

    # --- видео и анимация ---
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
        X, X_values, grown, lightning_fires = iterate_optimized(X, p, f, ny, nx)
        stats[i, :] = X_values
        s_arr[i] = grown / lightning_fires if lightning_fires > 0 else np.nan
        im.set_data(X)
        return [im]

    writer = FFMpegWriter(fps=fps, bitrate=int(size*size*fps*0.08))
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani = FuncAnimation(fig, animate, frames=frames, interval=1000/fps, blit=True)
    ani.save(tmp_file.name, writer=writer)
    plt.close(fig)
    return tmp_file.name, stats, s_arr

def gradio_grid(seed, n_sim, size, frames, p, f, forest_fraction, burning_fraction, skip_video, p_step, n_f_steps, delta_f, only_final_plot):
    video_items = []
    s_points = []
    ratio_points = []
    rho_points = []
    ratio_fp_points = []
    min_ratio_fp = None
    min_rho = None

    for sim_idx in range(n_sim):
        cell_seed = seed + sim_idx
        p_current = p + p_step * sim_idx
        for f_idx in range(n_f_steps):
            f_current = f + delta_f * f_idx
            video_path, stats, s_arr = create_forest_animation(
                seed=cell_seed, frames=frames, size=size, p=p_current, f=f_current,
                forest_fraction=forest_fraction, burning_fraction=burning_fraction, skip_video=skip_video or only_final_plot
            )
            mean_s = np.nanmean(s_arr)
            # Сохраняем точки для итогового графика s vs (f/p)^-1
            if p_current > 0 and f_current > 0:
                ratio_inv = p_current / f_current
                ratio_points.append(ratio_inv)
                s_points.append(mean_s)
                # Для второго графика: средняя плотность деревьев (без рамки)
                # stats[:, 1] - количество деревьев на каждом шаге
                # (size-2)*(size-2) - число клеток без рамки
                rho_t = np.nanmean(stats[:, 1] / ((size-2)*(size-2)))
                ratio_fp = f_current / p_current
                rho_points.append(rho_t)
                ratio_fp_points.append(ratio_fp)
                # Запоминаем минимальное (f/p) и соответствующее rho_t
                if min_ratio_fp is None or ratio_fp < min_ratio_fp:
                    min_ratio_fp = ratio_fp
                    min_rho = rho_t
            # Если выбран только итоговый график, пропускаем остальные
            if only_final_plot:
                continue
            label = f"Seed: {cell_seed}, p={p_current:.4f}, f={f_current:.5f}"
            plt.figure(figsize=(5,3))
            x = np.arange(frames)
            plt.plot(x, stats[:, 0], color='gray', label='EMPTY')
            plt.plot(x, stats[:, 1], color='green', label='TREE')
            plt.plot(x, stats[:, 2], color='red', label='FIRE')
            plt.plot(x, s_arr, color='blue', label='s (grown/lightning)')
            mean_empty = stats[:, 0].mean()
            mean_tree = stats[:, 1].mean()
            mean_fire = stats[:, 2].mean()
            plt.axhline(mean_empty, color='gray', linestyle='--', alpha=0.7, label=f'EMPTY mean: {mean_empty:.0f}')
            plt.axhline(mean_tree, color='green', linestyle='--', alpha=0.7, label=f'TREE mean: {mean_tree:.0f}')
            plt.axhline(mean_fire, color='red', linestyle='--', alpha=0.7, label=f'FIRE mean: {mean_fire:.0f}')
            plt.axhline(mean_s, color='blue', linestyle='--', alpha=0.7, label=f's mean: {mean_s:.2f}')
            plt.xlabel('Frame')
            plt.ylabel('Cell count / s')
            # Форматируем f и p без незначащих нулей после запятой
            def trim_float(val):
                return ('{0:.10g}'.format(val)).rstrip('.').rstrip('0') if '.' in '{0:.10g}'.format(val) else str(val)
            f_str = trim_float(f_current)
            p_str = trim_float(p_current)
            if p_current > 0 and f_current > 0:
                ratio_str = f"({f_str}/{p_str})^{{-1}} = {ratio_inv:.2f}"
            else:
                ratio_str = "undefined"
            plt.title(
                f'Forest size: {size} | Frames: {frames} | Seed: {cell_seed}, p={p_str}, f={f_str}\n'
                f'$ (f/p)^{{-1}} = {ratio_str} $',
                fontsize=10
            )
            plt.legend(loc='upper right', framealpha=0.5)
            plt.tight_layout()
            img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(img_tmp.name)
            plt.close()
            if not skip_video and video_path is not None:
                video_items.append((video_path, label))
            video_items.append((img_tmp.name, f"Cells vs Time: {label}"))
    # Итоговый график s vs (f/p)^-1
    if len(ratio_points) > 0 and len(s_points) > 0:
        plt.figure(figsize=(5,3), dpi=300)  # <-- dpi=300
        plt.scatter(ratio_points, s_points, color='blue')
        plt.xlabel('$(f/p)^{-1}$')
        plt.ylabel('s (mean grown/lightning)')
        plt.title('s vs $(f/p)^{-1}$ for all simulations')
        plt.tight_layout()
        img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(img_tmp.name)
        plt.close()
        video_items.append((img_tmp.name, "s vs (f/p)^-1"))
    # Итоговый график rho_t vs (f/p)
    if len(ratio_fp_points) > 0 and len(rho_points) > 0:
        plt.figure(figsize=(5,3), dpi=300)  # <-- dpi=300
        plt.scatter(ratio_fp_points, rho_points, color='green')
        plt.xlabel('$(f/p)$')
        plt.ylabel(r'$\rho_t$ (mean tree density)')
        if min_ratio_fp is not None and min_rho is not None:
            plt.title(r'$\rho_t$ vs $(f/p)$ for all simulations' + f"\n" + r'$\rho_t^c$ = ' + f"{min_rho:.4f} at min $(f/p)$ = {min_ratio_fp:.4g}")
        else:
            plt.title(r'$\rho_t$ vs $(f/p)$ for all simulations')
        plt.tight_layout()
        img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(img_tmp.name)
        plt.close()
        video_items.append((img_tmp.name, r"$\rho_t$ vs (f/p)"))
    return video_items

gr.Interface(
    fn=gradio_grid,
    inputs=[
        gr.Number(label="Random Seed", value=0),
        gr.Slider(1, 100, value=1, step=1, label="Number of simulations (SL)"),
        gr.Slider(50, 2000, value=500, step=10, label="Forest Size (NxN)"),
        gr.Slider(10, 20000, value=1000, step=1, label="Number of Frames"),
        gr.Slider(0.001, 0.1, value=0.02, step=0.001, label="Tree Growth Probability (p)"),
        gr.Slider(0.0, 0.01, value=0.001, step=0.0001, label="Lightning Probability (f)"),
        gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="Initial Forest Fraction"),
        gr.Slider(0.0, 0.5, value=0, step=0.01, label="Initial Burning Fraction"),
        gr.Checkbox(label="Only plot graphs (no video)", value=True),
        gr.Number(label="Tree Growth Probability Step (Δp)", value=-0.00199, precision=6),
        gr.Number(label="N_Δf (Lightning Probability Steps)", value=1, precision=0),
        gr.Number(label="Δf (Lightning Probability Step)", value=-0.0000999, precision=6),
        gr.Checkbox(label="Show only final s vs (f/p)^-1 plot", value=False),  # Новый чекбокс
    ],
    outputs=gr.Gallery(label="Forest Fire Animation Grid (MP4 + TREE Graph)", columns=2, height="auto"),
    title="Forest Fire Cellular Automaton Animation Grid",
    description="Simulates SL independent forest fire cellular automata. For each simulation, shows MP4 animation and TREE count graph. Optionally, only the final s vs (f/p)^-1 plot can be shown."
).launch()
