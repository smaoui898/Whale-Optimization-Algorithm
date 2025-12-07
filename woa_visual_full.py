"""
woa_visual_full.py
WOA 2D visualization: heatmap, 3D surface, and HD MP4 animation.

Save as woa_visual_full.py and run with: python woa_visual_full.py
Requires: numpy, matplotlib, (ffmpeg for MP4)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# ---------------------------
# User parameters
# ---------------------------
benchmark = "sphere"   # options: "sphere", "rastrigin", "ackley", "rosenbrock"
N = 40                 # population size (number of whales)
max_iter = 150         # number of iterations / animation frames
dim = 2                # 2D for visualization
grid_res = 300         # resolution for contour/surface grid
out_dir = "."          # output directory (change if desired)

# ---------------------------
# Benchmark functions
# ---------------------------

def sphere_grid(X, Y):
    return X**2 + Y**2

def sphere_point(x):
    return np.sum(x**2)

def rastrigin_grid(X, Y):
    A = 10
    return A*2 + (X**2 - A*np.cos(2*np.pi*X)) + (Y**2 - A*np.cos(2*np.pi*Y))

def rastrigin_point(x):
    A = 10
    return A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))

def ackley_grid(X, Y):
    a = 20; b = 0.2; c = 2*np.pi
    Z = -a*np.exp(-b*np.sqrt((X**2+Y**2)/2)) - np.exp((np.cos(c*X)+np.cos(c*Y))/2) + a + np.e
    return Z

def ackley_point(x):
    a = 20; b = 0.2; c = 2*np.pi; d = len(x)
    return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
            - np.exp(np.sum(np.cos(c * x)) / d)
            + a + np.e)

def rosenbrock_grid(X, Y):
    return 100*(Y - X**2)**2 + (1 - X)**2

def rosenbrock_point(x):
    # for >2D, see example usage; here we implement 2D pair
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# select function handles and bounds
if benchmark == "sphere":
    grid_func = sphere_grid; point_func = sphere_point; lb, ub = -10, 10
elif benchmark == "rastrigin":
    grid_func = rastrigin_grid; point_func = rastrigin_point; lb, ub = -5.12, 5.12
elif benchmark == "ackley":
    grid_func = ackley_grid; point_func = ackley_point; lb, ub = -32, 32
elif benchmark == "rosenbrock":
    grid_func = rosenbrock_grid; point_func = rosenbrock_point; lb, ub = -2, 2
else:
    raise ValueError("Unknown benchmark: " + str(benchmark))

# ---------------------------
# WOA implementation (history)
# ---------------------------
def initialize_population(N, dim, lb, ub):
    return np.random.uniform(lb, ub, size=(N, dim))

def clip_positions(pop, lb, ub):
    return np.clip(pop, lb, ub)

def woa_run_history(point_func, N=30, dim=2, lb=-10, ub=10, max_iter=100, seed=0):
    np.random.seed(seed)
    pop = initialize_population(N, dim, lb, ub)
    fitness = np.array([point_func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    hist_pos = [pop.copy()]
    hist_best = [best_fit]
    for t in range(1, max_iter+1):
        a = 2 - 2 * (t / max_iter)  # linear decrease 2->0
        for i in range(N):
            r = np.random.rand(dim)
            A = 2 * a * r - a
            C = 2 * r
            p = np.random.rand()
            if p < 0.5:
                if np.all(np.abs(A) < 1):
                    # exploitation (encircling)
                    D = np.abs(C * best_pos - pop[i])
                    pop[i] = best_pos - A * D
                else:
                    # exploration
                    rand_idx = np.random.randint(0, N)
                    X_rand = pop[rand_idx]
                    D_rand = np.abs(C * X_rand - pop[i])
                    pop[i] = X_rand - A * D_rand
            else:
                # spiral updating
                D_prime = np.abs(best_pos - pop[i])
                b = 1.0
                l = np.random.uniform(-1, 1, size=dim)
                pop[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
        pop = clip_positions(pop, lb, ub)
        fitness = np.array([point_func(ind) for ind in pop])
        cur_best_idx = np.argmin(fitness)
        cur_best_fit = fitness[cur_best_idx]
        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_pos = pop[cur_best_idx].copy()
        hist_pos.append(pop.copy())
        hist_best.append(best_fit)
    return np.array(hist_pos), np.array(hist_best), best_pos, best_fit

# run WOA and collect history
history_positions, history_best, best_pos, best_fit = woa_run_history(point_func, N=N, dim=dim, lb=lb, ub=ub, max_iter=max_iter, seed=1)

print("WOA finished. Benchmark:", benchmark)
print("Best position:", best_pos)
print("Best fitness:", best_fit)

# ---------------------------
# Create search-space heatmap and 3D surface images
# ---------------------------
xs = np.linspace(lb, ub, grid_res)
ys = np.linspace(lb, ub, grid_res)
Xg, Yg = np.meshgrid(xs, ys)
Zg = grid_func(Xg, Yg)

heatmap_path = os.path.join(out_dir, "woa_heatmap.png")
plt.figure(figsize=(8,6))
plt.contourf(Xg, Yg, Zg, levels=60, cmap='viridis')
plt.colorbar(label='Fitness')
plt.scatter(history_positions[0][:,0], history_positions[0][:,1], c='white', s=18, label='initial whales')
plt.scatter(best_pos[0], best_pos[1], marker='*', c='red', s=130, label='best final')
plt.title(f"Search Space Heatmap ({benchmark})")
plt.xlabel('x1'); plt.ylabel('x2')
plt.legend()
plt.tight_layout()
plt.savefig(heatmap_path, dpi=150)
plt.close()

surface_path = os.path.join(out_dir, "woa_surface.png")
fig = plt.figure(figsize=(8,6))
ax3 = fig.add_subplot(111, projection='3d')
# for memory safety, plot a coarser surface if grid_res is large
coarse = max(40, int(grid_res/6))
xs2 = np.linspace(lb, ub, coarse)
ys2 = np.linspace(lb, ub, coarse)
X2, Y2 = np.meshgrid(xs2, ys2)
Z2 = grid_func(X2, Y2)
ax3.plot_surface(X2, Y2, Z2, cmap='viridis', linewidth=0, antialiased=False, alpha=0.9)
ax3.set_xlabel('x1'); ax3.set_ylabel('x2'); ax3.set_zlabel('Fitness')
ax3.set_title(f"3D Surface ({benchmark})")
plt.tight_layout()
plt.savefig(surface_path, dpi=150)
plt.close()

print("Saved heatmap ->", heatmap_path)
print("Saved surface  ->", surface_path)

# ---------------------------
# Create full HD animation MP4 (left: contour + convergence inset; right: 3D surface)
# ---------------------------
anim_path = os.path.join(out_dir, "woa_full_HD_animation.mp4")
fig = plt.figure(figsize=(16,9), dpi=100)  # 1600x900 canvas
ax_left = fig.add_subplot(1,2,1)
ax_right = fig.add_subplot(1,2,2, projection='3d')

# Left: contour and scatter placeholders
ax_left.contourf(Xg, Yg, Zg, levels=60, cmap='viridis')
scat = ax_left.scatter([], [], c='white', s=40, edgecolor='k')
best_sc = ax_left.scatter([], [], marker='*', c='red', s=140, edgecolor='k')
ax_left.set_xlim(lb, ub); ax_left.set_ylim(lb, ub)
ax_left.set_title('Contour & Whale Positions')
ax_left.set_xlabel('x1'); ax_left.set_ylabel('x2')

# Right: 3D surface (plotted once)
# reuse the coarser surface to keep animation responsive
ax_right.plot_surface(X2, Y2, Z2, cmap='viridis', linewidth=0, antialiased=False, alpha=0.85)
traj_scatter = ax_right.scatter([], [], [], c='white', s=30, edgecolor='k')
best_sc_3d = ax_right.scatter([], [], [], marker='*', c='red', s=140, edgecolor='k')
ax_right.set_xlim(lb, ub); ax_right.set_ylim(lb, ub)
zmin = np.nanmin(Zg); zmax = np.nanpercentile(Zg, 99)
ax_right.set_zlim(zmin, zmax)
ax_right.set_title('3D Surface & Whale Positions')
ax_right.set_xlabel('x1'); ax_right.set_ylabel('x2'); ax_right.set_zlabel('Fitness')

# convergence axes inset at bottom-left
conv_ax = fig.add_axes([0.08, 0.06, 0.35, 0.18])
conv_ax.set_title('Convergence Curve')
conv_line, = conv_ax.plot([], [], lw=2, color='tab:orange')
conv_ax.set_xlim(0, max_iter)
# set y-limits safely (avoid identical min/max)
ymin = np.min(history_best) if np.min(history_best) < np.max(history_best) else np.min(history_best) - 1.0
ymax = np.max(history_best) + 1e-8
conv_ax.set_ylim(ymin*0.9, ymax*1.1)
conv_ax.set_xlabel('Iter'); conv_ax.set_ylabel('Best fitness')
conv_ax.grid(True)

iter_text = ax_left.text(0.02, 0.95, '', transform=ax_left.transAxes,
                         color='white', bbox=dict(facecolor='black', alpha=0.5))

def init_anim():
    scat.set_offsets(np.empty((0,2)))
    best_sc.set_offsets(np.empty((0,2)))
    traj_scatter._offsets3d = ([], [], [])
    best_sc_3d._offsets3d = ([], [], [])
    conv_line.set_data([], [])
    iter_text.set_text('')
    return scat, best_sc, traj_scatter, best_sc_3d, conv_line, iter_text

def update_anim(frame):
    pos = history_positions[frame]  # (N,2)
    scat.set_offsets(pos)
    # compute fitness at this frame for selection of local best
    fitness_now = np.array([point_func(ind) for ind in pos])
    idx = np.argmin(fitness_now)
    best = pos[idx]
    best_sc.set_offsets([best])
    # update 3d scatter
    xs = pos[:,0]; ys = pos[:,1]; zs = grid_func(xs, ys)
    traj_scatter._offsets3d = (xs, ys, zs)
    best_sc_3d._offsets3d = ([best[0]], [best[1]], [grid_func(best[0], best[1])])
    # update convergence line
    conv_line.set_data(np.arange(frame+1), history_best[:frame+1])
    iter_text.set_text(f"Iter: {frame}\nBest: {history_best[frame]:.3e}")
    return scat, best_sc, traj_scatter, best_sc_3d, conv_line, iter_text

anim = animation.FuncAnimation(fig, update_anim, frames=len(history_positions),
                               init_func=init_anim, interval=120, blit=False)

# Try to save as MP4 (ffmpeg). If it fails, fallback to GIF.
saved_ok = False
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='WOA'), bitrate=4000)
    anim.save(anim_path, writer=writer)
    saved_ok = True
except Exception as e:
    print("MP4 save failed (ffmpeg not available?). Error:", e)
    gif_path = os.path.join(out_dir, "woa_full_HD_animation.gif")
    try:
        anim.save(gif_path, writer='pillow', fps=10)
        anim_path = gif_path
        saved_ok = True
    except Exception as e2:
        print("GIF fallback failed:", e2)
        saved_ok = False
        anim_path = None

plt.close(fig)

if saved_ok:
    print("Animation saved to:", anim_path)
else:
    print("Animation could not be saved. You can still generate frames by running the script locally with ffmpeg installed.")

# ---------------------------
# Done
# ---------------------------
print("\nFiles created (if running locally):")
print(" - Heatmap :", heatmap_path)
print(" - Surface :", surface_path)
print(" - Animation :", anim_path if anim_path is not None else "not created")
