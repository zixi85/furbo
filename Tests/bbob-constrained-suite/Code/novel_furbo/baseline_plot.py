import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_problem(problem_dir, out_dir=None, show=False):
    # load final npy
    y_mono = os.path.join(problem_dir, "01_Y_mono.npy")
    y_best = os.path.join(problem_dir, "02_Y_best.npy")
    c_best = os.path.join(problem_dir, "02_C_best.npy")

    if not (os.path.exists(y_mono) and os.path.exists(y_best) and os.path.exists(c_best)):
        print("Missing processed npy files in", problem_dir)
        return

    y_mono = np.load(y_mono)
    y_best = np.load(y_best)
    c_best = np.load(c_best)

    if out_dir is None:
        out_dir = problem_dir
    os.makedirs(out_dir, exist_ok=True)

    # best feasible: plot per-repetition curves (faint), mean curve with ±1 std shading
    fig, ax = plt.subplots()

    # y_best shape expected: (n_reps, n_evals)
    if y_best.ndim == 1:
        y_best = y_best.reshape(1, -1)

    n_reps, n_evals = y_best.shape
    x = np.arange(1, n_evals + 1)

    # Plot each repetition as faint grey lines
    for i in range(n_reps):
        ax.plot(x, y_best[i], color='gray', alpha=0.25, linewidth=1)

    # Mean and std across repetitions
    mean_curve = np.mean(y_best, axis=0)
    std_curve = np.std(y_best, axis=0)

    ax.plot(x, mean_curve, color='C0', linewidth=2.5, label='Mean best feasible')
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color='C0', alpha=0.2, label='±1 std')

    ax.set_title(os.path.basename(problem_dir))
    ax.set_xlabel(f'Evaluation (T={n_evals})')
    ax.set_ylabel('Best feasible objective')
    ax.legend()

    fig.savefig(os.path.join(out_dir, "best_feasible.png"), bbox_inches='tight')
    plt.close(fig)

    print("Saved aggregated plot for", problem_dir)
