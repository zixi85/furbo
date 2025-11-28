"""Plot baseline qLogEI results saved by `baseline_experiment.py` or `baseline_runner.py`.

This script expects directories like `results/baseline_qlogei/f2_i0_d2/` containing
files `obj_rep{r}.npy`, `cons_rep{r}.npy`, `X_rep{r}.npy` (or the single-run names `obj.npy`, `cons.npy`, `X.npy`).

Outputs PNG plots in each problem directory: `obj_trace_rep{r}.png` and `best_feasible.png` (aggregated).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_runs(problem_dir):
    runs = []
    # support both obj_rep*.npy and obj.npy naming
    obj_files = sorted([f for f in os.listdir(problem_dir) if f.startswith('obj') and f.endswith('.npy')])
    cons_files = sorted([f for f in os.listdir(problem_dir) if f.startswith('cons') and f.endswith('.npy')])

    if not obj_files:
        return runs

    # Match files by rep index if possible
    for i, objf in enumerate(obj_files):
        obj = np.load(os.path.join(problem_dir, objf), allow_pickle=True)
        # try to find corresponding cons file
        cons = None
        if i < len(cons_files):
            cons = np.load(os.path.join(problem_dir, cons_files[i]), allow_pickle=True)
        else:
            # try same base name
            base = objf.replace('obj', 'cons')
            if os.path.exists(os.path.join(problem_dir, base)):
                cons = np.load(os.path.join(problem_dir, base), allow_pickle=True)

        # normalize saved None -> actual None (np.save(None) becomes array(None, dtype=object))
        if isinstance(cons, np.ndarray):
            try:
                if cons.shape == () and cons.item() is None:
                    cons = None
            except Exception:
                pass

        runs.append((obj, cons, objf))

    return runs


def compute_best_feasible(y, cons):
    """Compute cumulative best feasible curve for minimization.

    y: 1D array of objective values
    cons: array shape (n_samples,) or (n_samples, n_constraints)
    Returns array best_feasible of same length with np.nan before first feasible.
    """
    y = np.asarray(y).reshape(-1)
    if cons is None:
        feasible = np.ones_like(y, dtype=bool)
    else:
        c = np.asarray(cons)
        if c.ndim == 1:
            feasible = c <= 0
        else:
            feasible = np.all(c <= 0, axis=1)

    best = np.full_like(y, np.nan, dtype=float)
    current_best = np.nan
    for i in range(len(y)):
        if feasible[i]:
            if np.isnan(current_best) or y[i] < current_best:
                current_best = float(y[i])
        best[i] = current_best

    return best, feasible


def plot_problem(problem_dir, out_dir=None, show=False):
    runs = load_runs(problem_dir)
    if not runs:
        print('No runs found in', problem_dir)
        return

    if out_dir is None:
        out_dir = problem_dir
    os.makedirs(out_dir, exist_ok=True)

    best_curves = []

    for idx, (obj, cons, name) in enumerate(runs):
        y = np.asarray(obj).squeeze()
        # if stored as column vectors
        if y.ndim > 1:
            y = y.reshape(-1)
        best, feasible = compute_best_feasible(y, cons)
        best_curves.append(best)

        # Plot objective trace with feasible markers
        fig, ax = plt.subplots()
        ax.plot(y, label='objective')
        ax.scatter(np.where(feasible)[0], y[feasible], color='green', label='feasible')
        ax.scatter(np.where(~feasible)[0], y[~feasible], color='red', label='infeasible', s=10)
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Objective')
        ax.set_title(os.path.basename(problem_dir) + ' - ' + name)
        ax.legend()
        fig.savefig(os.path.join(out_dir, f'obj_trace_{idx}.png'), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

        # Also save best curve per rep
        fig, ax = plt.subplots()
        ax.plot(best, label='best feasible (per-rep)')
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Best feasible objective')
        ax.set_title(os.path.basename(problem_dir) + ' - best feasible - ' + name)
        fig.savefig(os.path.join(out_dir, f'best_feasible_rep{idx}.png'), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # Aggregate across runs: compute mean/std of best curves (align lengths)
    max_len = max(len(b) for b in best_curves)
    arr = np.full((len(best_curves), max_len), np.nan)
    for i, b in enumerate(best_curves):
        arr[i, :len(b)] = b

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    fig, ax = plt.subplots()
    x = np.arange(len(mean))
    ax.plot(x, mean, label='mean best feasible')
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, label='std')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Best feasible objective')
    ax.set_title(os.path.basename(problem_dir) + ' - aggregated best feasible')
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'best_feasible.png'), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print('Saved plots for', problem_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/baseline_qlogei', help='Base results directory')
    parser.add_argument('--problem', type=str, default=None, help='Specific problem folder name (e.g., f2_i0_d2)')
    args = parser.parse_args()

    base = args.results
    # Validate base directory
    if not os.path.exists(base):
        print(f"Results base directory does not exist: {base}")
        print("Run the baseline experiment first (see README) or pass a different --results path.")
        print("Available directories in current working directory:")
        for name in sorted(os.listdir('.'))[:50]:
            print('  ', name)
        raise SystemExit(1)

    if args.problem:
        problem_dir = os.path.join(base, args.problem)
        if not os.path.isdir(problem_dir):
            print(f"Requested problem folder not found: {problem_dir}")
            print("Available problem folders under", base)
            for name in sorted(os.listdir(base)):
                print('  ', name)
            raise SystemExit(1)
        plot_problem(problem_dir)
    else:
        found = False
        for name in sorted(os.listdir(base)):
            problem_dir = os.path.join(base, name)
            if os.path.isdir(problem_dir):
                found = True
                plot_problem(problem_dir)
        if not found:
            print(f"No problem folders found under {base}. Did the baseline run complete and save results there?")
            raise SystemExit(1)


if __name__ == '__main__':
    main()
