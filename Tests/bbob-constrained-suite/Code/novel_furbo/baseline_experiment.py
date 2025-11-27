"""
Run BoTorch qLogEI baseline experiments with simple CLI.

Usage examples:
  python baseline_experiment.py --full
  python baseline_experiment.py --funcs 2 --insts 0 --dims 2 --reps 3
"""

import argparse
import os
import numpy as np
from baseline_botorch_qlogei import run_botorch_qlogei
from baseline_plot import plot_problem
from baseline_postprocess import aggregate_problem


def parse_list(s):
    """Parse comma-separated integers."""
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run the full baseline set')
    parser.add_argument('--funcs', type=str, help='Comma-separated function ids, e.g. "2,4,6"')
    parser.add_argument('--insts', type=str, help='Comma-separated instance indices, e.g. "0,1,2"')
    parser.add_argument('--dims', type=str, help='Comma-separated dimensions, e.g. "2,10"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='results/baseline_qlogei')

    args = parser.parse_args()

    # run full mandatory baseline
    if args.full:
        from baseline_runner import run_baseline_all
        run_baseline_all()
        return

    # run specified subset
    if not (args.funcs and args.insts and args.dims):
        parser.error("Specify --funcs, --insts, --dims, or use --full")

    funcs = parse_list(args.funcs)
    insts = parse_list(args.insts)
    dims = parse_list(args.dims)

    os.makedirs(args.outdir, exist_ok=True)

    for f in funcs:
        for inst in insts:
            for d in dims:
                budget = 10 * d
                problem_dir = os.path.join(args.outdir, f"f{f}_i{inst}_d{d}")
                os.makedirs(problem_dir, exist_ok=True)

                for rep in range(args.reps):
                    print(f"Running qLogEI baseline: f{f}, inst {inst}, dim {d}, rep {rep}, budget {budget}")

                    y_obj, y_con, X = run_botorch_qlogei(
                        dim=d,
                        budget=budget,
                        coco_fun=f,
                        coco_instance=inst,
                        seed=args.seed + rep,
                    )

                    # Helper to convert torch tensors or lists to numpy arrays
                    def to_numpy(arr):
                        try:
                            # torch Tensor
                            import torch
                            if isinstance(arr, torch.Tensor):
                                return arr.detach().cpu().numpy()
                        except Exception:
                            # torch not installed or other issue -> fallthrough
                            pass

                        # numpy array already
                        try:
                            import numpy as _np
                            if isinstance(arr, _np.ndarray):
                                return arr
                        except Exception:
                            pass

                        # list-like fallback
                        try:
                            return np.asarray(arr)
                        except Exception:
                            return np.array([arr])

                    y_obj_np = to_numpy(y_obj)
                    y_con_np = None if y_con is None else to_numpy(y_con)
                    X_np = to_numpy(X)

                    np.save(os.path.join(problem_dir, f"obj_rep{rep}.npy"), y_obj_np)
                    np.save(os.path.join(problem_dir, f"cons_rep{rep}.npy"), y_con_np)
                    np.save(os.path.join(problem_dir, f"X_rep{rep}.npy"), X_np)

                    # Also save a torch-style history for downstream postprocessing
                    try:
                        import torch
                        c_for_torch = y_con_np if y_con_np is not None else np.zeros_like(y_obj_np)
                        event = {'batch': {'Y': torch.as_tensor(y_obj_np), 'C': torch.as_tensor(c_for_torch)}}
                        torch_fname = f"baseline_f{f}_i{inst}_d{d}_it_{rep}.torch"
                        torch.save([event], os.path.join(problem_dir, torch_fname))
                    except Exception:
                        pass

                # plot after all reps
                try:
                    # ensure aggregated files exist
                    try:
                        aggregate_problem(problem_dir, n_iteration=budget)
                    except Exception:
                        pass
                    plot_problem(problem_dir, out_dir=problem_dir, show=True)
                except Exception as e:
                    print("Plotting failed:", e)


if __name__ == '__main__':
    main()
