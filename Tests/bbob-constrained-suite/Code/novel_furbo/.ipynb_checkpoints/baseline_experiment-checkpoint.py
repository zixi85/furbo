"""Run BoTorch qLogEI baseline experiments with simple CLI.

Usage examples:
  # run the built-in full baseline set (may be long)
  python baseline_experiment.py --full

  # run a single short case (f2,i0,d2)
  python baseline_experiment.py --funcs 2 --insts 0 --dims 2 --reps 1

Results are saved under `results/baseline_qlogei` by default.
"""
import argparse
import os
import numpy as np
from baseline_botorch_qlogei import run_botorch_qlogei


def parse_list(s):
    # parse comma separated ints
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run the full baseline set from baseline_runner')
    parser.add_argument('--funcs', type=str, help='Comma-separated function ids, e.g. "2,4,6"')
    parser.add_argument('--insts', type=str, help='Comma-separated instance indices (0-based), e.g. "0,1"')
    parser.add_argument('--dims', type=str, help='Comma-separated dimensions, e.g. "2,10"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reps', type=int, default=1, help='Number of repetitions per (f,inst,dim)')
    parser.add_argument('--outdir', type=str, default='results/baseline_qlogei', help='Directory to save results')

    args = parser.parse_args()

    if args.full:
        # delegate to baseline_runner for the original full set
        from baseline_runner import run_baseline_all
        run_baseline_all()
        return

    if not (args.funcs and args.insts and args.dims):
        parser.error('Specify --funcs, --insts and --dims, or use --full')

    funcs = parse_list(args.funcs)
    insts = parse_list(args.insts)
    dims = parse_list(args.dims)

    os.makedirs(args.outdir, exist_ok=True)

    for f in funcs:
        for inst in insts:
            for d in dims:
                budget = 30 * d
                for rep in range(args.reps):
                    print(f"Running baseline qLogEI: f{f}, inst {inst}, dim {d}, rep {rep}")
                    y_obj, y_con, X = run_botorch_qlogei(
                        dim=d,
                        budget=budget,
                        coco_fun=f,
                        coco_instance=inst,
                        seed=args.seed + rep
                    )

                    save_dir = os.path.join(args.outdir, f"f{f}_i{inst}_d{d}")
                    os.makedirs(save_dir, exist_ok=True)

                    # y_obj and y_con may be torch tensors
                    try:
                        np.save(os.path.join(save_dir, f"obj_rep{rep}.npy"), y_obj.numpy())
                    except Exception:
                        np.save(os.path.join(save_dir, f"obj_rep{rep}.npy"), np.array(y_obj))

                    try:
                        np.save(os.path.join(save_dir, f"cons_rep{rep}.npy"), y_con.numpy())
                    except Exception:
                        np.save(os.path.join(save_dir, f"cons_rep{rep}.npy"), np.array(y_con))

                    np.save(os.path.join(save_dir, f"X_rep{rep}.npy"), X)

                    print("Saved:", save_dir)
                # After finishing all repetitions for this (f,inst,dim), plot and show results
                try:
                    from baseline_plot import plot_problem
                    # show=True will call plt.show() for each generated figure
                    plot_problem(save_dir, out_dir=save_dir, show=True)
                except Exception as e:
                    print("Plotting failed:", e)


if __name__ == '__main__':
    main()
