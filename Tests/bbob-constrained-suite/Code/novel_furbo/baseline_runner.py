from baseline_botorch_qlogei import run_botorch_qlogei
import numpy as np
import os
from baseline_plot import plot_problem


def run_baseline_all(
    funs=None,
    instances=None,
    dims=None,
    reps=5,
    base_outdir="results/baseline_qlogei",
    seed=0,
):
    """
    Run the full mandatory baseline set:

    - Functions: F2, F4, F6, F50, F52, F54
    - Instances: 0, 1, 2
    - Dimensions: 2, 10
    - Repetitions: 5 per (f, inst, dim)
    - Budget: 10 * D (total evaluations)
    """
    if funs is None:
        funs = [2, 4, 6, 50, 52, 54]
    if instances is None:
        instances = [0, 1, 2]
    if dims is None:
        dims = [2, 10]

    os.makedirs(base_outdir, exist_ok=True)

    for f in funs:
        for inst in instances:
            for d in dims:
                budget = 10 * d
                problem_dir = os.path.join(base_outdir, f"f{f}_i{inst}_d{d}")
                os.makedirs(problem_dir, exist_ok=True)

                print(f"Running baseline qLogEI: f{f}, inst {inst}, dim {d}, budget {budget}, reps {reps}")

                for rep in range(reps):
                    print(f"  Rep {rep}")
                    y_obj, y_con, X = run_botorch_qlogei(
                        dim=d,
                        budget=budget,
                        coco_fun=f,
                        coco_instance=inst,
                        seed=seed + rep,
                    )

                    np.save(os.path.join(problem_dir, f"obj_rep{rep}.npy"), y_obj.detach().cpu().numpy())
                    np.save(os.path.join(problem_dir, f"cons_rep{rep}.npy"), y_con.detach().cpu().numpy())
                    np.save(os.path.join(problem_dir, f"X_rep{rep}.npy"), X)

                # Plot after finishing all reps
                print("  Plotting", problem_dir)
                try:
                    plot_problem(problem_dir, out_dir=problem_dir, show=False)
                except Exception as e:
                    print("  Plotting failed:", e)

                print("Saved and plotted:", problem_dir)
