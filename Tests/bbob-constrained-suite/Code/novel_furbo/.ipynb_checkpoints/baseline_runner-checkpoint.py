from baseline_botorch_qlogei import run_botorch_qlogei
import numpy as np
import pickle
import os

def run_baseline_all():
    funs = [50,52,54]
    instances = [0,1,2]
    dims = [2,10]

    for f in funs:
        for inst in instances:
            for d in dims:
                budget = 30 * d

                print(f"Running baseline qLogEI: f{f}, inst {inst}, dim {d}")
                y_obj, y_con, X = run_botorch_qlogei(
                    dim=d,
                    budget=budget,
                    coco_fun=f,
                    coco_instance=inst,
                    seed=0
                )

                save_dir = f"results/baseline_qlogei/f{f}_i{inst}_d{d}"
                os.makedirs(save_dir, exist_ok=True)

                np.save(f"{save_dir}/obj.npy", y_obj.numpy())
                np.save(f"{save_dir}/cons.npy", y_con.numpy())
                np.save(f"{save_dir}/X.npy", X)

                print("Saved:", save_dir)
