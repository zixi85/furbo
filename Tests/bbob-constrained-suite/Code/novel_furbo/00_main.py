# Full code for modified FuRBO (iteration / restart logic matched to original main.py)
#
# March 2024 (modified version, iteration logic aligned with original main.py)
##########
# Imports
import cocoex  # experimentation module
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import random
import warnings
import time

from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from botorch.sampling.qmc import NormalQMCEngine

###
# Custom imports
import constraints
from FuRBOSamplingStrategies import get_initial_points_rotated_TR as get_initial_points
from FuRBOSamplingStrategies import generate_batch_thompson_sampling_rotated_TR as generate_batch
from FuRBOStates import variant_one
from FuRBOStopping import max_evaluations as stopping_criterion
from FuRBOTrustUpdate import multinormal_radius as update_tr
from FuRBORestart import min_radius as restart_criterion
import objectives
import plotting

##########
# Main code    

# Define COCO input
suite_name = "bbob-constrained"
suite = cocoex.Suite(suite_name, "", "")

# Load random seeds
seeds = np.load('random_seeds.npy')

# Base directory for saving files
cwd_base = os.path.join(os.getcwd(), 'results')
os.makedirs(cwd_base, exist_ok=True)

# General log file
f_gen = open(os.path.join(cwd_base, '00_GeneralLog.txt'), 'w')

functions_to_run = ['f002']
instances_to_run = ['i01']
dimensions_to_run = ['d10']
repetitions_per_instance = 5

for p in suite:

    func_id = p.id.split('_')[1]   # e.g., 'f002'
    instance_id = p.id.split('_')[2] # e.g., 'i01'
    dim_id = p.id.split('_')[3]      # e.g., 'd02'

    if func_id not in functions_to_run or instance_id not in instances_to_run or dim_id not in dimensions_to_run:
        continue

    print(f"Running problem {p.id}")
    print(f"{p.index}) {p.id}:", file=f_gen)
    print("\t Started", file=f_gen)

    # Create directory for problem
    cwd_current = os.path.join(cwd_base, p.id)
    if not os.path.exists(cwd_current):
        os.mkdir(cwd_current)

    # Log file
    f = open(os.path.join(cwd_current, '00_Log.txt'), 'w')
    print(f"{p.id}", file=f)
    print(f"{time.strftime('%x - %X')}: Start", file=f)

    # Start time evaluation
    tic = time.time()

    # Initialize FuRBO parameters for this problem
    batch_size = int(3 * p.dimension)
    n_init = int(3 * p.dimension)
    n_iteration = int(10 * p.dimension)
    tr_number = 3        # keep your modified value (original main used 1)
    N_CANDIDATES = 2000


    # Perform repetitions
    for i, seed in enumerate(seeds[:repetitions_per_instance]):

        filename_torch = p.id + '_it_' + str(i) + '.torch'
        if os.path.exists(os.path.join(cwd_current, filename_torch)):
            continue

        seed_j = 0
        FuRBO_seed = int(seed[seed_j])
        torch.manual_seed(FuRBO_seed)

        warnings.filterwarnings("ignore")

        device = torch.device("cpu")
        dtype = torch.double
        tkwargs = {"device": device, "dtype": dtype}

        # Initialize FuRBO parameters (match original)
        history = []
        iteration = 0
        n_samples = 0


        # FuRBO state initialization (first init)
        FuRBO_status = variant_one(
            obj=p,
            cons=p.constraint,
            batch_size=batch_size,
            n_init=n_init,
            n_iteration=n_iteration,
            tr_number=tr_number,
            seed=FuRBO_seed,
            history=history,
            iteration=iteration,
            samples_evaluated=n_samples,
            **tkwargs
        )
        global_iter = 0

        # === Main restart/iteration loop (matches original logic) ===
        while not FuRBO_status.finish_trigger:

            if FuRBO_status.restart_trigger:
                seed_j += 1
                FuRBO_seed = int(seed[seed_j])
                torch.manual_seed(FuRBO_seed)

            # Reinitialize FuRBO status for a restart (this mirrors the original main.py pattern)
            FuRBO_status = variant_one(
                obj=p,
                cons=p.constraint,
                batch_size=batch_size,
                n_init=n_init,
                n_iteration=n_iteration,
                tr_number=tr_number,
                seed=FuRBO_seed,
                history=history,
                iteration=iteration,
                samples_evaluated=n_samples,
                **tkwargs
            )
  

            # generate initial batch (Sobol or rotated TR version)
            X_next = get_initial_points(FuRBO_status, **tkwargs)

            # Optimization loop within current initialized state (until restart or finish)
            while not FuRBO_status.restart_trigger and not FuRBO_status.finish_trigger:
                
                # Evaluate batch
                Y_next = []
                C_next = []
                for x in X_next:
                    Y_next.append(-1 * FuRBO_status.obj(unnormalize(x, [p.lower_bounds, p.upper_bounds])))
                    C_next.append(FuRBO_status.cons(unnormalize(x, [p.lower_bounds, p.upper_bounds])))
                Y_next = torch.tensor(Y_next).unsqueeze(-1)
                C_next = torch.tensor(C_next)

                # print(f"[DEBUG] Raw batch Y_next: {Y_next.squeeze().cpu().numpy()}")
                # print(f"[DEBUG] Raw batch C_next: {C_next.cpu().numpy()}")

                # Update FuRBO status with evaluated batch
                FuRBO_status.update(X_next, Y_next, C_next, **tkwargs)
                print( FuRBO_status.samples_evaluated)
                global_iter += 1
                # ADD DEBUG HERE
                print(
                    f"[ITER {global_iter}] Best_Y: {FuRBO_status.best_Y}, "
                )

                # ----- Global best evaluation across all TRs -----
                # keep your original print/log logic but use FuRBO_status aggregated fields
                try:
                    # Many variants of FuRBO store per-TR bests differently; use safe access
                    if hasattr(FuRBO_status, 'best_C') and hasattr(FuRBO_status, 'best_Y'):
                        if (FuRBO_status.best_C <= 0).all():
                            best = FuRBO_status.best_Y.amax()
                            print(f"{FuRBO_status.it_counter-1}) Best value: {best:.2e}, MG radius: {FuRBO_status.radius}", file=f)
                            print(f"{FuRBO_status.it_counter-1}) Best value: {best:.2e}, MG radius: {FuRBO_status.radius}")
                        else:
                            violation = FuRBO_status.best_C.clamp(min=0).sum()
                            print(f"{FuRBO_status.it_counter-1}) No feasible point yet! Smallest total violation: "
                                  f"{violation:.2e}, MG radius: {FuRBO_status.radius}", file=f)
                            print(f"{FuRBO_status.it_counter-1}) No feasible point yet! Smallest total violation: "
                                  f"{violation:.2e}, MG radius: {FuRBO_status.radius}")
                    else:
                        # Fallback to aggregated per-TR bests stored in FuRBO_status.tr_best_Y / tr_best_C if present
                        best_Y_list = [y for y in getattr(FuRBO_status, 'tr_best_Y', []) if y is not None]
                        best_C_list = [c for c in getattr(FuRBO_status, 'tr_best_C', []) if c is not None]
                        if best_Y_list:
                            feasible_idx = [ii for ii, c in enumerate(best_C_list) if (c <= 0).all()]
                            if feasible_idx:
                                best_global_Y = max([best_Y_list[ii] for ii in feasible_idx])
                                print(f"{FuRBO_status.it_counter-1}) Best feasible value: {best_global_Y:.2e}, MG radius: {FuRBO_status.radius}", file=f)
                                print(f"{FuRBO_status.it_counter-1}) Best feasible value: {best_global_Y:.2e}, MG radius: {FuRBO_status.radius}")
                            else:
                                best_global_Y = max(best_Y_list)
                                print(f"{FuRBO_status.it_counter-1}) No feasible point yet! Best (infeasible) value: {best_global_Y:.2e}, MG radius: {FuRBO_status.radius}", file=f)
                                print(f"{FuRBO_status.it_counter-1}) No feasible point yet! Best (infeasible) value: {best_global_Y:.2e}, MG radius: {FuRBO_status.radius}")
                except Exception:
                    # Avoid breaking if status doesn't have attributes; continue
                    pass

                # Optional: print current batch results
                # print(f"Y batch: {Y_next}")
                # print(f"C batch: {C_next}")

                # Update Trust regions
                FuRBO_status = update_tr(FuRBO_status, **tkwargs)

                # Generate new batch
                X_next = generate_batch(FuRBO_status, N_CANDIDATES, **tkwargs)

                # Update stopping/restart triggers (exact same calls as original)
                FuRBO_status.finish_trigger = stopping_criterion(FuRBO_status)
                FuRBO_status.restart_trigger = restart_criterion(FuRBO_status)



            # After inner loop (either restart or finish), persist history and counters like original
            history = FuRBO_status.history
            iteration = FuRBO_status.it_counter
            print(f"the iteration is : {iteration}")
            n_samples = FuRBO_status.samples_evaluated
            print(f"het aantal samples: {n_samples}")

        print(
        f"[DEBUG] Saving run {i}: Total history entries = {len(FuRBO_status.history)} "
        f"(max allowed = {n_iteration}), samples = {n_samples}",
        file=f
        )
        # Save history (same filename convention)
        torch.save(FuRBO_status.history, os.path.join(cwd_current, filename_torch))
        t = (time.time() - tic) % 60
        print(f"{time.strftime('%x - %X')}: Finish", file=f)
        print(f"Computation time: {t:.2f} seconds", file=f)
        print(f"\t Completed - time: {t:.2f} seconds", file=f_gen)
        del FuRBO_status

    # ----- Post-processing (aligned with original main.py) -----

    states = []
    for torch_file in os.listdir(cwd_current):
        if torch_file.endswith('.torch'):
            full_path = os.path.join(cwd_current, torch_file)
            obj = torch.load(full_path, map_location="cpu")
            states.append(obj)   # store history (list of events)

    Y_batch = []
    C_batch = []

    for r, state in enumerate(states):
        Y_batch.append(np.concatenate([-1 * event['batch']['Y'].cpu().numpy() for event in state]).reshape(-1)[:n_iteration])
        C_batch.append(np.concatenate([np.max(event['batch']['C'].cpu().numpy(), axis=1) for event in state])[:n_iteration])
            
        # print(f"[DEBUG] Repetition {r}: Last 10 Ys before truncation/padding: {Y_batch[-10:]}")
        # print(f"[DEBUG] Repetition {r}: Last 10 Cs before truncation/padding: {C_batch[-10:]}")

    Y_best = np.array(Y_batch)
    C_best = np.array(C_batch)

    Y_f = np.copy(Y_best)
    C_f = np.copy(C_best)

    Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = 0
    Y_f[np.where(C_f > 0)[0], np.where(C_f > 0)[1]] = np.amax(Y_f)

    Y_f_monotonic = []
    for YY in Y_f:
        y_mono = []
        for yy in YY:
            if len(y_mono) == 0:
                y_mono = [yy]
            else:
                if yy < y_mono[-1]:
                    y_mono.append(yy)
                else:
                    y_mono.append(y_mono[-1])
        
        Y_f_monotonic.append(y_mono)
        
    Y_f_monotonic = np.array(Y_f_monotonic)
    

    # Save results same filenames as original
    np.save(os.path.join(cwd_current, '01_Y_mono.npy'), Y_f_monotonic)
    np.save(os.path.join(cwd_current, '02_Y_best.npy'), Y_best)
    np.save(os.path.join(cwd_current, '02_C_best.npy'), C_best)

    # Mark complete
    open(os.path.join(cwd_current, 'complete'), 'w').close()
