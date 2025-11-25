# FuRBOStates.py
# FuRBO state initiate for PCA-based multi-TR sampling

from botorch.models.model_list_gp_regression import ModelListGP
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from utilities import get_best_index_for_batch, get_fitted_model

class variant_one():
    def __init__(self, obj, cons, batch_size, n_init, n_iteration,
                 tr_number, seed, history, iteration, samples_evaluated, **tkwargs):

        # Objective and constraint functions
        self.obj = obj
        self.lb, self.ub = obj.lower_bounds, obj.upper_bounds
        self.cons = cons

        # Problem dimension and batch parameters
        self.dim = obj.dimension
        self.batch_size = batch_size
        self.n_init = n_init
        self.n_iteration = n_iteration

        # Trust region parameters
        self.tr_number = tr_number
        self.radius = 1.0
        self.radius_min = 0.5**7
        self.success_tolerance = 2
        self.failure_tolerance = 3

        # Per-TR PCA-based information
        self.tr_center = torch.zeros((tr_number, self.dim), **tkwargs)
        self.tr_radii = torch.ones(tr_number, **tkwargs) * self.radius
        id_mats = [torch.eye(self.dim, **tkwargs) for _ in range(tr_number)]
        self.tr_R = torch.stack(id_mats, dim=0)  # Rotation matrices (tr_number x dim x dim)

        # Initialize TR centers with Sobol draws
        try:
            sobol_tmp = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
            self.tr_center = sobol_tmp.draw(tr_number).to(**tkwargs)
        except Exception:
            self.tr_center = torch.zeros((tr_number, self.dim), **tkwargs)

        # Per-TR performance tracking
        self.tr_success_counter = torch.zeros(tr_number, dtype=torch.int32)
        self.tr_failure_counter = torch.zeros(tr_number, dtype=torch.int32)

        # Per-TR bests
        self.tr_best_X = [None for _ in range(tr_number)]
        self.tr_best_Y = [None for _ in range(tr_number)]
        self.tr_best_C = [None for _ in range(tr_number)]

        # Global bests
        self.best_X = None
        self.best_Y = None
        self.best_C = None
        self.success_counter = 0
        self.failure_counter = 0

        # Iteration and stopping
        self.it_counter = iteration
        self.finish_trigger = False
        self.restart_trigger = False
        self.failed_GP = False
        self.history = history

        # Sample storage
        self.X = None
        self.Y = None
        self.C = None
        self.batch_X = None
        self.batch_Y = None
        self.batch_C = None
        self.samples_evaluated = samples_evaluated

        # Utilities
        self.seed = seed
        self.sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)

    def update(self, X_next, Y_next, C_next, **tkwargs):
        # Merge new samples
        if self.X is None:
            self.X = X_next.clone()
            self.Y = Y_next.clone()
            self.C = C_next.clone()
        else:
            self.X = torch.cat((self.X, X_next), dim=0)
            self.Y = torch.cat((self.Y, Y_next), dim=0)
            self.C = torch.cat((self.C, C_next), dim=0)

        # Fit GP models
        try:
            self.Y_model = get_fitted_model(self.X, self.Y, self.dim, max_cholesky_size=float("inf"))
            self.C_model = ModelListGP(*[
                get_fitted_model(self.X, C.reshape(-1,1), self.dim, max_cholesky_size=float("inf"))
                for C in self.C.t()])
            self.failed_GP = False
        except Exception:
            self.failed_GP = True

        # Update batch
        self.batch_X = X_next
        self.batch_Y = Y_next
        self.batch_C = C_next

        # Get per-TR best indices
        best_id = get_best_index_for_batch(n_tr=self.tr_number, Y=self.Y, C=self.C)
        per_tr_indices = None
        try:
            if hasattr(best_id, '__len__') and len(best_id) == self.tr_number:
                per_tr_indices = [int(x) for x in best_id]
            else:
                best_global_index = int(best_id)
        except Exception:
            best_global_index = int(best_id)

        # Update per-TR bests
        if per_tr_indices is not None:
            for j in range(self.tr_number):
                idx = per_tr_indices[j]
                if idx < 0 or idx >= self.X.shape[0]:
                    continue
                cand_X, cand_Y, cand_C = self.X[idx], self.Y[idx], self.C[idx]
                prev_Y, prev_C = self.tr_best_Y[j], self.tr_best_C[j]

                if prev_Y is None:
                    self.tr_best_X[j], self.tr_best_Y[j], self.tr_best_C[j] = cand_X.clone(), cand_Y.clone(), cand_C.clone()
                    continue

                if (cand_C <= 0).all():
                    if (prev_C > 0).any() or (cand_Y > prev_Y).any():
                        self.tr_best_X[j], self.tr_best_Y[j], self.tr_best_C[j] = cand_X.clone(), cand_Y.clone(), cand_C.clone()
                        self.tr_success_counter[j] += 1
                        self.tr_failure_counter[j] = 0
                    else:
                        self.tr_failure_counter[j] += 1
                else:
                    total_violation_new = cand_C.clamp(min=0).sum()
                    total_violation_prev = prev_C.clamp(min=0).sum()
                    if total_violation_new < total_violation_prev:
                        self.tr_best_X[j], self.tr_best_Y[j], self.tr_best_C[j] = cand_X.clone(), cand_Y.clone(), cand_C.clone()
                        self.tr_success_counter[j] += 1
                        self.tr_failure_counter[j] = 0
                    else:
                        self.tr_failure_counter[j] += 1

            # Update global best from TR bests
            tr_best_pairs = [(float(self.tr_best_Y[j].item()), j) for j in range(self.tr_number) if self.tr_best_Y[j] is not None]
            if len(tr_best_pairs) > 0:
                best_val, best_tr = max(tr_best_pairs, key=lambda t: t[0])
                self.best_X = self.tr_best_X[best_tr].clone()
                self.best_Y = self.tr_best_Y[best_tr].clone()
                self.best_C = self.tr_best_C[best_tr].clone()
            self.success_counter = int(self.tr_success_counter.sum().item())
            self.failure_counter = int(self.tr_failure_counter.sum().item())
        else:
            # Fallback: old single-best logic
            idx = best_global_index
            idx = min(max(idx, 0), self.X.shape[0]-1)
            cand_X, cand_Y, cand_C = self.X[idx], self.Y[idx], self.C[idx]
            if self.best_Y is None:
                self.best_X, self.best_Y, self.best_C = cand_X.clone(), cand_Y.clone(), cand_C.clone()
            elif (cand_C <= 0).all() and ((self.best_C > 0).any() or (cand_Y > self.best_Y).any()):
                self.best_X, self.best_Y, self.best_C = cand_X.clone(), cand_Y.clone(), cand_C.clone()

        # Update history
        event = {'iteration': self.it_counter,
                 'batch': {'X': self.batch_X, 'Y': self.batch_Y, 'C': self.batch_C},
                 'best': {'X': self.best_X, 'Y': self.best_Y, 'C': self.best_C},
                 'trust_region': {'tr_center': self.tr_center, 'tr_radii': self.tr_radii, 'tr_rotations': self.tr_R},
                 'performance': {'n_success_per_tr': self.tr_success_counter.clone(), 'n_failures_per_tr': self.tr_failure_counter.clone()},
                 'seed': self.seed}
        self.history.append(event)

        # Update counters
        self.it_counter += 1
        self.samples_evaluated += len(Y_next)
        # print(len(Y_next))
