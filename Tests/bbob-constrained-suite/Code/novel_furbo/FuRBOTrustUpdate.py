# FuRBOTrustUpdate.py
# FuRBO trust region updates for different loops (PCA-based multi-TR)
#
# March 2024 (updated)
##########
# Imports
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.generation.sampling import MaxPosteriorSampling

import math
import torch

###
# Custom imports
from utilities import multivariate_circular_two
from utilities import multivariate_circular
from utilities import get_fitted_model

# -------------------------------------------------------------------
# Keep your existing helpers (fixed_percentage & changing_percentage)
# -------------------------------------------------------------------
def fixed_percentage(state,              # FuRBO state
                     percentage,         # Percentage to take (value 0 - 1)
                     **tkwargs
                     ):
    
    # Update the trust regions based on the feasible region
    n_samples = 1000 * state.dim
    lb = torch.zeros(state.dim, **tkwargs)
    ub = torch.ones(state.dim, **tkwargs)
    
    # If state.best_X is a per-TR list (new state), iterate accordingly
    best_list = state.best_X if hasattr(state, 'best_X') and isinstance(state.best_X, (list, tuple)) else [state.best_X]

    for ind, x_candidate in enumerate(best_list):
        if x_candidate is None:
            continue

        # Generate the samples to evaluathe the feasible area on
        samples = multivariate_circular_two(x_candidate, n_samples, lb=lb, ub=ub, **tkwargs)
        
        # Evaluate samples on the models of the objective -> yy Tensor
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()
        
        # Evaluate samples on the models of the constraints -> yy Tensor
        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean
        
        # Combine the constraints values: Normalize and reduce to scalar per sample
        samples_cc = samples_cc / (torch.abs(samples_cc).max(dim=0).values + 1e-12)
        samples_cc = torch.max(samples_cc, dim=1).values
        
        # Take the best portion of drawn samples
        n_samples_tr = max(int(n_samples * percentage), 4)
        if torch.any(samples_cc < 0):
            feasible_samples_id = torch.where(samples_cc <= 0)[0]
            infeasible_samples_id = torch.where(samples_cc > 0)[0]
            
            feasible_cc = samples_yy[feasible_samples_id]
            infeasible_cc = samples_cc[infeasible_samples_id]
            
            feasible_sorted, feasible_sorted_id = torch.sort(feasible_cc)
            infeasible_sorted, infeasible_sorted_id = torch.sort(infeasible_cc)
            
            original_feasible_sorted_indices = feasible_samples_id[feasible_sorted_id]
            original_infeasible_sorted_indices = infeasible_samples_id[infeasible_sorted_id]
            
            top_indices = torch.cat((original_feasible_sorted_indices, original_infeasible_sorted_indices))[:n_samples_tr]
        else:
            if n_samples_tr > len(samples_cc):
                n_samples_tr = len(samples_cc)
            top_values, top_indices = torch.topk(samples_cc, n_samples_tr, largest=False)
        
        # Set the axis-aligned box bounds (compatibility)
        if hasattr(state, 'tr_lb') and hasattr(state, 'tr_ub'):
            state.tr_lb[ind] = torch.min(samples[top_indices], dim=0).values
            state.tr_ub[ind] = torch.max(samples[top_indices], dim=0).values
            # Update volume
            state.tr_vol[ind] = torch.prod(state.tr_ub[ind] - state.tr_lb[ind])
        
    # return updated status with new trust regions
    return state

def changing_percentage(state,
                        **tkwargs):
    
    if hasattr(state, 'success_counter') and state.success_counter == state.success_tolerance:  # Expand trust region
        state.percentage = min(4.0 * state.percentage, 1.0)
        state.success_counter = 0
    elif hasattr(state, 'failure_counter') and state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.percentage /= 4.0
        state.failure_counter = 0
        
    return fixed_percentage(state,
                            state.percentage,
                            **tkwargs)

# -------------------------------------------------------------------
# Helper: PCA on samples
# -------------------------------------------------------------------
def _compute_pca_from_samples(samples: torch.Tensor, eps: float = 1e-8):
    """
    samples: (n_samples, d) in [0,1] space
    returns: mu (d,), R (d,d) rotation matrix (columns = principal directions), eigvals (d,)
    """
    device = samples.device
    dtype = samples.dtype
    mu = samples.mean(dim=0)
    Xc = samples - mu
    n = samples.shape[0]
    d = samples.shape[1]
    if n <= 1:
        return mu, torch.eye(d, dtype=dtype, device=device), torch.ones(d, dtype=dtype, device=device)
    # covariance
    C = (Xc.t() @ Xc) / max(n - 1, 1)
    C = C + eps * torch.eye(d, device=device, dtype=dtype)
    eigvals, eigvecs = torch.linalg.eigh(C)  # ascending
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    return mu, eigvecs, eigvals

def _make_radii_from_eigvals(eigvals: torch.Tensor, base_scale: float, min_length: float, max_length: float):
    """
    Convert eigenvalues into per-axis half-radii in z-space.
    - base_scale scales how wide the TR is relative to sqrt(eigvals).
    - clamp to [min_length, max_length]
    """
    lengths = base_scale * torch.sqrt(torch.clamp(eigvals, min=1e-12))
    lengths = lengths.clamp(min=min_length, max=max_length)
    return lengths

# -------------------------------------------------------------------
# Modified generate_batch_one: respects rotation matrices if present
# -------------------------------------------------------------------
def generate_batch_one(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol,
    **tkwargs):
    
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):
        # If rotated TRs exist, use them; else fallback to axis aligned tr_lb/tr_ub
        use_rotated = hasattr(state, 'tr_R') and state.tr_R is not None and state.tr_R.shape[0] == state.tr_number

        dim = X.shape[-1]

        if use_rotated:
            R = state.tr_R[i]               # (d,d)
            mu = state.tr_center[i]         # (d,)
            radii = state.tr_radii[i]       # (d,) half-lengths in z-space

            # sample z in [-1,1]^d via sobol then scale
            z_unit = 2.0 * sobol.draw(n_candidates).to(**tkwargs) - 1.0  # in [-1,1]
            # Ensure radii has proper shape
            if radii.dim() == 0:
                radii = radii.repeat(dim)
            z_pert = z_unit * radii.unsqueeze(0)   # scale per-axis
            # Transform back to x
            x_pert = (R @ z_pert.t()).t() + mu.unsqueeze(0)
            x_pert = torch.clamp(x_pert, 0.0, 1.0)

            # perturbation mask logic (same as before)
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind) > 0:
                mask[ind, torch.randint(0, dim, size=(len(ind),), **tkwargs)] = 1

            X_cand = state.best_batch_X[i].expand(n_candidates, dim).clone()
            X_cand[mask] = x_pert[mask]

        else:
            tr_lb = state.tr_lb[i]
            tr_ub = state.tr_ub[i]

            pert = sobol.draw(n_candidates).to(**tkwargs)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind) > 0:
                mask[ind, torch.randint(0, dim - 1, size=(len(ind),), **tkwargs)] = 1

            X_cand = state.best_batch_X[i].expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]
        
        # Use constrained sampling or constraint-minimizing sampling as before
        if torch.any(torch.max(C, dim=1).values <= 0):
            constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
                model=model, constraint_model=constraint_model, replacement=False
                )
            with torch.no_grad():
                X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=batch_size)
        else:
            constraint_model.eval()
            with torch.no_grad():
                posterior = constraint_model.posterior(X_cand)
                C_cand = posterior.mean

            C_cand = C_cand / (torch.abs(C_cand).max(dim=0).values + 1e-12)
            C_cand = -1 * C_cand.max(dim=1).values
            C_cand = C_cand.view(-1, 1)
            constraint_model_united = get_fitted_model(X_cand, C_cand)
            
            constraint_sampling = MaxPosteriorSampling(
                model=constraint_model_united, replacement=False)
            with torch.no_grad():
                X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constraint_sampling(X_cand, num_samples=batch_size)

    return X_next

# -------------------------------------------------------------------
# New: PCA-based multi-TR updater (keeps name multinormal_radius for compatibility)
# -------------------------------------------------------------------
def multinormal_radius(state,              # FuRBO state
                       n_samples_factor: int = 1000,
                       percentage: float = 0.1,
                       base_scale: float = 2.0,
                       min_axis_fraction: float = 1e-3,
                       max_axis_fraction: float = 0.5,
                       eps: float = 1e-8,
                       **tkwargs
                       ):
    """
    Adaptive multi-TR updater using local PCA on promising surrogate samples.
    - n_samples_factor: number of surrogate samples = n_samples_factor * dim
    - percentage: fraction of top local samples to use for PCA
    - base_scale: scales eigenvalue -> axis lengths
    - min_axis_fraction/max_axis_fraction: clamp axis lengths relative to domain (normalized to [0,1])
    """
    d = state.dim
    n_samples = max(4, n_samples_factor * d)
    lb = torch.zeros(d, **tkwargs)
    ub = torch.ones(d, **tkwargs)

    # Lazy-initialize per-TR attributes if missing
    if not hasattr(state, 'tr_center') or state.tr_center is None:
        try:
            sobol_tmp = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=state.seed)
            state.tr_center = sobol_tmp.draw(state.tr_number).to(**tkwargs)
        except Exception:
            state.tr_center = torch.zeros((state.tr_number, d), **tkwargs)
    if not hasattr(state, 'tr_radii') or state.tr_radii is None:
        state.tr_radii = torch.ones(state.tr_number, **tkwargs) * state.radius
    if not hasattr(state, 'tr_R') or state.tr_R is None:
        id_mats = [torch.eye(d, **tkwargs) for _ in range(state.tr_number)]
        state.tr_R = torch.stack(id_mats, dim=0)

    # Update global radius using old global counters (keeps compatibility)
    if hasattr(state, 'success_counter') and state.success_counter == state.success_tolerance:
        state.radius = min(2.0 * state.radius, 1.0)
        state.success_counter = 0
    elif hasattr(state, 'failure_counter') and state.failure_counter == state.failure_tolerance:
        state.radius = max(state.radius / 2.0, eps)
        state.failure_counter = 0

    # Domain clamping values (normalized domain)
    min_axis_length = min_axis_fraction
    max_axis_length = max_axis_fraction

    # Decide centers to iterate over: prefer state.tr_center if available, else fallback to state.best_X (list or single)
    if hasattr(state, 'tr_center') and state.tr_center is not None:
        centers_iter = [state.tr_center[i] for i in range(state.tr_number)]
    else:
        # support older style where best_X is a list or single
        if isinstance(state.best_X, (list, tuple)):
            centers_iter = state.best_X
        else:
            centers_iter = [state.best_X for _ in range(state.tr_number)]

    # For each trust region index
    for ind in range(state.tr_number):
        x_candidate = centers_iter[ind]
        if x_candidate is None:
            # fallback: use global center at 0.5
            x_candidate = 0.5 * torch.ones(d, **tkwargs)

        # Draw local samples around x_candidate in a multivariate circular neighborhood
        samples = multivariate_circular(x_candidate, state.radius, n_samples, lb=lb, ub=ub, **tkwargs)

        # Evaluate surrogate models for objective and constraints
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()

        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean

        # Normalize constraint outputs and reduce to scalar
        samples_cc = samples_cc / (torch.abs(samples_cc).max(dim=0).values + eps)
        samples_cc = torch.max(samples_cc, dim=1).values

        

        # Select best samples: prefer feasible then by objective
        n_keep = max(int(n_samples * percentage), 4)
        if torch.any(samples_cc < 0):
            feasible_idx = torch.where(samples_cc <= 0)[0]
            infeasible_idx = torch.where(samples_cc > 0)[0]

            feasible_vals = samples_yy[feasible_idx]
            infeasible_vals = samples_cc[infeasible_idx]

            feasible_sorted_ids = torch.argsort(feasible_vals)  # ascending (smaller is better)
            infeasible_sorted_ids = torch.argsort(infeasible_vals)

            chosen_idx = torch.cat([
                feasible_idx[feasible_sorted_ids],
                infeasible_idx[infeasible_sorted_ids]
            ])[:n_keep]
        else:
            top_vals, top_idx = torch.topk(samples_yy, k=min(n_keep, len(samples_yy)), largest=False)
            chosen_idx = top_idx

        chosen_samples = samples[chosen_idx]

        # Compute PCA from chosen samples
        mu, R, eigvals = _compute_pca_from_samples(chosen_samples, eps=eps)

        # Convert eigvals->axis lengths and clamp to domain fractions
        axis_lengths = _make_radii_from_eigvals(eigvals, base_scale=base_scale,
                                                min_length=min_axis_length,
                                                max_length=max_axis_length)
        # axis_lengths is (d,) in normalized units
        axis_lengths = axis_lengths.to(dtype=mu.dtype, device=mu.device)

        # Store results back to state
        # tr_R: (tr_number, d, d), tr_center: (tr_number, d), tr_radii: (tr_number, d)
        state.tr_R[ind] = R
        state.tr_center[ind] = mu
        # ensure tr_radii can store per-dim lengths; convert to shape (d,)
        # If tr_radii was scalar per TR, replace with per-dim vector
        if state.tr_radii.ndim == 1 and state.tr_radii.shape[0] == state.tr_number:
            # currently tr_radii stores one scalar per TR; expand to per-dim arrays by rewriting whole field
            # Build new tensor shape (tr_number, d) by repeating existing scalars
            existing = state.tr_radii.clone()
            new_radii = torch.stack([existing[i].repeat(d) for i in range(state.tr_number)], dim=0).to(dtype=axis_lengths.dtype, device=axis_lengths.device)
            state.tr_radii = new_radii
        # assign per-axis lengths
        state.tr_radii[ind] = axis_lengths

        # Backwards compatibility: compute axis-aligned bounding box of chosen_samples (or transform z-box corners)
        # We compute z-box corners based on axis_lengths and map back to x to get approximate lb/ub
        z_lb = -axis_lengths
        z_ub = axis_lengths
        corners = []
        for mask in range(1 << d):
            bits = [(mask >> k) & 1 for k in range(d)]
            z_corner = torch.tensor([z_ub[k] if bits[k] else z_lb[k] for k in range(d)], dtype=mu.dtype, device=mu.device)
            x_corner = (R @ z_corner) + mu
            corners.append(x_corner)
        corners = torch.stack(corners, dim=0)
        if hasattr(state, 'tr_lb') and hasattr(state, 'tr_ub'):
            state.tr_lb[ind] = torch.clamp(torch.min(corners, dim=0).values, 0.0, 1.0)
            state.tr_ub[ind] = torch.clamp(torch.max(corners, dim=0).values, 0.0, 1.0)
            state.tr_vol[ind] = torch.prod(state.tr_ub[ind] - state.tr_lb[ind])

        # Update per-TR success/failure counters based on best sample inside chosen_idx
        # Find the best candidate among chosen_idx (prefer feasible)
        if len(chosen_idx) == 0:
            continue
        local_cc = samples_cc[chosen_idx]
        local_yy = samples_yy[chosen_idx]
        # pick best index: feasible and lower objective
        if torch.any(local_cc <= 0):
            feasible_ids = torch.where(local_cc <= 0)[0]
            best_local_rel = feasible_ids[torch.argmin(local_yy[feasible_ids])]
            best_global_idx = chosen_idx[best_local_rel]
        else:
            best_rel = torch.argmin(local_cc)
            best_global_idx = chosen_idx[best_rel]

        cand_Y = samples_yy[best_global_idx]
        cand_C = samples_cc[best_global_idx]

        # Ensure per-TR counters exist
        if not hasattr(state, 'tr_success_counter'):
            state.tr_success_counter = torch.zeros(state.tr_number, dtype=torch.int32)
        if not hasattr(state, 'tr_failure_counter'):
            state.tr_failure_counter = torch.zeros(state.tr_number, dtype=torch.int32)

        # Compare with stored per-TR best if exists (fall back to global logic otherwise)
        if hasattr(state, 'tr_best_Y') and state.tr_best_Y[ind] is not None:
            prev_C = state.tr_best_C[ind]
            prev_Y = state.tr_best_Y[ind]
            # Prefer feasibility
            if cand_C <= 0:
                if (prev_C > 0).any() or (cand_Y > prev_Y).any():
                    state.tr_success_counter[ind] += 1
                    state.tr_failure_counter[ind] = 0
                    state.tr_best_X[ind] = samples[best_global_idx].clone()
                    state.tr_best_Y[ind] = cand_Y.clone()
                    state.tr_best_C[ind] = cand_C.clone()
                else:
                    state.tr_failure_counter[ind] += 1
            else:
                # infeasible candidate: compare violation (we used normalized scalar cand_C)
                total_violation_new = cand_C.clamp(min=0).sum()
                total_violation_prev = prev_C.clamp(min=0).sum()
                if total_violation_new < total_violation_prev:
                    state.tr_success_counter[ind] += 1
                    state.tr_failure_counter[ind] = 0
                    state.tr_best_X[ind] = samples[best_global_idx].clone()
                    state.tr_best_Y[ind] = cand_Y.clone()
                    state.tr_best_C[ind] = cand_C.clone()
                else:
                    state.tr_failure_counter[ind] += 1
        else:
            # initialize per-TR bests
            if not hasattr(state, 'tr_best_X'):
                state.tr_best_X = [None for _ in range(state.tr_number)]
                state.tr_best_Y = [None for _ in range(state.tr_number)]
                state.tr_best_C = [None for _ in range(state.tr_number)]
            state.tr_best_X[ind] = samples[best_global_idx].clone()
            state.tr_best_Y[ind] = cand_Y.clone()
            state.tr_best_C[ind] = cand_C.clone()

    # Optional: update aggregate global counters for compatibility
    try:
        state.success_counter = int(state.tr_success_counter.sum().item())
        state.failure_counter = int(state.tr_failure_counter.sum().item())
    except Exception:
        pass

    return state
