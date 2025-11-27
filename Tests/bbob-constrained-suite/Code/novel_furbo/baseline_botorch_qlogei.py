import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.objective import ConstrainedMCObjective
try:
    from botorch.acquisition.monte_carlo import qLogExpectedImprovement as _qAcq
except Exception:
    from botorch.acquisition.monte_carlo import qExpectedImprovement as _qAcq
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from objectives import evaluate_objective
from constraints import evaluate_constraints


def safe_fit_gp(gp, name="gp"):
    """Fit GP safe mode."""
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"[WARN] GP fit failed on {name}, continuing. Error: {e}")


def run_botorch_qlogei(dim, budget, coco_fun, coco_instance, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------
    # 1. Initial Design
    # -----------------------------
    n_init = min(10, 5 * dim)
    X = np.random.uniform(-5, 5, size=(n_init, dim))

    obj_vals = []
    con_vals = []

    for x in X:
        obj_vals.append(evaluate_objective(x, coco_fun, coco_instance, dim))
        con_vals.append(evaluate_constraints(x, coco_fun, coco_instance, dim))

    X_t = torch.tensor(X, dtype=torch.double)
    y_obj = torch.tensor(obj_vals, dtype=torch.double).unsqueeze(-1)
    y_con = torch.tensor(con_vals, dtype=torch.double)

    if y_con.dim() == 1:
        y_con = y_con.unsqueeze(1)

    # -----------------------------
    # 2. Fit initial GP models
    # -----------------------------
    gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
    safe_fit_gp(gp_obj, "objective")

    gp_cons = []
    for j in range(y_con.shape[1]):
        gpj = SingleTaskGP(
            X_t,
            y_con[:, j:j+1],
            outcome_transform=Standardize(m=1),
        )
        safe_fit_gp(gpj, f"constraint_{j}")
        gp_cons.append(gpj)

    model = ModelListGP(gp_obj, *gp_cons)

    # -----------------------------
    # 3. Constrained MC Objective
    # -----------------------------
    def obj_from_samples(samples, X=None, **kwargs):
        return samples[..., 0]

    def make_con(j):
        # Syntax-correct lambda
        return lambda samples, j=j, X=None, **kwargs: samples[..., j + 1]

    constraints = [make_con(j) for j in range(y_con.shape[1])]

    objective = ConstrainedMCObjective(
        objective=obj_from_samples,
        constraints=constraints,
    )

    # -----------------------------
    # 4. BO Loop
    # -----------------------------
    eval_count = n_init
    bounds = torch.tensor(
        [[-5.0] * dim, [5.0] * dim],
        dtype=torch.double,
    )

    while eval_count < budget:

        # best_f from feasible points
        feas = torch.all(y_con <= 0, dim=1)
        if torch.any(feas):
            best_f = y_obj[feas].min()
        else:
            best_f = y_obj.min()

        # acquisition
        acq = _qAcq(
            model=model,
            best_f=best_f,
            objective=objective,
        )

        # optimize q=1
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )

        candidate = candidate.reshape(1, -1)
        x_new = candidate.detach().cpu().numpy()[0]

        # evaluate new point
        f_new = evaluate_objective(x_new, coco_fun, coco_instance, dim)
        g_new = evaluate_constraints(x_new, coco_fun, coco_instance, dim)

        # update dataset
        X_t = torch.cat([X_t, candidate], dim=0)
        y_obj = torch.cat([y_obj, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        g_new_t = torch.tensor([g_new], dtype=torch.double)
        if g_new_t.dim() == 1:
            g_new_t = g_new_t.unsqueeze(0)
        y_con = torch.cat([y_con, g_new_t], dim=0)

        # refit GPs
        gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
        safe_fit_gp(gp_obj, "objective")

        gp_cons = []
        for j in range(y_con.shape[1]):
            gpj = SingleTaskGP(
                X_t,
                y_con[:, j:j+1],
                outcome_transform=Standardize(m=1),
            )
            safe_fit_gp(gpj, f"constraint_{j}")
            gp_cons.append(gpj)

        model = ModelListGP(gp_obj, *gp_cons)
        eval_count += 1

    return y_obj, y_con, X_t.numpy()
