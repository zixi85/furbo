import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.objective import ConstrainedMCObjective
# Acquisition: prefer qLogExpectedImprovement, fall back to qExpectedImprovement
try:
    from botorch.acquisition.monte_carlo import qLogExpectedImprovement as _qAcq
    _USE_LOG_ACQ = True
except Exception:
    from botorch.acquisition.monte_carlo import qExpectedImprovement as _qAcq
    _USE_LOG_ACQ = False
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import ModelFittingError

import numpy as np

# COCO objective & constraint
from objectives import evaluate_objective
from constraints import evaluate_constraints


def safe_fit_gp(gp, y_name="obj"):
    """Robust wrapper around fit_gpytorch_mll to avoid hard crashes."""
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except ModelFittingError as e:
        # 打个招呼但继续跑，用当前（初始化）超参数
        print(f"[WARN] Fitting GP for '{y_name}' failed with ModelFittingError. "
              f"Continuing with initial hyperparameters. Details: {e}")
    except RuntimeError as e:
        # 一些底层 Cholesky 之类的错误
        print(f"[WARN] RuntimeError while fitting GP for '{y_name}'. "
              f"Continuing with initial hyperparameters. Details: {e}")


def run_botorch_qlogei(dim, budget, coco_fun, coco_instance, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ==== 1. initial design ====
    n_init = min(5 * dim, 10)
    X = np.random.uniform(-5, 5, size=(n_init, dim))
    obj_vals = []
    cons_vals = []

    for x in X:
        f = evaluate_objective(x, coco_fun, coco_instance, dim=dim)
        g = evaluate_constraints(x, coco_fun, coco_instance, dim=dim)
        obj_vals.append(f)
        cons_vals.append(g)

    X_t = torch.tensor(X, dtype=torch.double)
    y_obj = torch.tensor(obj_vals, dtype=torch.double).unsqueeze(-1)  # (n,1)
    y_con = torch.tensor(cons_vals, dtype=torch.double)
    # Ensure constraints tensor is 2D: (n_samples, n_constraints)
    if y_con.dim() == 1:
        y_con = y_con.unsqueeze(1)

    # ==== 2. Build initial GPs ====
    # objective GP (with Standardize to stabilize scale)
    gp_obj = SingleTaskGP(
        X_t,
        y_obj,
        outcome_transform=Standardize(m=1),
    )
    safe_fit_gp(gp_obj, y_name="objective")

    # constraints GPs
    gp_cons = []
    for j in range(y_con.shape[1]):
        gp = SingleTaskGP(
            X_t,
            y_con[:, j:j+1],
            outcome_transform=Standardize(m=1),
        )
        safe_fit_gp(gp, y_name=f"constraint_{j}")
        gp_cons.append(gp)

    # ModelList: objective + constraints
    model = ModelListGP(gp_obj, *gp_cons)

    # ==== 3. Define constraint-aware MC objective ====
    # samples 的最后一维：[f, g1, g2, ...]
    def obj_from_samples(samples, X=None, **kwargs):
        # 第 0 列是 objective
        return samples[..., 0]

    cons_funcs = []
    # 第 j 个约束对应 samples[..., j+1]，约束形式为 g_j(x) <= 0
    # build constraint callables that accept optional kwargs (some BoTorch versions pass X)
    def _make_cons(jj):
        return lambda samples, X=None, **kwargs: samples[..., jj + 1]

    for j in range(y_con.shape[1]):
        cons_funcs.append(_make_cons(j))

    objective = ConstrainedMCObjective(
        objective=obj_from_samples,
        constraints=cons_funcs,
    )

    # ==== 4. Optimization loop ====
    eval_count = n_init

    while eval_count < budget:
        # acquisition (try to construct sampler if available)
        sampler = None
        try:
            from botorch.sampling import SobolQMCNormalSampler
            sampler = SobolQMCNormalSampler(num_samples=256)
        except Exception:
            try:
                from botorch.sampling.qmc import NormalQMCEngine
                sampler = NormalQMCEngine(sample_shape=torch.Size([256]))
            except Exception:
                sampler = None

        # BoTorch 新版 qEI / qLogEI 不需要 q 参数，q 由 optimize_acqf 的 q 决定
        acq_kwargs = {
            'model': model,
            'best_f': y_obj.min(),
            'objective': objective,
        }
        if sampler is not None:
            acq_kwargs['sampler'] = sampler

        ei = _qAcq(**acq_kwargs)

        # optimize acquisition
        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.double),
            q=1,
            num_restarts=5,
            raw_samples=50,
        )

        # `optimize_acqf` 有时会多一两个 batch 维度，统一成 (1, d)
        candidate = candidate.detach()
        if candidate.dim() > 2:
            candidate = candidate.reshape(-1, candidate.size(-1))
        candidate = candidate[0].unsqueeze(0)  # 取第一个 q 点
        candidate = candidate.to(dtype=X_t.dtype, device=X_t.device)

        x_new = candidate.cpu().numpy()[0]

        # evaluate on COCO
        f_new = evaluate_objective(x_new, coco_fun, coco_instance, dim=dim)
        g_new = evaluate_constraints(x_new, coco_fun, coco_instance, dim=dim)

        # append new data
        X_t = torch.cat([X_t, candidate], dim=0)
        y_obj = torch.cat(
            [y_obj, torch.tensor([[f_new]], dtype=torch.double)],
            dim=0,
        )
        g_tensor = torch.tensor([g_new], dtype=torch.double)
        if g_tensor.dim() == 1:
            g_tensor = g_tensor.unsqueeze(0)  # (1, n_constraints)
        y_con = torch.cat([y_con, g_tensor], dim=0)

        # ==== 5. 更新 GPs ====
        gp_obj = SingleTaskGP(
            X_t,
            y_obj,
            outcome_transform=Standardize(m=1),
        )
        safe_fit_gp(gp_obj, y_name="objective")

        gp_cons = []
        for j in range(y_con.shape[1]):
            gp = SingleTaskGP(
                X_t,
                y_con[:, j:j+1],
                outcome_transform=Standardize(m=1),
            )
            safe_fit_gp(gp, y_name=f"constraint_{j}")
            gp_cons.append(gp)

        # 重新拼 joint model
        model = ModelListGP(gp_obj, *gp_cons)

        eval_count += 1

    return y_obj, y_con, X_t.numpy()
