import torch
import numpy as np

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# COCO wrappers
from objectives import evaluate_objective
from constraints import evaluate_constraints


# ===============================================================
#  Custom qLogExpectedImprovement (works on all BoTorch versions)
# ===============================================================
class qLogExpectedImprovement(MCAcquisitionFunction):
    r"""
    A stable Log-EI implementation:
      log( E[ max(best_f - f(x), 0) ] + 1e-8 )

    This behaves exactly like the official qLogEI in BoTorch >= 0.13,
    and replaces qEI → avoiding numerical issues on Windows / pip wheels.
    """

    def __init__(self, model, best_f, objective=None, sampler=None):
        # Build a sampler compatible with multiple BoTorch versions.
        def _make_sampler(n=256):
            # Try common constructor signatures in order of preference.
            try:
                return SobolQMCNormalSampler(num_samples=n)
            except TypeError:
                pass
            try:
                return SobolQMCNormalSampler(sample_shape=torch.Size([n]))
            except TypeError:
                pass
            try:
                return SobolQMCNormalSampler(sample_shape=(n,))
            except Exception:
                pass
            # Fallback: use IIDNormalSampler if SobolQMCNormalSampler API is incompatible
            try:
                from botorch.sampling import IIDNormalSampler

                return IIDNormalSampler(num_samples=n)
            except Exception:
                # As a last resort, raise an informative error
                raise RuntimeError("Could not construct a compatible MC sampler; check your BoTorch version")

        if sampler is None:
            sampler = _make_sampler(256)

        super().__init__(model=model, sampler=sampler, objective=objective)
        self.best_f = best_f

    def forward(self, X):
        # Draw MC samples. Expected sampler/posterior shapes vary across
        # BoTorch versions; `get_samples` normalizes to a tensor where the
        # first dimension is the MC sample dimension when applicable.
        samples = self.get_samples(X)

        # Apply objective. Typical shape: (num_samples, batch_shape..., q)
        obj = self.objective(samples=samples, X=X)

        # Ensure obj is at least 2D: (num_samples, batch_shape) or (batch_shape,)
        obj_t = obj

        # If we have an MC sample axis (first dim), average over it first.
        if obj_t.dim() >= 2:
            # Assume sample dim is 0 if present
            # Compute improvement per-sample: (num_samples, batch_shape..., q)
            improvement = (self.best_f - obj_t).clamp_min(0)

            # If last dim is q, average across q to get per-sample values
            if improvement.dim() >= 2:
                # average over q (last dim)
                per_sample = improvement.mean(dim=-1)
            else:
                per_sample = improvement

            # Now average over MC samples (dim 0) to get shape (batch_shape...)
            ei = per_sample.mean(dim=0)
        else:
            # No sample axis: obj is (batch_shape,) or scalar
            improvement = (self.best_f - obj_t).clamp_min(0)
            ei = improvement

        # stable log EI (add small constant to avoid log(0))
        return torch.log(ei + 1e-8)

    def get_samples(self, X):
        """Return MC posterior samples for input `X` with multiple fallbacks.

        Tries several BoTorch sampler and posterior APIs to maintain
        compatibility across versions.
        """
        # Ensure tensor
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=next(self.model.parameters()).dtype)

        posterior = None
        try:
            posterior = self.model.posterior(X)
        except Exception:
            # Some older Model implementations expect 2D inputs
            posterior = self.model.posterior(X.unsqueeze(0))

        # Try calling the sampler directly: sampler(posterior)
        if hasattr(self, 'sampler') and self.sampler is not None:
            # Try multiple callable signatures
            try:
                return self.sampler(posterior)
            except TypeError:
                pass

            # Try passing explicit num_samples / sample_shape
            n = None
            for attr in ('num_samples', '_num_samples'):
                n = getattr(self.sampler, attr, None)
                if n is not None:
                    break
            if n is None:
                sh = getattr(self.sampler, 'sample_shape', None)
                if isinstance(sh, (tuple, list, torch.Size)) and len(sh) > 0:
                    try:
                        n = int(sh[0])
                    except Exception:
                        n = None

            if n is None:
                n = 256

            # Try common sampler call patterns
            try:
                return self.sampler(posterior, num_samples=n)
            except Exception:
                pass
            try:
                return self.sampler(posterior, sample_shape=torch.Size([n]))
            except Exception:
                pass

        # Fallback to posterior.rsample(sample_shape)
        try:
            return posterior.rsample(torch.Size([n]))
        except Exception:
            pass

        raise RuntimeError('Unable to draw MC samples from the model posterior with available sampler APIs')


# ===============================================================
#  Safe GP fit (avoid crashes)
# ===============================================================
def safe_fit_gp(gp, name="gp"):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"[WARN] GP fit failed for {name}, continuing. Error: {e}")


# ===============================================================
#               Baseline qLogEI for COCO constrained problem
# ===============================================================
def run_botorch_qlogei(dim, budget, coco_fun, coco_instance, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------------------------
    # 1. Initial random design
    # ---------------------------------------
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
        y_con = y_con.unsqueeze(1)   # shape (n, n_constraints)

    # ---------------------------------------
    # 2. Fit initial GP models
    # ---------------------------------------
    gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
    safe_fit_gp(gp_obj, "objective")

    gp_cons = []
    for j in range(y_con.shape[1]):
        gpj = SingleTaskGP(
            X_t,
            y_con[:, j:j+1],
            outcome_transform=Standardize(m=1)
        )
        safe_fit_gp(gpj, f"constraint_{j}")
        gp_cons.append(gpj)

    model = ModelListGP(gp_obj, *gp_cons)

    # ---------------------------------------
    # 3. Constrained MC Objective
    # ---------------------------------------
    def obj_from_samples(samples, X=None, **kwargs):
        return samples[..., 0]

    def make_con(j):
        return lambda samples, j=j, X=None, **kwargs: samples[..., j + 1]

    constraints = [make_con(j) for j in range(y_con.shape[1])]

    objective = ConstrainedMCObjective(
        objective=obj_from_samples,
        constraints=constraints,
    )

    # ---------------------------------------
    # 4. BO Loop
    # ---------------------------------------
    eval_count = n_init
    bounds = torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.double)

    while eval_count < budget:

        # best feasible f
        feas = torch.all(y_con <= 0, dim=1)
        if torch.any(feas):
            best_f = y_obj[feas].min()
        else:
            best_f = y_obj.min()

        # acquisition = qLogEI
        acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            objective=objective
        )

        # optimize acq
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )

        candidate = candidate.reshape(1, -1)
        x_new = candidate.detach().cpu().numpy()[0]

        f_new = evaluate_objective(x_new, coco_fun, coco_instance, dim)
        g_new = evaluate_constraints(x_new, coco_fun, coco_instance, dim)

        # add point
        X_t = torch.cat([X_t, candidate], dim=0)
        y_obj = torch.cat([y_obj, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        g_new_t = torch.tensor([g_new], dtype=torch.double)
        if g_new_t.dim() == 1:
            g_new_t = g_new_t.unsqueeze(0)
        y_con = torch.cat([y_con, g_new_t], dim=0)

        # refit all GPs
        gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
        safe_fit_gp(gp_obj, "objective")

        gp_cons = []
        for j in range(y_con.shape[1]):
            gpj = SingleTaskGP(
                X_t,
                y_con[:, j:j+1],
                outcome_transform=Standardize(m=1)
            )
            safe_fit_gp(gpj, f"constraint_{j}")
            gp_cons.append(gpj)

        model = ModelListGP(gp_obj, *gp_cons)
        eval_count += 1

    return y_obj, y_con, X_t.numpy()
