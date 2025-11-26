# Class of objective functions for Teemo
# 
# March 2024
##########
# Imports
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

import numpy as np
import os
import torch

##########
# Objectives functions to test FuRBO with
###
# Ackley function, dimension 2 
class ack():
    
    def __init__(self, dim, negate, **tkwargs):
        
        self.fun = Ackley(dim = dim, negate = negate).to(**tkwargs)
        self.fun.bounds[0, :].fill_(-5)
        self.fun.bounds[1, :].fill_(10)
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        
    def eval_(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return self.fun(unnormalize(x, self.fun.bounds))
    
class keane():
    
    def __init__(self, dim, lower_bound, upper_bound, **tkwargs):
        
        self.dim = dim
        self.lb = lower_bound * torch.ones(dim)
        self.ub = upper_bound * torch.ones(dim)
        
    def eval_(self, x):
        
        x = unnormalize(x, [self.lb, self.ub])
        
        sum_ = torch.sum(torch.cos(x) ** 4)
        prod_ = torch.prod(torch.cos(x) ** 2)
        
        num_ = sum_ - 2 * prod_
        den_ = torch.sqrt((torch.arange(1, self.dim + 1, dtype = x.dtype).dot(x ** 2)))
        
        return torch.abs(num_/den_)
        
class mopta_obj():
    def __init__(self, dim, lower_bound, upper_bound):
        
        self.dim = dim
        self.lb = lower_bound * torch.ones(dim)
        self.ub = upper_bound * torch.ones(dim)
        return
        
    def eval_(self, x, **tkwargs):
        
        # transform to numpy
        x = x.cpu().numpy()
        np.save("mopta_in.npy", x)
        
        # Evaluate
        os.system("wsl python /home/paoloa/mopta_runner.py")
        
        # transform to torch
        y = torch.tensor(np.load("mopta_out.npy"), **tkwargs)
        
        return y[0]
        
        
if __name__ == '__main__':
    
    fcn = mopta_obj()
    
    x = torch.tensor([1., 2., 3., 4., 5., 6.])
    
    y = fcn.eval_(x)


def evaluate_objective(x, coco_fun, coco_instance, dim=None):
    """Evaluate the COCO objective for given function/instance.

    x: 1D numpy array or list (in the problem input space)
    coco_fun: integer function id (e.g., 2 -> f002)
    coco_instance: integer instance index (0-based; mapped to i01..)
    dim: optional dimension (int). If provided, will match p.dimension.
    Returns a scalar float.
    """
    import cocoex
    import numpy as _np

    # Normalize instance numbering: incoming code uses 0-based instances
    inst_id = f"i{coco_instance+1:02d}"
    fun_id = f"f{coco_fun:03d}"

    suite = cocoex.Suite("bbob-constrained", "", "")
    for p in suite:
        parts = p.id.split('_')
        if len(parts) >= 4 and parts[1] == fun_id and parts[2] == inst_id:
            if dim is None or parts[3] == f"d{int(dim):02d}":
                arr = _np.asarray(x, dtype=_np.float64)
                return float(p(arr))

    raise ValueError(f"COCO problem f{coco_fun} i{coco_instance+1} not found in suite")