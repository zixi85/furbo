import os
import numpy as np
import torch

def aggregate_problem(problem_dir, n_iteration=None):
    """
    Read per-repetition files (.torch or obj/cons npy pairs) from `problem_dir`,
    compute aggregated arrays and save:
      - '01_Y_mono.npy' : monotonic best-per-iteration curves (per repetition)
      - '02_Y_best.npy' : raw per-repetition objective per iteration
      - '02_C_best.npy' : raw per-repetition max-constraint per iteration

    Also returns the computed arrays.
    """
    # find torch files first
    files = sorted(os.listdir(problem_dir))
    torch_files = [f for f in files if f.endswith('.torch')]

    states = []

    if len(torch_files) > 0:
        for tf in torch_files:
            path = os.path.join(problem_dir, tf)
            try:
                st = torch.load(path, map_location=torch.device('cpu'))
            except Exception:
                continue
            # Expect st to be a history-like list; if it's a raw tensor, wrap
            if isinstance(st, dict) or torch.is_tensor(st):
                st = [st]
            states.append(st)
    else:
        # fallback to numpy files saved by baseline runner
        # look for matching pairs obj_rep{rep}.npy and cons_rep{rep}.npy
        obj_files = sorted([f for f in files if f.startswith('obj_rep') and f.endswith('.npy')])
        cons_files = sorted([f for f in files if f.startswith('cons_rep') and f.endswith('.npy')])

        # pair by rep index
        for objf in obj_files:
            rep_idx = objf.replace('obj_rep', '').replace('.npy', '')
            consf = f'cons_rep{rep_idx}.npy'
            obj_path = os.path.join(problem_dir, objf)
            cons_path = os.path.join(problem_dir, consf)
            if not os.path.exists(cons_path):
                # try X-only case
                continue
            Y = np.load(obj_path)
            C = np.load(cons_path)
            # ensure shapes: (n_iter,) or (n_iter,1)
            try:
                Y_t = torch.as_tensor(Y)
            except Exception:
                Y_t = torch.tensor(Y)
            try:
                C_t = torch.as_tensor(C)
            except Exception:
                C_t = torch.tensor(C)

            # create a minimal history list with one event containing entire sequences
            event = {'batch': {'Y': Y_t, 'C': C_t}}
            states.append([event])

    if len(states) == 0:
        raise RuntimeError(f'No repetition files found in {problem_dir}')

    # determine n_iteration if not provided
    if n_iteration is None:
        # find minimal length among reps
        lengths = []
        for state in states:
            # concatenate Y across events
            total = 0
            for ev in state:
                y = ev['batch']['Y']
                if torch.is_tensor(y):
                    total += y.numel()
                else:
                    total += np.asarray(y).size
            lengths.append(total)
        n_iteration = min(lengths)

    # Build Y_batch and C_batch as in other scripts
    Y_batch = []
    C_batch = []
    for state in states:
        # concatenate Y values
        Ys = []
        Cs = []
        for ev in state:
            y = ev['batch']['Y']
            c = ev['batch']['C']
            # convert to numpy
            if torch.is_tensor(y):
                y_np = y.detach().cpu().numpy().reshape(-1)
            else:
                y_np = np.asarray(y).reshape(-1)

            if torch.is_tensor(c):
                c_np = c.detach().cpu().numpy()
            else:
                c_np = np.asarray(c)

            # if constraints have shape (n, m) we take max per evaluation
            if c_np.ndim == 2:
                c_max = np.max(c_np, axis=1)
            else:
                c_max = c_np.reshape(-1)

            Ys.append(y_np)
            Cs.append(c_max)

        Y_concat = np.concatenate(Ys, axis=0)[:n_iteration]
        C_concat = np.concatenate(Cs, axis=0)[:n_iteration]

        # match FuRBO postprocessing sign convention: negate if necessary
        # We keep values as-is; other code may expect negated values depending on algorithm.
        Y_batch.append(Y_concat)
        C_batch.append(C_concat)

    Y_best = np.array(Y_batch)
    C_best = np.array(C_batch)

    # Create a monotonic curve similar to original scripts
    Y_f = np.copy(Y_best)
    C_f = np.copy(C_best)
    # For locations where constraint violated (C>0), set Y to max so they are not considered
    mask = (C_f > 0)
    if mask.any():
        # set to very large (worst) value per row
        max_per_row = np.max(Y_f, axis=1)
        for i in range(Y_f.shape[0]):
            Y_f[i, mask[i]] = max_per_row[i]

    Y_f_monotonic = []
    for YY in Y_f:
        y_mono = []
        for yy in YY:
            if len(y_mono) == 0:
                y_mono = [yy]
            else:
                # ensure monotonic non-increasing (best-so-far)
                if yy < y_mono[-1]:
                    y_mono.append(yy)
                else:
                    y_mono.append(y_mono[-1])

        Y_f_monotonic.append(y_mono)

    Y_f_monotonic = np.array(Y_f_monotonic)

    # Save files
    np.save(os.path.join(problem_dir, '01_Y_mono.npy'), Y_f_monotonic)
    np.save(os.path.join(problem_dir, '02_Y_best.npy'), Y_best)
    np.save(os.path.join(problem_dir, '02_C_best.npy'), C_best)

    return Y_f_monotonic, Y_best, C_best
