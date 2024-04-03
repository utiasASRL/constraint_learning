"""
This script was an unsuccessful attempt to use Parameters to speedup the solver (for warm starting).
"""

import time
from copy import deepcopy

import cvxpy as cp
import numpy as np
import pandas as pd
from cert_tools import adjust_Q, adjust_tol, options_cvxpy

from _scripts.run_clique_study import read_saved_learner
from auto_template.learner import ADJUST_Q, PRIMAL, TOL, Learner
from auto_template.sim_experiments import create_newinstance
from lifters.wahba_lifter import WahbaLifter

USE_PARAMETERS = True
VERBOSE = False


def generate_robust_results(lifter):
    saved_learner = read_saved_learner(lifter)

    scale = 1.0
    offset = 0.0

    if TOL:
        adjust_tol(options_cvxpy, TOL)
    options_cvxpy["verbose"] = VERBOSE
    if not USE_PARAMETERS:
        options_cvxpy["ignore_dpp"] = True
    else:
        options_cvxpy["ignore_dpp"] = False

    learner = Learner(lifter=lifter, n_inits=1)
    learner.find_local_solution()

    results = []
    for n_params in np.arange(10, 31, step=5):
        print(f"params: {n_params}")
        new_lifter = create_newinstance(lifter, n_params)
        for clique_size in [5, 6, 7]:
            print(f" clique_size: {clique_size}")
            # regenerate A_list.
            new_lifter.ALL_PAIRS = False
            new_lifter.CLIQUE_SIZE = clique_size
            learner.lifter = new_lifter

            learner.templates = saved_learner.templates
            learner.apply_templates()

            A_b_list = learner.get_A_b_list()
            As, b = zip(*A_b_list)

            if USE_PARAMETERS:
                mask, _ = new_lifter.get_Q()
                mask = abs(mask) > 0
                mask += sum([abs(A) > 0 for A in As])
                ii, jj = mask.nonzero()
                I = ii[jj >= ii]
                J = jj[jj >= ii]
                iiz, jjz = (mask == 0).nonzero()
                Iz = iiz[jjz >= iiz]
                Jz = jjz[jjz >= iiz]

                Q = cp.Parameter(len(I))
                As_vec = np.array([np.array(A[I, J]).flatten() for A in As])

            else:
                Q = cp.Parameter(As[0].shape)

            if PRIMAL:
                """
                min < Q, X >
                s.t.  trace(Ai @ X) == bi, for all i.
                """
                X = cp.Variable(As[0].shape, symmetric=True)
                constraints = [X >> 0]
                constraints += [cp.trace(A @ X) == b for A, b in A_b_list]
                if USE_PARAMETERS:
                    objective = cp.Minimize(X[I, J] @ Q)
                else:
                    objective = cp.Minimize(cp.trace(Q @ X))

            else:  # Dual
                m = len(A_b_list)
                y = cp.Variable(shape=(m))

                b = np.concatenate([np.atleast_1d(bi) for bi in b])
                objective = cp.Maximize(b @ y)

                # We want the lagrangian to be H := Q - sum l_i * A_i + sum u_i * B_i.
                # With this choice, l_0 will be negative
                if USE_PARAMETERS:
                    H = cp.Variable(As[0].shape, PSD=True)
                    constraints = [H[I, J] == Q - y @ As_vec]
                    constraints += [H[Iz, Jz] == 0]
                else:
                    LHS = cp.sum([y[i] * As[i] for i in range(len(As))])
                    constraints = [LHS << Q]

            cprob = cp.Problem(objective, constraints)

            for n_inliers in np.arange(3, min(8, n_params)):
                print(f"   n_inliers: {n_inliers}")
                # regenerate Q, qcqp_cost:
                n_outliers = new_lifter.n_landmarks - n_inliers
                learner.lifter = create_newinstance(new_lifter, n_params, n_outliers)
                learner.find_local_solution()

                Q_val = deepcopy(learner.solver_vars["Q"])
                if ADJUST_Q:
                    Q_val, scale, offset = adjust_Q(Q_val)

                if USE_PARAMETERS:
                    values = np.array(Q_val[I, J]).flatten()
                    # values[I == J] *= factor
                    Q.value = values
                else:
                    Q.value = Q_val

                try:
                    t1 = time.time()
                    cprob.solve(
                        solver="MOSEK",
                        **options_cvxpy,
                    )
                    time_solve = time.time() - t1
                    print(f"solving took: {time_solve:.2f}s")
                except cp.SolverError as e:
                    print("Solver failed with error:", e)
                else:
                    dual_cost = cprob.value * scale + offset
                primal_cost = learner.solver_vars["qcqp_cost"]
                print(f"d={dual_cost:4.4f}, p={primal_cost:4.4f}")

                results.append(
                    {
                        "dual_cost": dual_cost,
                        "primal_cost": primal_cost,
                        "n_inliers": n_inliers,
                        "n_outliers": n_outliers,
                        "n_landmarks": n_params,
                        "clique_size": clique_size,
                        "time_solve": time_solve,
                        "primal": PRIMAL,
                    }
                )
            df = pd.DataFrame(results)
            fname = f"_results/{lifter}_cliques.pkl"
            df.to_pickle(fname)
            print(f"saved intermediate as {fname}")
    df = pd.DataFrame(results)
    df.to_pickle(f"_results/{lifter}_cliques.pkl")
    return df


if __name__ == "__main__":
    lifter = WahbaLifter(n_landmarks=5, d=3, robust=True, level="xwT", n_outliers=1)
    df = generate_robust_results(lifter)
