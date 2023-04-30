import cvxpy as cp
import numpy as np

from .common import adjust_Q, adjust_tol, solver_options

VERBOSE = False
SOLVER = "MOSEK"


def solve_lambda(
    Q,
    A_b_list,
    xhat,
    force_first=1,
    adjust=True,
    solver=SOLVER,
    opts=solver_options[SOLVER],
    primal=False,
    verbose=True,
    tol=None,
):
    """Determine lambda with an SDP.
    :param force_first: number of constraints on which we do not put a L1 cost, effectively encouraging the problem to use them. 
    """

    adjust_tol([opts], tol) if tol else None
    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    if primal:
        raise NotImplementedError("primal form not implemented yet")
    else:  # Dual
        """
        max | y |
        s.t. H := Q + sum(Ai * yi for all i) >> 0
             H xhat == 0
        """
        m = len(A_b_list)
        y = cp.Variable(shape=(m,))

        As, b = zip(*A_b_list)
        H = Q_here + cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])

        objective = cp.Minimize(cp.norm1(y[force_first:]))

        constraints = [H >> 0]
        constraints += [H @ xhat == 0]

        cprob = cp.Problem(objective, constraints)
        try:
            cprob.solve(
                solver=solver,
                **opts,
            )
        except:
            lamda = y.value
            X = None
        else:
            lamda = None
            X = constraints[0].dual_value

    # reverse Q adjustment
    if lamda:
        lamda[0] *= scale
        lamda[0] += offset
    return X, y.value
