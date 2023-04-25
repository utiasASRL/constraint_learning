import cvxpy as cp
import numpy as np

from .common import adjust_Q, adjust_tol, solver_options

VERBOSE = False
SOLVER = "MOSEK"


def solve_lambda(
    Q,
    A_b_list,
    xhat,
    adjust=True,
    solver=SOLVER,
    opts=solver_options[SOLVER],
    primal=False,
    verbose=True,
    tol=None,
):
    """Determine lambda with an SDP.

    Args:
        Q (_type_): Cost matrix
        A_b_list (_type_): constraint tuple, (A,b) such that trace(A @ X) == b
        adjust (bool, optional): Choose to normalize Q matrix.
        primal (bool, optional): Choose to solve primal or dual. Defaults to False (dual).
        verbose (bool, optional): Print verbose ouptut. Defaults to True.

    Returns:
        _type_: (X, lamda_out): solution matrix and lambda vector.
    """

    adjust_tol([opts], tol) if tol else None
    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    if primal:
        """
        min < Q, X >
        s.t.  trace(Ai @ X) == bi, for all i.
        """
        raise NotImplementedError("primal form not implemented yet")
        X = cp.Variable(Q.shape, symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A @ X) == b for A, b in A_b_list]
        cprob = cp.Problem(cp.Minimize(cp.trace(Q_here @ X)), constraints)
        cprob.solve(
            solver=solver,
            save_file="solve_cvxpy_primal.ptf",
            **opts,
            verbose=verbose,
        )
        cost = cprob.value
        X = X.value
    else:  # Dual
        """
        max < y, b >
        s.t. sum(Ai * yi for all i) << Q
        """
        m = len(A_b_list)
        y = cp.Variable(shape=(m,))
        As, b = zip(*A_b_list)
        b = np.concatenate([np.atleast_1d(bi) for bi in b])

        H = Q_here + cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])

        objective = cp.Minimize(cp.norm1(y[1:]))

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
