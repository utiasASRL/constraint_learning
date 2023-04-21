import numpy as np

import cvxpy as cp

solver_options = {
    "CVXOPT": {
        "verbose": False,
        "refinement": 1,
        "kktsolver": "qr",  # important so that we can solve with redundant constraints
        "abstol": 1e-7,  # will be changed according to primal
        "reltol": 1e-6,  # will be changed according to primal
        "feastol": 1e-9,
    }
}

msk_opts = {}
msk_opts["sdp_solver"] = cp.MOSEK
msk_opts["mosek_params"] = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
    "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
}


def solve_sdp_cvxpy(
    Q, A_b_list, adjust=(1.0, 0.0), opts=msk_opts, primal=False, verbose=True
):
    """Run CVXPY to solve a semidefinite program.

    Args:
        Q (_type_): Cost matrix
        A_b_list (_type_): constraint tuple, (A,b) such that trace(A @ X) == b
        adjust (tuple, optional): tuple of scale and offset to adjust the cost. Defaults to (1.0,0.0).
        primal (bool, optional): Choose to solve primal or dual. Defaults to False (dual).
        verbose (bool, optional): Print verbose ouptut. Defaults to True.

    Returns:
        _type_: (X, cost_out): solution matrix and output cost.
    """
    if primal:
        """
        min < Q, X >
        s.t.  trace(Ai @ X) == bi, for all i.
        """
        X = cp.Variable(Q.shape, symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A @ X) == b for A, b in A_b_list]
        cprob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints)
        cprob.solve(
            solver=opts["sdp_solver"],
            save_file="solve_cvxpy_primal.ptf",
            mosek_params=opts["mosek_params"],
            verbose=verbose,
        )
        # Get cost by reversing scaling
        cost_out = cprob.value * adjust[0] + adjust[1]
        X = X.value
        # Get Dual Variables
        H = constraints[0].dual_value
        yvals = [c.dual_value for c in constraints[1:]]
    else:  # Dual
        """
        max < y, b >
        s.t. sum(Ai * yi for all i) << Q
        """
        m = len(A_b_list)
        y = cp.Variable(shape=(m,))
        As, b = zip(*A_b_list)
        b = np.concatenate([np.atleast_1d(bi) for bi in b])
        objective = cp.Maximize(b @ y)
        LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])
        constraint = LHS << Q
        cprob = cp.Problem(objective, [constraint])
        cprob.solve(
            solver=opts["sdp_solver"],
            save_file="solve_cvxpy_dual.ptf",
            mosek_params=opts["mosek_params"],
            verbose=verbose,
        )
        cost_out = cprob.value * adjust[0] + adjust[1]
        X = constraint.dual_value
        # Dual variables
        H = Q - LHS.value
        yvals = [x.value for x in y]
    # Get information
    info = {"H": H, "yvals": yvals, "cost": cost_out}
    return X, info


def solve_dual(Q, A_list, tol=1e-6, solver="CVXOPT", verbose=True):
    solver_options[solver]["abstol"] = tol
    solver_options[solver]["reltol"] = tol
    if verbose:
        print("running with solver options", solver_options[solver])
    rho = cp.Variable()

    A_0 = np.zeros(Q.shape)
    A_0[0, 0] = 1.0

    H = cp.Parameter(Q.shape)

    H.value = Q
    H += A_0 * rho

    cp_variables = {}
    for i, Ai in enumerate(A_list):
        cp_variables[f"l{i}"] = cp.Variable()
        H += Ai * cp_variables[f"l{i}"]

    constraints = [H >> 0]

    prob = cp.Problem(cp.Maximize(-rho), constraints)
    try:
        prob.solve(solver=solver, **solver_options[solver])
        if verbose:
            print("solution:", prob.status)
        return -rho.value, H.value, prob.status
    except Exception as e:
        if verbose:
            print("solution:", prob.status)
        return None, None, prob.status


def find_local_minimum(lifter, y, delta=1e-3, verbose=False):
    local_solutions = []
    costs = []

    inits = [lifter.get_vec_around_gt(delta=0)]  # initialize at gt
    inits += [lifter.get_vec_around_gt(delta=delta) for i in range(10)]  # around gt
    for t_init in inits:
        t_local, msg, cost_solver = lifter.local_solver(t_init, y=y, verbose=verbose)
        # print(msg)
        if t_local is not None:
            cost_lifter = lifter.get_cost(t_local, y=y)
            costs.append(cost_lifter)
            local_solutions.append(t_local)
    local_solutions = np.array(local_solutions)

    if len(costs):
        min_local_ind = np.argmin(costs)
        min_local_cost = costs[min_local_ind]
        best_local_solution = local_solutions[min_local_ind]
        return best_local_solution, min_local_cost
    return None, None
