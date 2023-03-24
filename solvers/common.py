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


def find_local_minimum(lifter, a, y, delta=1e-3, verbose=False):
    local_solutions = []
    costs = []

    inits = [lifter.get_vec_around_gt(delta=0)]  # initialize at gt
    inits += [lifter.get_vec_around_gt(delta=delta) for i in range(10)]  # around gt
    for t_init in inits:
        t_local, msg, cost_solver = lifter.local_solver(
            a=a, y=y, t_init=t_init, W=lifter.W, verbose=verbose
        )
        # print(msg)
        if t_local is not None:
            cost_lifter = lifter.get_cost(a=a, y=y, t=t_local, W=lifter.W)
            costs.append(cost_lifter)
            local_solutions.append(t_local)
    local_solutions = np.array(local_solutions)

    if len(costs):
        min_local_ind = np.argmin(costs)
        min_local_cost = costs[min_local_ind]
        best_local_solution = local_solutions[min_local_ind]
        return best_local_solution, min_local_cost
    return None, None
