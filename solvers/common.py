import cvxpy as cp
import numpy as np

SOLVER = "MOSEK"
solver_options = {
    "CVXOPT": {
        "verbose": False,
        "refinement": 1,
        "kktsolver": "qr",  # important so that we can solve with redundant constraints
        "abstol": 1e-7,  # will be changed according to primal
        "reltol": 1e-6,  # will be changed according to primal
        "feastol": 1e-9,
    },
    "MOSEK": {},
}


def solve_dual(Q, A_list, tol=1e-6, solver=SOLVER, verbose=True):
    options = solver_options[solver]
    if "abstol" in options:
        options["abstol"] = tol
    if "reltol" in options:
        options["reltol"] = tol

    if verbose:
        print("running with solver options", solver_options[solver])
    rho = cp.Variable()

    import scipy.sparse as sp

    A_0 = sp.csr_array(([1.0], ([0], [0])), shape=Q.shape)

    H = Q
    # cp.Parameter(Q.shape)
    # H.value = Q
    H += rho * A_0

    cp_variables = {}
    for i, Ai in enumerate(A_list):
        cp_variables[f"l{i}"] = cp.Variable()
        H += cp_variables[f"l{i}"] * Ai

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


def find_local_minimum(lifter, y, delta=1e-3, verbose=False, n_inits=10):
    local_solutions = []
    costs = []

    inits = [lifter.get_vec_around_gt(delta=0)]  # initialize at gt
    inits += [
        lifter.get_vec_around_gt(delta=delta) for i in range(n_inits - 1)
    ]  # around gt
    for t_init in inits:
        t_local, msg, cost_solver = lifter.local_solver(t_init, y=y, verbose=verbose)
        # print(msg)
        if t_local is not None:
            # cost_lifter = lifter.get_cost(t_local, y=y)
            costs.append(cost_solver)
            local_solutions.append(t_local)
    local_solutions = np.array(local_solutions)

    if len(costs):
        min_local_ind = np.argmin(costs)
        min_local_cost = costs[min_local_ind]
        best_local_solution = local_solutions[min_local_ind]
        return best_local_solution, min_local_cost
    return None, None
