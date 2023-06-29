import cvxpy as cp
import numpy as np
import scipy.sparse.linalg as spl

VERBOSE = False
SOLVER = "MOSEK"  # first choice
solver_options = {
    None: {},
    "CVXOPT": {
        "verbose": VERBOSE,
        "refinement": 1,
        "kktsolver": "qr",  # important so that we can solve with redundant constraints
        "abstol": 1e-7,  # will be changed according to primal
        "reltol": 1e-6,  # will be changed according to primal
        "feastol": 1e-9,
    },
    "MOSEK": {
        "verbose": VERBOSE,
        "mosek_params": {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,  # was 1e-12
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,  # was 1e-12
            # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,  # this made the problem infeasible sometimes
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,  # was 1e-10
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
            "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
        },
    },
}


def adjust_tol(options, tol):
    for opt in options:
        if "mosek_params" in opt:
            opt["mosek_params"].update(
                {
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
                    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": tol,
                }
            )
        else:
            opt.update(
                {
                    "abstol": tol,
                    "reltol": tol,
                    "feastol": tol,
                }
            )


def adjust_Q(Q, offset=True, scale=True, plot=False):
    from copy import deepcopy

    ii, jj = (Q == Q.max()).nonzero()
    if (ii[0], jj[0]) != (0, 0) or (len(ii) > 1):
        print(
            "Warning: largest element of Q is not unique or not in top-left. Check ordering?"
        )

    Q_mat = deepcopy(Q)
    if offset:
        Q_offset = Q_mat[0, 0]
    else:
        Q_offset = 0
    Q_mat[0, 0] -= Q_offset

    if scale:
        # Q_scale = spl.norm(Q_mat, "fro")
        Q_scale = Q_mat.max()
    else:
        Q_scale = 1.0
    Q_mat /= Q_scale
    if plot:
        import matplotlib.pylab as plt

        fig, axs = plt.subplots(1, 2)
        axs[0].matshow(np.log10(np.abs(Q.toarray())))
        axs[1].matshow(np.log10(np.abs(Q_mat.toarray())))
        plt.show()
    return Q_mat, Q_scale, Q_offset


def solve_sdp_cvxpy(
    Q,
    A_b_list,
    adjust=True,
    solver=SOLVER,
    primal=False,
    verbose=True,
    tol=None,
):
    """Run CVXPY to solve a semidefinite program.

    Args:
        Q (_type_): Cost matrix
        A_b_list (_type_): constraint tuple, (A,b) such that trace(A @ X) == b
        adjust (bool, optional): Choose to normalize Q matrix.
        primal (bool, optional): Choose to solve primal or dual. Defaults to False (dual).
        verbose (bool, optional): Print verbose ouptut. Defaults to True.

    Returns:
        _type_: (X, cost_out): solution matrix and output cost.
    """
    opts = solver_options[solver]

    if tol:
        adjust_tol([opts], tol)

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)
    if primal:
        """
        min < Q, X >
        s.t.  trace(Ai @ X) == bi, for all i.
        """
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
        constraint = LHS << Q_here

        cost = None
        X = None
        H = None
        yvals = None

        cprob = cp.Problem(objective, [constraint])
        try:
            cprob.solve(
                solver=solver,
                **opts,
            )
        except:
            if verbose:
                print(f"Solver {solver} failed! solving again with verbose option.")
                from copy import deepcopy

                o_here = deepcopy(opts)
                o_here["verbose"] = True
                try:
                    cprob.solve(
                        solver=solver,
                        **o_here,
                    )
                except:
                    pass
        else:
            cost = cprob.value
            X = constraint.dual_value
            H = Q_here - LHS.value
            yvals = [x.value for x in y]
    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset
        yvals[0] = yvals[0] * scale + offset
        H *= scale
        H[0, 0] += offset
    info = {"H": H, "yvals": yvals, "cost": cost}
    return X, info


def find_local_minimum(lifter, y, delta=1e-3, verbose=False, n_inits=10, plot=False):
    local_solutions = []
    costs = []

    inits = [lifter.get_vec_around_gt(delta=0)]  # initialize at gt
    inits += [
        lifter.get_vec_around_gt(delta=delta) for i in range(n_inits - 1)
    ]  # around gt
    for i, t_init in enumerate(inits):
        if plot:
            import matplotlib.pylab as plt

            fig, ax = plt.subplots()
            p0, a0 = lifter.get_positions_and_landmarks(t_init)
            ax.scatter(*p0.T, color=f"C{0}", marker="o")
            ax.scatter(*a0.T, color=f"C{0}", marker="x")
        t_local, msg, cost_solver = lifter.local_solver(t_init, y=y, verbose=verbose)

        # print(msg)
        if t_local is not None:
            # cost_lifter = lifter.get_cost(t_local, y=y)
            costs.append(cost_solver)
            local_solutions.append(t_local)

            if plot:
                p0, a0 = lifter.get_positions_and_landmarks(t_local)
                ax.scatter(*p0.T, color=f"C{1}", marker="*")
                ax.scatter(*a0.T, color=f"C{1}", marker="+")
    local_solutions = np.array(local_solutions)

    if len(costs):
        min_local_ind = np.argmin(costs)
        min_local_cost = costs[min_local_ind]
        best_local_solution = local_solutions[min_local_ind]
        return best_local_solution, min_local_cost
    return None, None
