import cvxpy as cp
import numpy as np
import matplotlib.pylab as plt

from lifters.state_lifter import StateLifter
from lifters.range_only_lifters import RangeOnlyLocLifter

from cert_tools import solve_sdp_mosek
from cert_tools import solve_feasibility_sdp

TOL = 1e-10  # can be overwritten by a parameter.

# Reference for MOSEK parameters explanations:
# https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list
VERBOSE = False
solver_options = {
    "verbose": VERBOSE,
    "mosek_params": {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOL,  # was 1e-8
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOL,  # was 1e-8
        # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7,
        # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,  # this made the problem infeasible sometimes
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOL,
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
        "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
    },
}


def adjust_Q(Q, offset=True, scale=True, plot=False):
    from copy import deepcopy

    ii, jj = (Q == Q.max()).nonzero()
    if (ii[0], jj[0]) != (0, 0) or (len(ii) > 1):
        pass
        # print("Warning: largest element of Q is not unique or not in top-left. Check ordering?")

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
        fig, axs = plt.subplots(1, 2)
        axs[0].matshow(np.log10(np.abs(Q.toarray())))
        axs[1].matshow(np.log10(np.abs(Q_mat.toarray())))
        plt.show()
    return Q_mat, Q_scale, Q_offset


def adjust_tol(options, tol):
    options["mosek_params"].update(
        {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol,
        }
    )


def solve_sdp_cvxpy(
    Q,
    A_b_list,
    B_list=[],
    adjust=True,
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
    opts = solver_options
    opts["verbose"] = verbose

    if tol:
        adjust_tol(opts, tol)

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)
    if primal:
        """
        min < Q, X >
        s.t.  trace(Ai @ X) == bi, for all i.
        """
        X = cp.Variable(Q.shape, symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A @ X) == b for A, b in A_b_list]
        constraints += [cp.trace(B @ X) <= 0 for B in B_list]
        cprob = cp.Problem(cp.Minimize(cp.trace(Q_here @ X)), constraints)
        try:
            cprob.solve(
                solver="MOSEK",
                **opts,
            )
        except Exception as e:
            print(e)
            cost = None
            X = None
            H = None
            yvals = None
            msg = "infeasible / unknown"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = X.value
                H = constraints[0].dual_value
                yvals = [c.dual_value for c in constraints[1:]]
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"
    else:  # Dual
        """
        max < y, b >
        s.t. sum(Ai * yi for all i) << Q
        """
        m = len(A_b_list)
        y = cp.Variable(shape=(m,))

        k = len(B_list)
        if k > 0:
            u = cp.Variable(shape=(k,))

        As, b = zip(*A_b_list)
        b = np.concatenate([np.atleast_1d(bi) for bi in b])
        objective = cp.Maximize(b @ y)

        # We want the lagrangian to be H := Q - sum l_i * A_i + sum u_i * B_i.
        # With this choice, l_0 will be negative
        LHS = cp.sum(
            [y[i] * Ai for (i, Ai) in enumerate(As)]
            + [-u[i] * Bi for (i, Bi) in enumerate(B_list)]
        )
        # this does not include symmetry of Q!!
        constraints = [LHS << Q_here]
        constraints += [LHS == LHS.T]
        if k > 0:
            constraints.append(u >= 0)

        cprob = cp.Problem(objective, constraints)
        try:
            cprob.solve(
                solver="MOSEK",
                **opts,
            )
        except Exception as e:
            print(e)
            cost = None
            X = None
            H = None
            yvals = None
            msg = "infeasible / unknown"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = constraints[0].dual_value
                H = Q_here - LHS.value
                yvals = [x.value for x in y]

                # sanity check for inequality constraints.
                # we want them to be inactive!!!
                if len(B_list):
                    mu = np.array([ui.value for ui in u])
                    i_nnz = np.where(mu > 1e-10)[0]
                    if len(i_nnz):
                        for i in i_nnz:
                            print(
                                f"Warning: is constraint {i} active? (mu={mu[i]:.4e}):"
                            )
                            print(np.trace(B_list[i] @ X))
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"

    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset

        H = Q_here - cp.sum(
            [yvals[i] * Ai for (i, Ai) in enumerate(As)]
            + [-u[i] * Bi for (i, Bi) in enumerate(B_list)]
        )
        yvals[0] = yvals[0] * scale + offset
        # H *= scale
        # H[0, 0] += offset

    info = {"H": H, "yvals": yvals, "cost": cost, "msg": msg}
    return X, info


def solve_sdp_cvxpy_new(
    Q,
    A_b_list,
    B_list=[],
    adjust=True,
    primal=False,
    verbose=True,
    tol=None,
):
    # TODO: below doesn't currently give the same results as when running the above function.
    # Need to figure out why and fix this.
    assert primal is False, "Option primal not supported anymore, it is less efficient."
    assert len(B_list) == 0, "Inequality constraints not supported anymore."

    opts = solver_options
    opts["verbose"] = verbose
    if tol:
        adjust_tol(opts, tol)

    return solve_sdp_mosek(Q, A_b_list, adjust=adjust, verbose=verbose)


def find_local_minimum(
    lifter: StateLifter, y, delta=1.0, verbose=False, n_inits=10, plot=False
):
    local_solutions = []
    costs = []
    max_res = []
    cond_Hess = []
    failed = []

    inits = [lifter.get_vec_around_gt(delta=0)]  # initialize at gt
    inits += [
        lifter.get_vec_around_gt(delta=delta) for i in range(n_inits - 1)
    ]  # around gt
    info = {"success": False}
    for i, t_init in enumerate(inits):
        try:
            t_local, info_here, cost_solver = lifter.local_solver(
                t_init, y=y, verbose=verbose
            )
        except NotImplementedError:
            print("Warning: local solver not implemented.")
            return None, None, info

        if t_local is None:
            cost_solver = np.nan
            t_local = np.nan
            failed.append(i)

        costs.append(cost_solver)
        local_solutions.append(t_local)
        max_res.append(info_here.get("max res", np.nan))
        cond_Hess.append(info_here.get("cond Hess", np.nan))

    if len(costs):
        info["success"] = True
        costs = np.round(costs, 8)
        global_cost = np.nanmin(costs)

        local_costs = np.unique(costs[~np.isnan(costs) & (costs != global_cost)])

        global_inds = np.where(costs == global_cost)[0]
        global_solution = local_solutions[global_inds[0]]
        local_inds = np.where(np.isin(costs, local_costs))[0]

        info["n global"] = len(global_inds)
        info["n local"] = len(costs) - info["n global"] - len(failed)
        info["n fail"] = len(failed)
        info["max res"] = max_res[global_inds[0]]
        info["cond Hess"] = cond_Hess[global_inds[0]]

        for local_cost in local_costs:
            local_ind = np.where(costs == local_cost)[0][0]
            info[f"local solution {i}"] = local_solutions[local_ind]
            info[f"local cost {i}"] = local_cost

        # if (info["n local"] or info["n fail"]) and fname_root != "":
        if plot:
            from utils.plotting_tools import plot_frame

            fig, ax = plt.subplots()

            ax.scatter(
                *lifter.all_landmarks[:, :2].T, color=f"k", marker="+", alpha=0.0
            )
            ax.scatter(*lifter.landmarks[:, :2].T, color=f"k", marker="+")

            # plot ground truth, global and local costs only once.
            plot_frame(
                lifter,
                ax,
                theta=lifter.theta,
                color="k",
                marker="*",
                ls="-",
                alpha=1.0,
                s=100,
                label=None,
            )
            plot_frame(
                lifter,
                ax,
                xtheta=global_solution,
                color="g",
                marker="*",
                label=f"candidate, q={global_cost:.2e}",
            )
            for local_cost in local_costs:
                local_ind = np.where(costs == local_cost)[0][0]
                xtheta = local_solutions[local_ind]
                plot_frame(
                    lifter,
                    ax,
                    xtheta=xtheta,
                    color="r",
                    marker="*",
                    label=f"candidate, q={local_cost:.2e}",
                )

            # plot all solutions that converged to those (for RO only, for stereo it's too crowded)
            if isinstance(lifter, RangeOnlyLocLifter):
                for i in global_inds[1:]:  # first one corresponds to ground truth
                    plot_frame(lifter, ax, xtheta=inits[i], color="g", marker=".")
                for i in local_inds:
                    plot_frame(lifter, ax, xtheta=inits[i], color="r", marker=".")

            ax.axis("equal")
            fig.set_size_inches(5, 5)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(framealpha=1.0)
        return global_solution, global_cost, info

    return None, None, info


def solve_certificate(
    Q,
    A_b_list,
    xhat,
    adjust=True,
    verbose=False,
    tol=None,
):
    """Solve certificate."""
    opts = solver_options
    opts["verbose"] = verbose
    if tol:
        adjust_tol(opts, tol)
    return solve_feasibility_sdp(Q, A_b_list, xhat, adjust=adjust, sdp_opts=opts)
