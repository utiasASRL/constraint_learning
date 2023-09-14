import cvxpy as cp
import numpy as np
import matplotlib.pylab as plt

from lifters.state_lifter import StateLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo_lifter import StereoLifter

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
