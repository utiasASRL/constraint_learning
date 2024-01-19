import matplotlib.pylab as plt
import numpy as np
from cert_tools.sdp_solvers import solve_feasibility_sdp

from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.state_lifter import StateLifter

TOL = 1e-10  # can be overwritten by a parameter.


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


# TODO(FD) delete below if not used anymore.
def solve_certificate(
    Q,
    A_b_list,
    xhat,
    adjust=True,
    verbose=False,
    tol=None,
):
    """Solve certificate."""
    print(
        "common.py -- WARNING: use solve_feasibility_sdp instead of solve_certificate! "
    )
    return solve_feasibility_sdp(
        Q, A_b_list, xhat, adjust=adjust, tol=tol, verobose=verbose
    )
