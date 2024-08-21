from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from cert_tools.admm_solvers import solve_alternating
from cert_tools.sparse_solvers import solve_oneshot

from decomposition.generate_cliques import create_clique_list_loc
from decomposition.sim_experiments import extract_solution
from lifters.matweight_lifter import MatWeightLocLifter
from utils.plotting_tools import savefig

if __name__ == "__main__":
    noise = 1.0
    sparsity = 1.0
    new_lifter = MatWeightLocLifter(n_landmarks=8, n_poses=100)

    np.random.seed(0)
    new_lifter.generate_random_setup()
    new_lifter.simulate_y(noise=noise, sparsity=sparsity)

    theta_gt = new_lifter.get_vec_around_gt(delta=0)

    theta_est, info, cost = new_lifter.local_solver(
        theta_gt, new_lifter.y_, verbose=True
    )

    clique_list = create_clique_list_loc(
        new_lifter,
        use_known=True,
        use_autotemplate=False,
        add_redundant=True,
    )

    X_list_dSDP, info_dSDP = solve_oneshot(
        clique_list,
        use_primal=True,
        use_fusion=True,
        verbose=True,
        tol=1e-10,
    )
    x_dSDP, evr_mean = extract_solution(new_lifter, X_list_dSDP)
    theta_dSDP = new_lifter.get_theta_from_x(x=x_dSDP)

    X_list_admm, info_admm = solve_alternating(
        deepcopy(clique_list),
        X0=None,
        verbose=True,
        **new_lifter.ADMM_OPTIONS,
    )
    x_admm, evr_mean = extract_solution(new_lifter, X_list_admm)
    theta_admm = new_lifter.get_theta_from_x(x=x_admm)

    fig, axs = plt.subplots(1, 2, sharex=True)
    new_lifter.prob.plot(ax=axs[0])
    axs[0].set_title("gorund truth")

    new_lifter.prob.plot_estimates(
        theta=theta_est, label="local", ax=axs[1], color="C1"
    )
    new_lifter.prob.plot_estimates(
        theta=theta_dSDP, label="dSDP", ax=axs[1], color="C2"
    )
    new_lifter.prob.plot_estimates(
        theta=theta_admm, label="ADMM", ax=axs[1], color="C3"
    )
    axs[1].set_title("estimates")
    axs[1].legend()
    axs[1].axis("equal")
    savefig(fig, f"_results/example_admm_{new_lifter}.png")
    print("done")
