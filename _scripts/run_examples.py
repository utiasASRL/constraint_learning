import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp

from decomposition.sim_experiments import ADJUST, TOL_SDP, USE_FUSION, USE_PRIMAL
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

USE_METHODS = ["local"]

RESULTS_WRITE = "_results"

if __name__ == "__main__":
    appendix = "exampleRO"
    # appendix = "exampleMW"
    seed = 0
    n_params = 4
    use_learning = True

    np.random.seed(0)
    if appendix == "exampleRO":
        new_lifter = RangeOnlyLocLifter(
            n_landmarks=8, n_positions=n_params, reg=Reg.CONSTANT_VELOCITY, d=2
        )
    elif appendix == "exampleMW":
        new_lifter = MatWeightLocLifter(n_landmarks=8, n_poses=n_params)
    fname = f"{RESULTS_WRITE}/{new_lifter}_{appendix}.pkl"
    add_redundant_constr = True if isinstance(new_lifter, MatWeightLocLifter) else False

    # ====== plot matrices =====
    np.random.seed(seed)
    new_lifter.generate_random_setup()
    new_lifter.simulate_y(noise=new_lifter.NOISE, sparsity=1.0)
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=False, output_poly=True)

    var_dict_plot = new_lifter.get_clique_vars_ij(*range(n_params))

    fig, axs = matshow_list(Q.get_matrix(var_dict_plot).toarray() != 0, log=True)
    savefig(fig, f"{RESULTS_WRITE}/Q_{appendix}.png")

    A_known = [new_lifter.get_A0(output_poly=True)]
    A_known += new_lifter.get_A_known(add_redundant=False, output_poly=True)
    mask = sum(A.get_matrix(var_dict_plot).toarray() != 0 for A in A_known)
    fig, axs = matshow_list(mask > 0, log=True)
    savefig(fig, f"{RESULTS_WRITE}/A_{appendix}.png")

    if use_learning:
        A_red = new_lifter.get_A_learned_simple(A_known=A_known, output_poly=True)
    else:
        A_red = new_lifter.get_A_known(add_redundant=True, output_poly=True)[
            len(A_known) :
        ]

    if len(A_red):
        mask = sum(A.get_matrix(var_dict_plot).toarray() != 0 for A in A_red)
        fig, axs = matshow_list(mask > 0, log=True)
        savefig(fig, f"{RESULTS_WRITE}/Ar_{appendix}.png")

    # ====== plot estimates =====
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=True)
    Constraints = [(new_lifter.get_A0(), 1.0)]
    Constraints += [(A, 0.0) for A in new_lifter.get_A_known()]

    np.random.seed(seed)
    new_lifter.generate_random_setup()
    new_lifter.simulate_y(noise=new_lifter.NOISE, sparsity=1.0)
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=False)
    theta_gt = new_lifter.get_vec_around_gt(delta=0)

    # check that local gives a good estimate...
    print("solving local...")
    theta_est, info, cost = new_lifter.local_solver(
        theta_gt, new_lifter.y_, verbose=True
    )

    X, info = solve_sdp(
        Q,
        Constraints,
        adjust=ADJUST,
        primal=USE_PRIMAL,
        use_fusion=USE_FUSION,
        tol=TOL_SDP,
        verbose=True,
    )
    x_sdp, info = rank_project(X, p=1)

    try:
        fig, ax = new_lifter.prob.plot()
        estimates = {}
        estimates["local"] = theta_est[:, : new_lifter.d]

        theta_sdp = x_sdp[1 : 1 + new_lifter.k * new_lifter.n_positions].reshape(
            -1, new_lifter.k
        )
        estimates["sdp"] = theta_sdp[:, : new_lifter.d]
        new_lifter.prob.plot_estimates(
            points_list=estimates.values(), labels=estimates.keys(), ax=ax
        )
    except Exception as e:
        print("Error plotting:", e)
    print("done")
