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
    overwrite = True
    n_threads_list = [2]

    appendix = "exampleRO"
    seed = 0

    np.random.seed(0)
    # new_lifter = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    new_lifter = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )

    fname = f"{RESULTS_WRITE}/{new_lifter}_{appendix}.pkl"
    add_redundant_constr = True if isinstance(new_lifter, MatWeightLocLifter) else False

    # ====== plot matrices =====
    np.random.seed(seed)
    new_lifter.generate_random_setup()
    new_lifter.simulate_y(noise=new_lifter.NOISE, sparsity=1.0)
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=False)
    fig, axs = matshow_list(Q.toarray() != 0, log=True)
    axs[0, 0].set_title("Q")

    new_lifter.prob.W = np.zeros(new_lifter.prob.W.shape)
    R = new_lifter.get_Q_from_y(new_lifter.y_, save=False)
    fig, axs = matshow_list(R.toarray() != 0, log=True)
    axs[0, 0].set_title("R")

    # ====== plot estimates =====
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
    fig, ax = new_lifter.prob.plot()
    estimates = {"local": theta_est[:, : new_lifter.d]}

    Constraints = [(new_lifter.get_A0(), 1.0)]
    Constraints += [(A, 0.0) for A in new_lifter.get_A_known()]
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
    theta_sdp = x_sdp[1 : 1 + new_lifter.k * new_lifter.n_positions].reshape(
        -1, new_lifter.k
    )
    estimates["sdp"] = theta_sdp[:, : new_lifter.d]
    new_lifter.prob.plot_estimates(
        points_list=estimates.values(), labels=estimates.keys(), ax=ax
    )
    print("done")
