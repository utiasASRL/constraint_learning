import numpy as np
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp

from decomposition.sim_experiments import ADJUST, TOL_SDP, USE_FUSION, USE_PRIMAL
from lifters.matweight_lifter import (
    MatWeightLifter,
    MatWeightLocLifter,
    MatWeightSLAMLifter,
)
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

USE_METHODS = ["local"]

RESULTS_DIR = "_results"


def plot_matrices(new_lifter: MatWeightSLAMLifter, n_params, use_learning, fname=""):
    # ====== plot matrices =====
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=False, output_poly=True)

    var_dict_plot = new_lifter.get_clique_vars_ij(*range(n_params))

    mask_Q = Q.get_matrix(var_dict_plot).toarray() != 0
    mask_all = mask_Q.astype(int)
    fig, axs = matshow_list(mask_Q, log=True)
    if fname != "":
        savefig(fig, fname + "_Q.png")

    A_known = [new_lifter.get_A0(output_poly=True)]
    A_known += new_lifter.get_A_known(add_redundant=False, output_poly=True)
    mask_known = sum(A.get_matrix(var_dict_plot).toarray() != 0 for A in A_known)
    mask_all += mask_known.astype(int)
    fig, axs = matshow_list(mask_known > 0, log=True)
    if fname != "":
        savefig(fig, fname + "_A.png")

    if use_learning:
        A_red = new_lifter.get_A_learned_simple(A_known=A_known, output_poly=True)
    else:
        A_red = new_lifter.get_A_known(add_redundant=True, output_poly=True)[
            len(A_known) :
        ]

    if len(A_red):
        mask_red = sum(A.get_matrix(var_dict_plot).toarray() != 0 for A in A_red)
        mask_all += mask_red.astype(int)
        fig, axs = matshow_list(mask_red > 0, log=True)
        if fname != "":
            savefig(fig, fname + "_Ar.png")
    A_all = A_known + A_red

    # chordal completion
    from solvers.chordal import investigate_sparsity

    fig, axs = matshow_list(mask_all > 0, log=True)
    mask_chordal = investigate_sparsity(mask_all > 0, ax=axs[0, 0])
    if fname != "":
        savefig(fig, fname + "_chordal.png")

    return [
        (A.get_matrix_sparse(var_dict_plot), b)
        for A, b in zip(A_all, [1] + [0] * (len(A_all) - 1))
    ]


def run_example(results_dir=RESULTS_DIR, appendix="exampleRO", n_landmarks=8):
    assert appendix in ["exampleRO", "exampleMW", "exampleMWslam"]
    seed = 0
    n_params = 4
    use_learning = False

    np.random.seed(seed)
    if appendix == "exampleRO":
        new_lifter = RangeOnlyLocLifter(
            n_landmarks=n_landmarks,
            n_positions=n_params,
            reg=Reg.CONSTANT_VELOCITY,
            d=2,
        )
    elif appendix == "exampleMW":
        new_lifter = MatWeightLocLifter(n_landmarks=n_landmarks, n_poses=n_params)
    elif appendix == "exampleMWslam":
        new_lifter = MatWeightSLAMLifter(n_landmarks=n_landmarks, n_poses=n_params)
    else:
        raise ValueError(appendix)

    np.random.seed(seed)
    new_lifter.generate_random_setup()
    new_lifter.simulate_y(noise=new_lifter.NOISE, sparsity=1.0)
    fname = f"{results_dir}/{appendix}"
    Constraints = plot_matrices(new_lifter, n_params, use_learning, fname=fname)

    # ====== plot estimates =====

    # check that local gives a good estimate...
    theta_gt = new_lifter.get_vec_around_gt(delta=0)
    print("solving local...")
    theta_est, info, cost = new_lifter.local_solver(
        theta_gt, new_lifter.y_, verbose=True
    )

    # solve the SDP
    # Constraints = [(new_lifter.get_A0(), 1.0)]
    # Constraints += [(A, 0.0) for A in new_lifter.get_A_known()]
    Q = new_lifter.get_Q_from_y(new_lifter.y_, save=True, output_poly=False)

    X, info_sdp = solve_sdp(
        Q,
        Constraints,
        adjust=ADJUST,
        primal=USE_PRIMAL,
        use_fusion=USE_FUSION,
        tol=TOL_SDP,
        verbose=False,
    )
    x_sdp, info_rank = rank_project(X, p=1)

    fig, ax = new_lifter.prob.plot()
    if isinstance(new_lifter, MatWeightLifter):
        theta_sdp = new_lifter.get_theta_from_x(x=x_sdp)
        new_lifter.prob.plot_estimates(
            theta=theta_est, label="local", ax=ax, color="C1"
        )
        new_lifter.prob.plot_estimates(theta=theta_sdp, label="sdp", ax=ax, color="C2")
    else:
        estimates = {}
        estimates["local"] = theta_est[:, : new_lifter.d]
        theta_sdp = new_lifter.get_theta_from_x(x=x_sdp)
        estimates["sdp"] = theta_sdp[:, : new_lifter.d]
        new_lifter.prob.plot_estimates(
            points_list=estimates.values(), labels=estimates.keys(), ax=ax
        )
    print("done")


if __name__ == "__main__":
    run_example(appendix="exampleRO")
    run_example(appendix="exampleMW", n_landmarks=3)
    # run_example(appendix="exampleMWslam", n_landmarks=8)
