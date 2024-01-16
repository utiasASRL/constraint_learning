import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.sparse_solvers import solve_oneshot

from _scripts.generate_cliques import create_clique_list_slam
from _scripts.run_clique_study import read_saved_learner
from auto_template.sim_experiments import create_newinstance
from lifters.matweight_lifter import MatWeightLocLifter

NOISE_SEED = 0

USE_KNOWN = True
USE_AUTOTEMPLATE = False

ADJUST = True
USE_PRIMAL = False
USE_FUSION = False
VERBOSE = False


def extract_solution(lifter: MatWeightLocLifter, X_list):
    x_dim = lifter.landmark_size()
    x, info = rank_project(X_list[0], p=1)
    evr_mean = [info["EVR"]]
    x_all = [x[1 : 1 + x_dim]]
    for X in X_list:
        x, info = rank_project(X, p=1)
        x_all.append(x[1 + x_dim : 1 + 2 * x_dim])
        evr_mean.append(info["EVR"])
    return np.vstack(x_all), np.mean(evr_mean)


def generate_results(lifter: MatWeightLocLifter, n_params_list=[10]):
    saved_learner = read_saved_learner(lifter)

    df_data = []
    for n_params in n_params_list:
        print(f"N={n_params}: ", end="")
        time_dict = {"n params": n_params}
        new_lifter = create_newinstance(lifter, n_params=n_params)

        np.random.seed(NOISE_SEED)
        Q, y = new_lifter.get_Q()

        theta_gt = new_lifter.get_vec_around_gt(delta=0)
        t1 = time.time()
        theta_est, info, cost = new_lifter.local_solver(theta_gt, lifter.y_)
        time_dict["t local"] = time.time() - t1
        time_dict["cost local"] = cost
        x = new_lifter.get_x(theta=theta_est)
        assert (x.T @ Q @ x - cost) / cost < 1e-7

        print("creating cliques...", end="")
        t1 = time.time()
        clique_list = create_clique_list_slam(
            new_lifter, use_known=USE_KNOWN, use_autotemplate=USE_AUTOTEMPLATE
        )
        time_dict["t create cliques"] = time.time() - t1
        time_dict["dim dSDP"] = clique_list[0].Q.shape[0]
        time_dict["m dSDP"] = sum(len(c.A_list) for c in clique_list)

        print("solving dSDP...", end="")
        t1 = time.time()
        X_list, info = solve_oneshot(
            clique_list,
            use_primal=True,
            use_fusion=True,
            verbose=VERBOSE,
        )
        time_dict["t dSDP"] = time.time() - t1
        time_dict["cost dSDP"] = info["cost"]

        x_dSDP, evr_mean = extract_solution(new_lifter, X_list)
        time_dict["evr dSDP"] = evr_mean

        print("creating constraints...", end="")
        t1 = time.time()
        if USE_AUTOTEMPLATE:
            new_constraints = new_lifter.apply_templates(saved_learner.templates, 0)
            Constraints = [(new_lifter.get_A0(), 1.0)] + [
                (c.A_sparse_, 0.0) for c in new_constraints
            ]
        else:
            if USE_KNOWN:
                Constraints = [(new_lifter.get_A0(), 1.0)] + [
                    (A, 0.0) for A in new_lifter.get_A_known()
                ]
            else:
                Constraints = [(new_lifter.get_A0(), 1.0)] + [
                    (A, 0.0) for A in new_lifter.get_A_learned_simple()
                ]
        time_dict["t create constraints"] = time.time() - t1

        if True:  # n_params <= 18:
            time_dict["dim SDP"] = Q.shape[0]
            time_dict["m SDP"] = len(Constraints)

            print("solving SDP...", end="")
            t1 = time.time()
            X, info = solve_sdp(
                Q, Constraints, adjust=ADJUST, primal=USE_PRIMAL, use_fusion=USE_FUSION
            )
            time_dict["t SDP"] = time.time() - t1
            time_dict["cost SDP"] = info["cost"]

            x_SDP, info = rank_project(X, p=1)
            time_dict["evr SDP"] = info["EVR"]
        print("done.")
        df_data.append(time_dict)
    return pd.DataFrame(df_data)


if __name__ == "__main__":
    np.random.seed(0)
    lifter = MatWeightLocLifter(n_landmarks=10, n_poses=10)
    lifter.ALL_PAIRS = False
    lifter.CLIQUE_SIZE = 2

    # n_params_list = np.logspace(1, 6, 11).astype(int)
    n_params_list = np.logspace(1, 2, 10).astype(int)
    # n_params_list = np.arange(10, 101, step=10).astype(int)
    # n_params_list = np.arange(10, 19)  # runs out of memory at 19!
    fname = f"_results/{lifter}_time_new.pkl"
    overwrite = False

    try:
        assert overwrite is False
        df = pd.read_pickle(fname)
    except (FileNotFoundError, AssertionError):
        df = generate_results(lifter, n_params_list=n_params_list)
        df.to_pickle(fname)

    for label, plot in zip(["t", "cost"], [sns.scatterplot, sns.barplot]):
        df_long = df.melt(
            id_vars=["n params"],
            value_vars=[f"{label} local", f"{label} dSDP", f"{label} SDP"],
            value_name=label,
            var_name="solver type",
        )
        fig, ax = plt.subplots()
        plot(df_long, x="n params", y=label, hue="solver type", ax=ax)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid("on")

    fig, ax = plt.subplots()
    ax.loglog(df["n params"], df["evr SDP"], label="SDP")
    ax.loglog(df["n params"], df["evr dSDP"], label="dSDP")
    ax.legend()
    ax.grid("on")
    ax.set_ylabel("EVR")
    ax.set_xlabel("n params")
    print("done")
