import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cert_tools.admm_solvers import solve_alternating
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.sparse_solvers import solve_oneshot

from _scripts.generate_cliques import create_clique_list_loc
from _scripts.run_clique_study import read_saved_learner
from auto_template.sim_experiments import create_newinstance
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.cert_matrix import get_cost_matrices
from ro_certs.generate_cliques import generate_clique_list
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

NOISE_SEED = 0

USE_KNOWN = True
USE_AUTOTEMPLATE = False

ADJUST = True
USE_PRIMAL = False
USE_FUSION = True
VERBOSE = False

# USE_METHODS = ["SDP", "dSDP", "ADMM"]
USE_METHODS = ["SDP", "dSDP", "ADMM"]
# USE_METHODS = ["dSDP", "SDP"]

DEBUG = False
TOL_SDP = 1e-10
TOL_DSDP = 1e-5


def extract_solution(lifter: MatWeightLocLifter, X_list):
    x_dim = lifter.node_size()
    x, info = rank_project(X_list[0], p=1)
    evr_mean = [info["EVR"]]
    x_all = [x[1 : 1 + x_dim]]
    for X in X_list:
        x, info = rank_project(X, p=1)
        x_all.append(x[1 + x_dim : 1 + 2 * x_dim])
        evr_mean.append(info["EVR"])
    return np.vstack(x_all), np.mean(evr_mean)


def generate_results(lifter: MatWeightLocLifter, n_params_list=[10], fname=""):
    if USE_AUTOTEMPLATE:
        saved_learner = read_saved_learner(lifter)

    df_data = []
    for n_params in n_params_list:
        print(f"N={n_params}: ", end="")
        data_dict = {"n params": n_params}
        new_lifter = create_newinstance(lifter, n_params=n_params)

        np.random.seed(NOISE_SEED)
        Q, y = new_lifter.get_Q()

        theta_gt = new_lifter.get_vec_around_gt(delta=0)

        print("solving local...", end="")
        t1 = time.time()
        theta_est, info, cost = new_lifter.local_solver(theta_gt, new_lifter.y_)
        data_dict["t local"] = time.time() - t1
        data_dict["cost local"] = cost
        x = new_lifter.get_x(theta=theta_est)
        assert (x.T @ Q @ x - cost) / cost < 1e-7
        print(f"cost local: {cost:.2f}")

        print("creating cliques...", end="")
        t1 = time.time()

        clique_list = create_clique_list_loc(
            new_lifter, use_known=USE_KNOWN, use_autotemplate=USE_AUTOTEMPLATE
        )
        if isinstance(new_lifter, RangeOnlyLocLifter) and DEBUG:
            cost_matrices = get_cost_matrices(new_lifter.prob)
            clique_list_new = generate_clique_list(new_lifter.prob, cost_matrices)
            for c1, c2 in zip(clique_list_new, clique_list):
                ii1, jj1 = c1.Q.nonzero()
                ii2, jj2 = c2.Q.nonzero()
                np.testing.assert_allclose(ii1, ii2)
                np.testing.assert_allclose(jj1, jj2)
                np.testing.assert_allclose(c1.Q.data, c2.Q.data)
                for i, (A1, A2) in enumerate(zip(c1.A_list, c2.A_list)):
                    np.testing.assert_allclose(A1.toarray(), A2.toarray())

        data_dict["t create cliques"] = time.time() - t1
        data_dict["dim dSDP"] = clique_list[0].Q.shape[0]
        data_dict["m dSDP"] = sum(len(c.A_list) for c in clique_list)

        if "dSDP" in USE_METHODS:
            print("solving dSDP...", end="")
            t1 = time.time()
            X_list, info = solve_oneshot(
                clique_list,
                use_primal=True,
                use_fusion=True,
                verbose=VERBOSE,
                tol=TOL_DSDP,
            )
            data_dict["t dSDP"] = time.time() - t1
            data_dict["cost dSDP"] = info["cost"]
            print(f"cost dSDP: {info['cost']:.2f}")

            try:
                x_dSDP, evr_mean = extract_solution(new_lifter, X_list)
                data_dict["evr dSDP"] = evr_mean
            except:
                print("Could not extract solution")

        if "ADMM" in USE_METHODS:
            print("running ADMM...", end="")
            if lifter.ADMM_INIT_XHAT:
                X0 = []
                for c in clique_list:
                    x_clique = new_lifter.get_x(theta=theta_est, var_subset=c.var_dict)
                    X0.append(np.outer(x_clique, x_clique))
            else:
                X0 = None
            print("target", info["cost"])
            t1 = time.time()
            X_list, info = solve_alternating(
                clique_list, X0=X0, verbose=True, **lifter.ADMM_OPTIONS
            )
            data_dict["t ADMM"] = time.time() - t1
            print(info["msg"], end="...")
            data_dict["cost ADMM"] = info["cost"]
            print(f"cost ADMM: {info['cost']:.2f}")

            try:
                x_ADMM, evr_mean = extract_solution(new_lifter, X_list)
                data_dict["evr ADMM"] = evr_mean
            except:
                print("Warning: could not extract solution")

        if "SDP" in USE_METHODS:
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
            data_dict["t create constraints"] = time.time() - t1

            if True:  # n_params <= 18:
                data_dict["dim SDP"] = Q.shape[0]
                data_dict["m SDP"] = len(Constraints)

                print("solving SDP...", end="")
                t1 = time.time()
                X, info = solve_sdp(
                    Q,
                    Constraints,
                    adjust=ADJUST,
                    primal=USE_PRIMAL,
                    use_fusion=USE_FUSION,
                    tol=TOL_SDP,
                )
                data_dict["t SDP"] = time.time() - t1
                data_dict["cost SDP"] = info["cost"]
                print(f"cost SDP: {info['cost']:.2f}")

                try:
                    x_SDP, info = rank_project(X, p=1)
                    data_dict["evr SDP"] = info["EVR"]
                except:
                    print("Could not extract solution")

        # from ro_certs.generate_cliques import combine
        # Q_test = combine(clique_list=clique_list)

        print("done.")
        df_data.append(data_dict)
        if fname != "":
            df = pd.DataFrame(df_data)
            df.to_pickle(fname)
            print("saved intermediate as", fname)
    return pd.DataFrame(df_data)


if __name__ == "__main__":
    np.random.seed(0)

    # n_params_list = np.logspace(1, 2, 10).astype(int)
    # appendix = "time"

    n_params_list = np.logspace(1, 3, 9).astype(int)
    appendix = "large"

    # n_params_list = [10, 20]
    # appendix = "test"
    overwrite = True

    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    for lifter in [lifter_ro, lifter_mat]:
        lifter.ALL_PAIRS = False
        lifter.CLIQUE_SIZE = 2

        fname = f"_results/{lifter}_{appendix}.pkl"

        try:
            assert overwrite is False
            df = pd.read_pickle(fname)
        except (FileNotFoundError, AssertionError):
            df = generate_results(lifter, n_params_list=n_params_list, fname=fname)
            df.to_pickle(fname)
            print("saved final as", fname)

        for label, plot in zip(["t", "cost"], [sns.scatterplot, sns.barplot]):
            value_vars = [
                f"{label} local",
                f"{label} dSDP",
                f"{label} SDP",
                f"{label} ADMM",
            ]
            value_vars = set(value_vars).intersection(df.columns.unique())
            df_long = df.melt(
                id_vars=["n params"],
                value_vars=value_vars,
                value_name=label,
                var_name="solver type",
            )
            fig, ax = plt.subplots()
            plot(df_long, x="n params", y=label, hue="solver type", ax=ax)
            ax.set_yscale("log")
            if label != "cost":
                ax.set_xscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}.png"))

        fig, ax = plt.subplots()
        value_vars = ["evr SDP", "evr dSDP", "evr ADMM"]
        value_vars = set(value_vars).intersection(df.columns.unique())
        for v in value_vars:
            ax.loglog(df["n params"], df[v], label=v.strip("evr "))
        ax.legend()
        ax.grid("on")
        ax.set_ylabel("EVR")
        ax.set_xlabel("n params")
        savefig(fig, fname.replace(".pkl", f"_evr.png"))
        print("done")
