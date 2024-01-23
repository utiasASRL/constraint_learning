import itertools
import time

import numpy as np
import pandas as pd
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


def generate_results(
    lifter: MatWeightLocLifter,
    n_params_list=[10],
    noise_list=[1e-2],
    sparsity_list=[1.0],
    n_seeds=1,
    fname="",
    use_methods=USE_METHODS,
):
    if USE_AUTOTEMPLATE:
        saved_learner = read_saved_learner(lifter)

    df_data = []
    for n_params, noise, sparsity in itertools.product(
        n_params_list, noise_list, sparsity_list
    ):
        for seed in range(n_seeds):
            np.random.seed(NOISE_SEED)
            print(f"N={n_params}: ", end="")
            data_dict = {
                "n params": n_params,
                "noise": noise,
                "sparsity": sparsity,
                "seed": seed,
            }
            new_lifter = create_newinstance(lifter, n_params=n_params)

            Q, y = new_lifter.get_Q(noise=noise)

            theta_gt = new_lifter.get_vec_around_gt(delta=0)

            print("solving local...", end="")
            t1 = time.time()
            theta_est, info, cost = new_lifter.local_solver(theta_gt, new_lifter.y_)
            data_dict["t local"] = time.time() - t1
            data_dict["cost local"] = cost
            print(f"cost local: {cost:.2f}")

            print("creating cliques...", end="")
            t1 = time.time()

            for add_redundant, appendix in zip([False, True], ["", "-redun"]):
                clique_list = create_clique_list_loc(
                    new_lifter,
                    use_known=USE_KNOWN,
                    use_autotemplate=USE_AUTOTEMPLATE,
                )
                data_dict["t create cliques"] = time.time() - t1
                data_dict["dim dSDP"] = clique_list[0].Q.shape[0]
                data_dict["m dSDP"] = sum(len(c.A_list) for c in clique_list)

                method = f"dSDP{appendix}"
                if method in use_methods:
                    print(f"solving {method}...", end="")
                    t1 = time.time()
                    X_list, info = solve_oneshot(
                        clique_list,
                        use_primal=True,
                        use_fusion=USE_FUSION,
                        verbose=VERBOSE,
                        tol=TOL_DSDP,
                    )
                    data_dict[f"t {method}"] = time.time() - t1
                    data_dict[f"cost {method}"] = info["cost"]
                    print(f"cost {method}: {info['cost']:.2f}")

                    try:
                        x_dSDP, evr_mean = extract_solution(new_lifter, X_list)
                        data_dict[f"evr {method}"] = evr_mean
                    except:
                        print("Could not extract solution")

                method = f"ADMM{appendix}"
                if method in use_methods:
                    print(f"running {method}...", end="")
                    if lifter.ADMM_INIT_XHAT:
                        X0 = []
                        for c in clique_list:
                            x_clique = new_lifter.get_x(
                                theta=theta_est, var_subset=c.var_dict
                            )
                            X0.append(np.outer(x_clique, x_clique))
                    else:
                        X0 = None
                    print("target", info["cost"])
                    t1 = time.time()
                    X_list, info = solve_alternating(
                        clique_list, X0=X0, verbose=True, **lifter.ADMM_OPTIONS
                    )
                    data_dict[f"t {method}"] = time.time() - t1
                    print(info["msg"], end="...")
                    data_dict[f"cost {method}"] = info["cost"]
                    print(f"cost {method}: {info['cost']:.2f}")

                    try:
                        x_ADMM, evr_mean = extract_solution(new_lifter, X_list)
                        data_dict[f"evr {method}"] = evr_mean
                    except:
                        print("Warning: could not extract solution")

                method = f"SDP{appendix}"
                if method in use_methods:
                    print("creating constraints...", end="")
                    t1 = time.time()
                    if USE_AUTOTEMPLATE:
                        new_constraints = new_lifter.apply_templates(
                            saved_learner.templates, 0
                        )
                        Constraints = [(new_lifter.get_A0(), 1.0)] + [
                            (c.A_sparse_, 0.0) for c in new_constraints
                        ]
                    else:
                        if USE_KNOWN:
                            Constraints = [(new_lifter.get_A0(), 1.0)] + [
                                (A, 0.0) for A in new_lifter.get_A_known()
                            ]
                            if add_redundant:
                                Constraints += [
                                    (A, 0.0) for A in new_lifter.get_A_redundant()
                                ]
                        else:
                            Constraints = [(new_lifter.get_A0(), 1.0)] + [
                                (A, 0.0) for A in new_lifter.get_A_learned_simple()
                            ]
                    data_dict["t create constraints"] = time.time() - t1

                    if True:  # n_params <= 18:
                        data_dict[f"dim {method}"] = Q.shape[0]
                        data_dict[f"m {method}"] = len(Constraints)

                        print(f"solving {method}...", end="")
                        t1 = time.time()
                        X, info = solve_sdp(
                            Q,
                            Constraints,
                            adjust=ADJUST,
                            primal=USE_PRIMAL,
                            use_fusion=USE_FUSION,
                            tol=TOL_SDP,
                        )
                        data_dict[f"t {method}"] = time.time() - t1
                        data_dict[f"cost {method}"] = info["cost"]
                        print(f"cost {method}: {info['cost']:.2f}")

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
