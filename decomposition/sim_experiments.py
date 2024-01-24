import itertools
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from cert_tools.admm_solvers import solve_alternating
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.sparse_solvers import solve_oneshot

from _scripts.run_clique_study import read_saved_learner
from auto_template.sim_experiments import create_newinstance
from decomposition.generate_cliques import create_clique_list_loc
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter

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
TOL_SDP = 1e-12
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


def get_relative_gap(cost_sdp, cost_global):
    """See Yang & Carlone 2023"""
    return abs(cost_global - cost_sdp) / (1 + abs(cost_global) + abs(cost_sdp))


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
            np.random.seed(seed)
            print(
                f"\t\t\t\t\t\t\t\t\t--- N={n_params} noise={noise:.2e} sparsity={sparsity:.1f} --- {seed+1}/{n_seeds}",
            )
            data_dict = {
                "n params": n_params,
                "noise": noise,
                "sparsity": sparsity,
                "seed": seed,
            }
            print("create instance...")
            new_lifter = create_newinstance(lifter, n_params=n_params)

            print("create Q...")
            Q, _ = new_lifter.get_Q(noise=noise, sparsity=sparsity)

            theta_gt = new_lifter.get_vec_around_gt(delta=0)

            print("solving local...", end="")
            t1 = time.time()
            theta_est, info, cost = new_lifter.local_solver(
                theta_gt, new_lifter.y_, verbose=True
            )
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
                    add_redundant=add_redundant,
                )
                data_dict["t create cliques"] = time.time() - t1
                data_dict[f"dim dSDP{appendix}"] = clique_list[0].Q.shape[0]
                data_dict[f"m dSDP{appendix}"] = sum(len(c.A_list) for c in clique_list)

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
                    data_dict[f"RDG {method}"] = get_relative_gap(
                        info["cost"], data_dict["cost local"]
                    )
                    print(f"cost {method}: {info['cost']:.2f}")

                    try:
                        x_dSDP, evr_mean = extract_solution(new_lifter, X_list)
                        data_dict[f"EVR {method}"] = evr_mean
                    except Exception as e:
                        print("Could not extract solution:", e)

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
                    data_dict[f"RDG {method}"] = get_relative_gap(
                        info["cost"], data_dict["cost local"]
                    )
                    print(f"cost {method}: {info['cost']:.2f}")

                    try:
                        x_ADMM, evr_mean = extract_solution(new_lifter, X_list)
                        data_dict[f"EVR {method}"] = evr_mean
                    except Exception as e:
                        print("Warning: could not extract solution:", e)

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
                                (A, 0.0)
                                for A in new_lifter.get_A_known(
                                    add_redundant=add_redundant
                                )
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
                        data_dict[f"RDG {method}"] = get_relative_gap(
                            info["cost"], data_dict["cost local"]
                        )
                        print(f"cost {method}: {info['cost']:.2f}")

                        try:
                            x_SDP, info = rank_project(X, p=1)
                            data_dict[f"EVR {method}"] = info["EVR"]
                        except Exception as e:
                            print("Could not extract solution:", e)

            df_data.append(deepcopy(data_dict))
            if fname != "":
                df = pd.DataFrame(df_data)
                df.to_pickle(fname)
                print("saved intermediate as", fname)
    return pd.DataFrame(df_data)
