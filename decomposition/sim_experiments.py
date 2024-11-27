import itertools
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from cert_tools.admm_solvers import solve_alternating, solve_parallel
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.sparse_solvers import solve_oneshot

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

# Maximum size for which we run SDP (to prevent memory error)
# Determined empirically. Set to inf for no effect.
SDP_MAX_N = 500

# USE_METHODS = ["SDP", "dSDP", "ADMM"]
USE_METHODS = ["local"]
# USE_METHODS = ["dSDP", "SDP"]

DEBUG = False
TOL_SDP = 1e-10
TOL_DSDP = 1e-10

ADD_REDUNDANT = False

# if EVR is bigger than EVR_THRESH, we consider success of SDP solvers.
EVR_THRESH = 1e7


def read_saved_learner(lifter):
    import pickle

    fname = f"_results/scalability_{lifter}_order_dict.pkl"
    try:
        with open(fname, "rb") as f:
            order_dict = pickle.load(f)
            saved_learner = pickle.load(f)
    except FileNotFoundError:
        print(f"did not find saved learner for {lifter}. Run run_..._study.py first.")
        raise
    return saved_learner


def extract_solution(lifter: MatWeightLocLifter | RangeOnlyLocLifter, X_list):
    # x_list has format [1, x1, x2], [1, x2, x3], [1,  x3, x4], ...
    x_dim = lifter.node_size()
    x, info = rank_project(X_list[0], p=1)
    evr_mean = [info["EVR"]]
    x_all = [x[1 : 1 + x_dim]]  # x1
    for X in X_list:
        x, info = rank_project(X, p=1)
        x_all.append(x[1 + x_dim : 1 + 2 * x_dim])  # x2
        evr_mean.append(info["EVR"])
    return np.vstack(x_all), np.mean(evr_mean)


def get_relative_gap(cost_sdp, cost_global):
    if cost_sdp is None:
        return np.inf
    else:
        return abs(cost_global - cost_sdp) / (1 + abs(cost_global) + abs(cost_sdp))


def generate_results(
    lifter: MatWeightLocLifter,
    n_params_list=[10],
    noise_list=[1e-2],
    sparsity_list=[1.0],
    n_threads_list=[12],
    n_seeds=1,
    fname="",
    use_methods=USE_METHODS,
    add_redundant_constr=ADD_REDUNDANT,
):
    if add_redundant_constr:
        red_parameters = {"": False, "-redun": True}
    else:
        red_parameters = {"": False}

    if USE_AUTOTEMPLATE:
        saved_learner = read_saved_learner(lifter)

    df_data = []
    for n_params, noise, sparsity in itertools.product(
        n_params_list, noise_list, sparsity_list
    ):
        for seed in range(n_seeds):
            print(
                f"\t\t\t\t\t\t\t\t\t--- N={n_params} noise={noise:.2e} sparsity={sparsity:.1f} --- {seed+1}/{n_seeds}",
            )
            data_dict = {
                "n params": n_params,
                "noise": noise,
                "sparsity": sparsity,
                "seed": seed,
            }
            new_lifter = create_newinstance(lifter, n_params=n_params)

            print("fill graph...")
            np.random.seed(seed)
            new_lifter.generate_random_setup()
            new_lifter.simulate_y(noise=noise, sparsity=sparsity)

            theta_gt = new_lifter.get_vec_around_gt(delta=0)
            theta_rand = new_lifter.get_vec_around_gt(delta=0.5)

            if "local-gt" in use_methods:
                print("solving local with gt...")
                t1 = time.time()
                theta_est, info, cost = new_lifter.local_solver(
                    theta_gt, new_lifter.y_, verbose=True
                )
                data_dict["success local-gt"] = info["success"]
                data_dict["t local-gt"] = time.time() - t1
                data_dict["cost local-gt"] = cost
                for key, val in new_lifter.get_error(theta_hat=theta_est).items():
                    data_dict[f"{key} local-gt"] = val
                print(f"cost local: {cost:.2f}")
            else:
                info = {"cost-gt": 0}
                data_dict["cost local-gt"] = 0
            if "local" in use_methods:
                print("solving local with random...")
                t1 = time.time()
                theta_est, info, cost = new_lifter.local_solver(
                    theta_rand, new_lifter.y_, verbose=True
                )
                data_dict["success local"] = info["success"]
                data_dict["t local"] = time.time() - t1
                data_dict["cost local"] = cost
                for key, val in new_lifter.get_error(theta_hat=theta_est).items():
                    data_dict[f"{key} local"] = val
                print(f"cost local: {cost:.2f}")
            else:
                info = {"cost": 0}
                data_dict["cost local"] = 0

            for appendix, add_redundant in red_parameters.items():
                print("creating cliques...")
                t1 = time.time()
                clique_list = create_clique_list_loc(
                    new_lifter,
                    use_known=USE_KNOWN,
                    use_autotemplate=USE_AUTOTEMPLATE,
                    add_redundant=add_redundant,
                )
                data_dict["t create cliques"] = time.time() - t1

                method = f"dSDP{appendix}"
                if method in use_methods:
                    print(f"solving {method}...")
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
                        info["cost"], data_dict["cost local-gt"]
                    )
                    print(f"cost {method}: {info['cost']:.2f}")
                    x_dSDP, evr_mean = extract_solution(new_lifter, X_list)
                    theta_dSDP = new_lifter.get_theta_from_x(
                        x=np.hstack([1.0, x_dSDP.flatten()])
                    )
                    for key, val in new_lifter.get_error(theta_hat=theta_dSDP).items():
                        data_dict[f"{key} {method}"] = val

                    data_dict[f"EVR {method}"] = evr_mean
                    data_dict[f"success {method}"] = evr_mean > EVR_THRESH

                method = f"ADMM{appendix}"
                if method in use_methods:
                    print(f"running {method}...")
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
                    # do deepcopy to make sure we can can use clique_list again.
                    X_list, info = solve_alternating(
                        deepcopy(clique_list),
                        X0=X0,
                        verbose=VERBOSE,
                        **lifter.ADMM_OPTIONS,
                    )
                    data_dict[f"t {method}"] = time.time() - t1
                    print(info["msg"])
                    data_dict[f"cost {method}"] = info["cost"]
                    data_dict[f"RDG {method}"] = get_relative_gap(
                        info["cost"], data_dict["cost local-gt"]
                    )
                    print(f"cost {method}: {info['cost']:.2f}")

                    x_ADMM, evr_mean = extract_solution(new_lifter, X_list)
                    theta_ADMM = new_lifter.get_theta_from_x(x=x_ADMM)
                    for key, val in new_lifter.get_error(theta_hat=theta_ADMM).items():
                        data_dict[f"{key} {method}"] = val
                    data_dict[f"EVR {method}"] = evr_mean
                    data_dict[f"success {method}"] = evr_mean > EVR_THRESH

                method = f"pADMM{appendix}"
                if method in use_methods:
                    for n_threads in n_threads_list:
                        if len(n_threads_list) > 1:
                            method = f"pADMM{appendix}-{n_threads}"
                        print(f"running {method}...")
                        if lifter.ADMM_INIT_XHAT:
                            X0 = []
                            for c in clique_list:
                                x_clique = new_lifter.get_x(
                                    theta=theta_est, var_subset=c.var_dict
                                )
                                X0.append(np.outer(x_clique, x_clique))
                        else:
                            X0 = None
                        print("target", data_dict["cost local-gt"])
                        t1 = time.time()
                        X_list, info = solve_parallel(
                            deepcopy(clique_list),
                            X0=X0,
                            verbose=False,
                            n_threads=n_threads,
                            **lifter.ADMM_OPTIONS,
                        )
                        data_dict[f"t total {method}"] = time.time() - t1
                        data_dict[f"t {method}"] = info["time running"]
                        data_dict[f"n threads"] = n_threads
                        print(info["msg"])
                        data_dict[f"cost {method}"] = info["cost"]
                        data_dict[f"cost history {method}"] = info["cost history"]
                        data_dict[f"RDG {method}"] = get_relative_gap(
                            info["cost"], data_dict["cost local-gt"]
                        )
                        print(f"cost {method}: {info['cost']:.2f}")

                        x_ADMM, evr_mean = extract_solution(new_lifter, X_list)
                        theta_ADMM = new_lifter.get_theta_from_x(
                            x=np.hstack([1.0, x_ADMM.flatten()])
                        )
                        for key, val in new_lifter.get_error(
                            theta_hat=theta_ADMM
                        ).items():
                            data_dict[f"{key} {method}"] = val
                        data_dict[f"EVR {method}"] = evr_mean
                        data_dict[f"success {method}"] = evr_mean > EVR_THRESH

                method = f"SDP{appendix}"
                if method in use_methods and (n_params > SDP_MAX_N):
                    print(f"Skipping SDP for n_params={n_params}")
                elif method in use_methods and (n_params <= SDP_MAX_N):
                    print("creating cost...")
                    Q, _ = new_lifter.get_Q()

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

                    data_dict[f"dim {method}"] = Q.shape[0]
                    data_dict[f"m {method}"] = len(Constraints)

                    print(f"solving {method}...")
                    t1 = time.time()
                    X, info = solve_sdp(
                        Q,
                        Constraints,
                        adjust=ADJUST,
                        primal=USE_PRIMAL,
                        use_fusion=USE_FUSION,
                        tol=TOL_SDP,
                        verbose=VERBOSE,
                    )
                    data_dict[f"t {method}"] = time.time() - t1
                    data_dict[f"cost {method}"] = info["cost"]
                    data_dict[f"RDG {method}"] = get_relative_gap(
                        info["cost"], data_dict["cost local-gt"]
                    )
                    print(f"cost {method}: {info['cost']:.2f}")

                    x_SDP, info = rank_project(X, p=1)
                    theta_SDP = new_lifter.get_theta_from_x(x=x_SDP)
                    for key, val in new_lifter.get_error(theta_hat=theta_SDP).items():
                        data_dict[f"{key} {method}"] = val
                    data_dict[f"EVR {method}"] = info["EVR"]
                    data_dict[f"success {method}"] = info["EVR"] > EVR_THRESH

            df_data.append(deepcopy(data_dict))
            if fname != "":
                df = pd.DataFrame(df_data)
                df.to_pickle(fname)
                print("saved intermediate as", fname)
    return pd.DataFrame(df_data)
