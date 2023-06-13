# intialization

import numpy as np
import pandas as pd

from lifters.stereo1d_slam_lifter import Stereo1DSLAMLifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from solvers.common import find_local_minimum

from utils.analyze_tightness import (
    generate_matrices,
    generate_orders,
    compute_tightness,
)
from utils.interpret import interpret_dataframe
from utils.plotting_tools import plot_basis, plot_tightness, plot_matrices

N_LANDMARKS = 3  # should be 4 for 3d
NOISE = 1e-2
SEED = 1
ADD_PARAMS = True

ORDER_PAIRS = [
    ("original", "increase"),
    ("optimization", "decrease"),
]
D_LIST = [2]
PARAM_LIST = ["incremental"]  # , "learned"]

if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import itertools

    root = Path(__file__).resolve().parents[1]

    recompute_matrices = True
    recompute_tightness = True

    n_matrices = None  # for debugging only. set to None to use all
    for d, param in itertools.product(D_LIST, PARAM_LIST):
        np.random.seed(SEED)
        if d == 1:
            # lifter = Stereo1DSLAMLifter(n_landmarks=N_LANDMARKS, level="all")  # za
            lifter = Stereo1DLifter(n_landmarks=N_LANDMARKS, add_parameters=ADD_PARAMS)
        elif d == 2:
            lifter = Stereo2DLifter(
                n_landmarks=N_LANDMARKS, level="urT", add_parameters=ADD_PARAMS
            )
        elif d == 3:
            lifter = Stereo3DLifter(
                n_landmarks=N_LANDMARKS, level="urT", add_parameters=ADD_PARAMS
            )
        lifter.generate_random_setup()

        fname_root = str(root / f"_results/experiments_{lifter}_{param}")

        # solve locally
        np.random.seed(SEED)
        Q, y = lifter.get_Q(noise=NOISE)

        fname = fname_root + "_data.pkl"
        if not recompute_matrices:
            with open(fname, "rb") as f:
                A_b_list_all = pickle.load(f)
                basis = pickle.load(f)
                names = pickle.load(f)
                order_dict = pickle.load(f)
                qcqp_cost = pickle.load(f)
                xhat = pickle.load(f)
        else:
            A_b_list_all, basis, names = generate_matrices(lifter, param, fname_root)

            # increase how many constraints we add to the problem
            qcqp_that, qcqp_cost = find_local_minimum(lifter, y=y, verbose=False)
            xhat = lifter.get_x(qcqp_that)
            if qcqp_cost is None:
                print("Warning: could not solve local.")
            elif qcqp_cost < 1e-7:
                print("Warning: too low qcqp cost, numerical issues.")

            order_dict = generate_orders(Q, A_b_list_all, xhat, qcqp_cost)
            with open(fname, "wb") as f:
                pickle.dump(A_b_list_all, f)
                pickle.dump(basis, f)
                pickle.dump(names, f)
                pickle.dump(order_dict, f)
                pickle.dump(qcqp_cost, f)
                pickle.dump(xhat, f)
            print("saved matrices as", fname)

        fname = fname_root + "_tight.pkl"
        if not recompute_tightness:
            df_tight = pd.read_pickle(fname)
        else:
            dfs = []
            for order_name, order_type in ORDER_PAIRS:
                print(f"{order_name} {order_type}")
                step = 1 if order_type == "increase" else -1
                order_arrays = order_dict[order_name]
                order = np.argsort(order_arrays)[::step]

                A_b_here = [A_b_list_all[0]] + [A_b_list_all[s + 1] for s in order]
                names_here = ["A0"] + [names[s] for s in order]

                if n_matrices is not None:
                    df_tight_order = compute_tightness(
                        Q, A_b_here[:n_matrices], names_here, qcqp_cost
                    )
                else:
                    df_tight_order = compute_tightness(
                        Q, A_b_here[:n_matrices], names_here, qcqp_cost
                    )
                df_tight_order.loc[:, "order_name"] = order_name
                df_tight_order.loc[:, "order_type"] = order_type
                dfs.append(df_tight_order)

            df_tight = pd.concat(dfs)
            pd.to_pickle(df_tight, fname)
            print("saved values as", fname)

        plot_basis(basis, lifter, fname_root)

        # plot the tightness for different orders
        plot_tightness(df_tight, qcqp_cost, fname_root)

        # plot the matrices
        plot_matrices(df_tight, fname_root)

        # interpret the obtained dataframe
        # interpret_dataframe(lifter, A_b_list_all, order_dict, fname_root)
