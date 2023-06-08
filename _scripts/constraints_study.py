# intialization

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from lifters.plotting_tools import savefig
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from solvers.common import find_local_minimum, solve_sdp_cvxpy
from solvers.sparse import solve_lambda

# assume strong duality when absolute gap is smaller
TOL_REL_GAP = 1e-3  # 1e-10  # was 1e-5

# tolerance for nullspace basis vectors
EPS_SVD = 1e-7
# tolerance for feasibility error of learned constraints
EPS_ERROR = 1e-8

NORMALIZE = False
METHOD = "qrp"

# TODO(FD) for some reason this is not helping for 1d problem
# remove Q[0,0] for optimization, and normalize entries
ADJUST = True

N_LANDMARKS = 3  # should be 4 for 3d?
NOISE = 1e-2
LEVEL = "urT"
SEED = 1

PARAMETER_DICT = {
    "learned": dict(  # fully learned
        add_known_redundant=False,
        use_known=True,
        incremental=False,
    ),
    "known": dict(  # fully known (only 1D)
        add_known_redundant=True,
        use_known=True,
        incremental=False,
    ),
    "incremental": dict(  # incremental
        add_known_redundant=False,
        use_known=False,
        incremental=True,
    ),
}

ORDER_PAIRS = [
    ("original", "increase"),
    ("optimization", "decrease"),
]


def generate_matrices(lifter, param, fname_root=""):
    params = PARAMETER_DICT[param]
    if (params["use_known"]) and (not params["incremental"]):
        A_known = lifter.get_A_known(add_known_redundant=params["add_known_redundant"])
    else:
        A_known = []
    print(f"adding {len(A_known)} known constraints.")

    A_all, S = lifter.get_A_learned(
        A_known=A_known,
        eps=EPS_SVD,
        normalize=NORMALIZE,
        return_S=True,
        method=METHOD,
        plot=False,
        incremental=params["incremental"],
    )
    errs, idxs = lifter.test_constraints(A_all, errors="print", tol=EPS_ERROR)
    print(f"found {len(idxs)} violating constraints")
    for idx in idxs[::-1]:
        del A_all[idx]
        del S[idx]
    n_learned = len(A_all) - len(A_known)
    print(f"left with {n_learned} learned cosntraints")

    # intermediate step: remove lin. dependant matrices from final list.
    # this should have happened earlier but for some reason there are some
    # residual dependent vectors that need to be removed.
    basis = np.concatenate([lifter.get_vec(A)[:, None] for A in A_all], axis=1)
    import scipy.linalg as la

    __, r, p = la.qr(basis, pivoting=True, mode="economic")
    rank = np.where(np.abs(np.diag(r)) > EPS_SVD)[0][-1] + 1
    if rank < len(A_all):
        A_reduced = [A_all[i] for i in p[:rank]]
        print(f"only {rank} of {len(A_all)} constraints are independent")

        # sanity check
        basis_reduced = np.concatenate(
            [lifter.get_vec(A)[:, None] for A in A_reduced], axis=1
        )
        __, r, p = la.qr(basis_reduced, pivoting=True, mode="economic")
        rank_new = np.where(np.abs(np.diag(r)) > EPS_SVD)[0][-1] + 1
        assert rank_new == rank
    else:
        A_reduced = A_all

    A_b_list_all = lifter.get_A_b_list(A_reduced)

    names = [f"A{i}:known" for i in range(len(A_known))]
    names += [f"A{len(A_known) + i}:learned" for i in range(n_learned)]
    return A_b_list_all, names


def generate_orders(A_b_list_all, xhat):
    # compute lamdas using optimization
    H, lamdas = solve_lambda(Q, A_b_list_all, xhat, force_first=1)
    if lamdas is None:
        print("Warning: problem doesn't have feasible solution!")
        lamdas = np.zeros(len(A_b_list_all))
    else:
        lamdas = np.abs(lamdas)

    # compute lambas by solving dual problem
    X, info = solve_sdp_cvxpy(Q, A_b_list_all, adjust=ADJUST)  # , rho_hat=qcqp_cost)
    if info["cost"] is None:
        print("Warning: infeasible?")
        yvals = np.zeros(len(A_b_list_all))
    else:
        yvals = np.abs(info["yvals"])
        if abs(qcqp_cost - info["cost"]) / qcqp_cost > TOL_REL_GAP:
            print("Warning: no strong duality?")
            print(f"qcqp cost: {qcqp_cost}")
            print(f"dual cost: {info['cost']}")

    # compute other scores
    order_dicts = {
        "original": np.arange(1, len(A_b_list_all)),
        "optimization": lamdas[1:],
    }
    return order_dicts


def compute_tightness(Q, A_b_here, names_here, qcqp_cost):
    from progressbar import ProgressBar

    data_tight = []

    tight_counter = 0
    min_number = None

    p = ProgressBar(max_value=len(A_b_here))
    for i in range(len(A_b_here)):
        p.update(i)
        X, info = solve_sdp_cvxpy(Q, A_b_here[: i + 1], adjust=ADJUST)

        tight = False
        if info["cost"] is not None:
            error = abs(qcqp_cost - info["cost"]) / qcqp_cost
            if error < TOL_REL_GAP:
                tight = True
                if min_number is None:
                    min_number = i
                tight_counter += 1
        else:
            print(f"Did not solve at constraint {i}")
        if tight_counter > 10:
            break

        data_tight.append(
            {
                "order_name": order_name,
                "H": info["H"],
                "A": A_b_here[i][0],
                "order_type": order_type,
                "cost": info["cost"],
                "tight": tight,
                "name": names_here[i],
            }
        )
    df_tight = pd.DataFrame(data_tight)
    return df_tight


def plot_tightness(df_tight, fname_root):
    fig, ax = plt.subplots()
    ax.axhline(qcqp_cost, color="k")
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        try:
            number = np.where(df_sub.tight.values)[0][0]
            label = f"{order_name} {order_type}: {number}"
        except IndexError:
            number = None
            label = f"{order_name} {order_type}: never tight"
        ax.semilogy(range(1, len(df_sub) + 1), np.abs(df_sub.cost.values), label=label)
    ax.legend()
    ax.grid()
    if fname_root != "":
        savefig(fig, fname_root + f"_tightness.png")


def plot_matrices(df_tight, fname_root):
    from lifters.plotting_tools import plot_matrix
    import itertools
    from math import ceil

    # for (order, order_type), df_sub in df_tight.groupby(["order", "type"]):
    matrix_types = ["A", "H"]
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        A_agg = None
        H = None

        n_cols = 10
        n_rows = min(ceil(len(df_sub) / n_cols), 5)  # plot maximum 5 * 2 rows
        fig, axs = plt.subplots(n_rows * 2, n_cols)
        fig.set_size_inches(n_cols, n_rows * 2)
        for j, matrix_type in enumerate(matrix_types):
            matrices = df_sub[matrix_type].values
            names_here = df_sub.name.values
            costs = df_sub.cost.values

            # make sure it's symmetric
            vmin = np.min([np.min(A) for A in matrices if (A is not None)])
            vmax = np.max([np.max(A) for A in matrices if (A is not None)])
            vmin = min(vmin, -vmax)
            vmax = max(vmax, -vmin)

            i = 0

            for row, col in itertools.product(range(n_rows), range(n_cols)):
                ax = axs[row * 2 + j, col]
                if i < len(matrices):
                    if matrices[i] is None:
                        continue
                    title = f"{matrix_type}{i}"
                    if matrix_type == "A":
                        title += f"\n{names_here[i]}"
                    else:
                        title += f"\nc={costs[i]:.2e}"

                    plot_matrix(
                        ax=ax,
                        Ai=matrices[i],
                        vmin=vmin,
                        vmax=vmax,
                        title=title,
                        colorbar=False,
                    )
                    ax.set_title(title, fontsize=5)

                    if matrix_type == "A":
                        if A_agg is None:
                            A_agg = np.abs(matrices[i].toarray()) > 1e-10
                        else:
                            A_agg = np.logical_or(
                                A_agg, (np.abs(matrices[i].toarray()) > 1e-10)
                            )
                    elif matrix_type == "H":
                        # make sure we store the last valid estimate of H, for plotting
                        if matrices[i] is not None:
                            H = matrices[i]
                else:
                    ax.axis("off")
                i += 1

        fname = fname_root + f"_{order_name}_{order_type}.png"
        savefig(fig, fname)

        matrix_agg = {
            "A": A_agg.astype(int),
            "H": H,
            "Q": Q.toarray(),
        }
        for matrix_type, matrix in matrix_agg.items():
            if matrix is None:
                continue
            fname = fname_root + f"_{order_name}_{order_type}_{matrix_type}.png"
            fig, ax = plt.subplots()
            plot_matrix(
                ax=ax,
                Ai=matrix,
                colorbar=True,
                title=matrix_type,
            )
            savefig(fig, fname)


def interpret_dataframe(lifter, A_b_list_all, order_dicts, fname_root):
    # create a new dataframe that is more readible than the expressions.
    from lifters.interpret import get_known_variables
    from poly_matrix.poly_matrix import PolyMatrix

    # expressions = []
    landmark_dict = get_known_variables(lifter)
    print("known values:", landmark_dict.keys())
    data_math = []
    for i, (A, b) in enumerate(A_b_list_all):
        # print(names[i])
        try:
            A_poly, var_dict = PolyMatrix.init_from_sparse(
                A, lifter.var_dict, unfold=True
            )
        except Exception as e:
            print(f"error at {i}:", e)
            continue
        sparse_series = A_poly.interpret(var_dict)
        data_math.append(sparse_series)

    df_math = pd.DataFrame(data=data_math, dtype="Sparse[object]")
    for name, values in order_dicts.items():
        df_math.loc[:, name] = values
    df_math.loc[:, "name"] = names

    def sort_fun(series):
        return series.isna()

    df_math.dropna(axis=1, how="all", inplace=True)
    df_math.sort_values(
        key=sort_fun,
        by=list(df_math.columns),
        axis=0,
        na_position="last",
        inplace=True,
    )
    if fname_root != "":
        fname = fname_root + "_math.pkl"
        pd.to_pickle(df_math, fname)
        print("saved math as", fname)
        fname = fname_root + "_math.csv"
        df_math.to_csv(fname)
        print("saved math as", fname)


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import itertools

    root = Path(__file__).resolve().parents[1]

    recompute_matrices = True
    recompute_tightness = True

    n_matrices = None  # for debugging only. set to None to use all
    # for d, param in itertools.product([3], ["learned"]):
    # for d, param in itertools.product([2, 3], ["learned", "incremental"]):
    for d, param in itertools.product([1], ["known", "learned", "incremental"]):
        fname_root = str(root / f"_results/experiments_{d}d_{param}")

        np.random.seed(SEED)
        if d == 1:
            lifter = Stereo1DLifter(n_landmarks=N_LANDMARKS)
        elif d == 2:
            lifter = Stereo2DLifter(n_landmarks=N_LANDMARKS, level=LEVEL)
        elif d == 3:
            lifter = Stereo3DLifter(n_landmarks=N_LANDMARKS, level=LEVEL)
        lifter.generate_random_setup()
        # solve locally
        np.random.seed(SEED)
        Q, y = lifter.get_Q(noise=NOISE)

        fname = fname_root + "_data.pkl"
        if not recompute_matrices:
            with open(fname, "rb") as f:
                A_b_list_all = pickle.load(f)
                names = pickle.load(f)
                order_dict = pickle.load(f)
                qcqp_cost = pickle.load(f)
                xhat = pickle.load(f)
        else:
            A_b_list_all, names = generate_matrices(lifter, param, fname_root)

            # increase how many constraints we add to the problem
            qcqp_that, qcqp_cost = find_local_minimum(lifter, y=y)
            xhat = lifter.get_x(qcqp_that)
            if qcqp_cost is None:
                print("Warning: could not solve local.")
            elif qcqp_cost < 1e-7:
                print("Warning: too low qcqp cost, numerical issues.")

            order_dict = generate_orders(A_b_list_all, xhat)
            with open(fname, "wb") as f:
                pickle.dump(A_b_list_all, f)
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

        # plot the tightness for different orders
        plot_tightness(df_tight, fname_root)

        # plot the matrices
        plot_matrices(df_tight, fname_root)

        # interpret the obtained dataframe
        # interpret_dataframe(lifter, A_b_list_all, order_dict, fname_root)
