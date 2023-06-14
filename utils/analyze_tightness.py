import numpy as np
import pandas as pd

from solvers.common import solve_sdp_cvxpy
from solvers.sparse import solve_lambda

from lifters.state_lifter import StateLifter

# assume strong duality when gap is smaller than this
TOL_REL_GAP = 1e-3  # 1e-10  # was 1e-5

# tolerance for nullspace basis vectors
EPS_SVD = 1e-5

NORMALIZE = False
METHOD = "qrp"

# TODO(FD) for some reason this is not helping for 1d problem
# remove Q[0,0] for optimization, and normalize entries
ADJUST = True


PARAMETER_DICT = {
    "learned": dict(  # fully learned
        add_known_redundant=False,
        use_known=False,
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


def generate_matrices(lifter: StateLifter, param):
    params = PARAMETER_DICT[param]
    if params["use_known"]:
        A_known = lifter.get_A_known(add_known_redundant=params["add_known_redundant"])
    else:
        A_known = []
    print(f"adding {len(A_known)} known constraints.")

    # find the patterns of constraints that can easily be generalized to any number of constraints.
    basis_dict = lifter.get_basis_dict(
        A_known=A_known,
        eps=EPS_SVD,
        method=METHOD,
        plot=False,
        incremental=params["incremental"],
    )

    from poly_matrix.poly_matrix import PolyMatrix

    basis_small = PolyMatrix(symmetric=False)
    m = 0
    for poly_mat_list in basis_dict.values():
        for mat in poly_mat_list:
            for key in mat.variable_dict_j:
                basis_small[m, key] = mat["l", key]
            m += 1

    # now actually generalize these patterns
    A_all, basis_all = lifter.get_A_learned(basis_dict, normalize=NORMALIZE)

    A_b_list_all = lifter.get_A_b_list(A_all)

    names = [f"A{i}:known" for i in range(len(A_known))]
    names += [f"A{len(A_known) + i}:learned" for i in range(len(A_all))]
    return A_b_list_all, basis_small, basis_all, names


def generate_orders(Q, A_b_list_all, xhat, qcqp_cost):
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
                "H": info["H"],
                "A": A_b_here[i][0],
                "cost": info["cost"],
                "tight": tight,
                "name": names_here[i],
            }
        )
    df_tight = pd.DataFrame(data_tight)
    return df_tight
