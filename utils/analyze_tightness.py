import numpy as np
import pandas as pd

from solvers.common import solve_sdp_cvxpy
from solvers.sparse import solve_lambda

# assume strong duality when gap is smaller than this
TOL_REL_GAP = 1e-3  # 1e-10  # was 1e-5

# tolerance for nullspace basis vectors
EPS_SVD = 1e-5
# tolerance for feasibility error of learned constraints
EPS_ERROR = 1e-8

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


def generate_matrices(lifter, param, fname_root="", prune=True):
    params = PARAMETER_DICT[param]
    if params["use_known"]:
        A_known = lifter.get_A_known(add_known_redundant=params["add_known_redundant"])
    else:
        A_known = []
    print(f"adding {len(A_known)} known constraints.")

    A_all, basis_full = lifter.get_A_learned(
        A_known=A_known,
        eps=EPS_SVD,
        normalize=NORMALIZE,
        method=METHOD,
        plot=False,
        incremental=params["incremental"],
    )
    errs, idxs = lifter.test_constraints(A_all, errors="print", tol=EPS_ERROR)
    print(f"found {len(idxs)} violating constraints")
    for idx in idxs[::-1]:
        del A_all[idx]
    n_learned = len(A_all) - len(A_known)
    print(f"left with {n_learned} learned constraints")

    if prune:
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
    else:
        A_b_list_all = lifter.get_A_b_list(A_all)

    names = [f"A{i}:known" for i in range(len(A_known))]
    names += [f"A{len(A_known) + i}:learned" for i in range(n_learned)]
    return A_b_list_all, basis_full, names


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
