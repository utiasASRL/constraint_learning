import pickle

import numpy as np
import scipy.linalg as la
import matplotlib

matplotlib.use("TkAgg")

from cert_tools.linalg_tools import get_nullspace
from cert_tools.eopt_solvers_qp import solve_Eopt_QP


def get_certificate(Q, Constraints, x_hat, centered=False):
    """
    Compute non-centered certificate.
    """
    L = np.concatenate([(A @ x_hat)[:, None] for A, b in Constraints], axis=1)
    b = -Q @ x_hat

    if centered:
        redundant_idx = np.where(np.all(L == 0, axis=0))[0]
        A_list_bar = [Constraints[i][0] for i in redundant_idx]
    else:
        null_basis, info = get_nullspace(L, method="qr")  # (n_null x N)
        A_list_bar = []
        for j in range(null_basis.shape[0]):
            # doesn't work with sparse:
            # np.sum([Ai * ni for Ai, ni in zip(A_list, null_basis[:, j])])
            A_bar = Constraints[0][0] * null_basis[j, 0]
            for i in range(1, null_basis.shape[1]):
                A_bar += Constraints[i][0] * null_basis[j, i]
            A_list_bar.append(A_bar)

    y, *_ = la.lstsq(L, b)
    Q_bar = Q + np.sum([Ab[0] * yi for Ab, yi in zip(Constraints, y)])

    x_init = np.zeros(len(A_list_bar))

    if centered:
        np.testing.assert_allclose(Q_bar[0, :], 0.0, atol=1e-8)
        Q_bar = Q_bar[1:, 1:]
        A_list_bar = [Ai[1:, 1:] for Ai in A_list_bar]
        alphas, info = solve_Eopt_QP(
            Q_bar, A_list_bar, x_init, verbose=2, lmin=True, l_threshold=1e-5
        )
    else:
        alphas, info = solve_Eopt_QP(
            Q_bar, A_list_bar, x_init, verbose=2, lmin=True, l_threshold=-1e-15
        )
    return info["success"]


if __name__ == "__main__":
    from examples.poly6_lifter import get_problem

    print("Original certificate...")
    prob = get_problem()
    success = get_certificate(prob["C"], prob["Constraints"], prob["x_cand"])

    print("Centered certificate...")
    prob = get_problem(centered=True)
    success = get_certificate(
        prob["C"], prob["Constraints"], prob["x_cand"], centered=True
    )
