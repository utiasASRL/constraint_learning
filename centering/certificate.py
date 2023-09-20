import numpy as np
import scipy.linalg as la
import matplotlib

matplotlib.use("TkAgg")

from cert_tools.linalg_tools import get_nullspace
from cert_tools.eopt_solvers import solve_Eopt_QP


def get_certificate(Q, A_list, x_hat, centered=False):
    """
    Compute non-centered certificate.
    """
    L = np.concatenate([(A @ x_hat)[:, None] for A in A_list], axis=1)
    b = -Q @ x_hat

    if centered:
        redundant_idx = np.where(np.all(L == 0, axis=0))[0]
        A_list_bar = [A_list[i] for i in redundant_idx]
    else:
        null_basis, info = get_nullspace(L, method="qr")  # (n_null x N)
        A_list_bar = []
        for j in range(null_basis.shape[0]):
            # doesn't work with sparse:
            # np.sum([Ai * ni for Ai, ni in zip(A_list, null_basis[:, j])])
            A_bar = A_list[0] * null_basis[j, 0]
            for i in range(1, null_basis.shape[1]):
                A_bar += A_list[i] * null_basis[j, i]
            A_list_bar.append(A_bar)

    y, *_ = la.lstsq(L, b)
    Q_bar = Q + np.sum([Ai * yi for Ai, yi in zip(A_list, y)])

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


def get_centered_certificate(Q, A_list, D):
    x_hat = np.zeros(Q.shape[0])
    x_hat[0] = 1.0

    Q_bar = D.T @ Q @ D
    return get_certificate(Q_bar, A_list, x_hat, centered=True)


if __name__ == "__main__":
    from lifters.poly_lifters import Poly6Lifter

    lifter = Poly6Lifter()
    Q, __ = lifter.get_Q()
    A_list = [lifter.get_A0()] + lifter.get_A_known()
    # A_b_list = lifter.get_A_b_list(A_list)

    t_init = -1
    t_hat, info, cost = lifter.local_solver(t_init)
    x_hat = lifter.get_x(t_hat)

    for A_i in A_list[1:]:
        assert abs(x_hat @ A_i @ x_hat) <= 1e-10

    success = get_certificate(Q, A_list, x_hat)

    D = lifter.get_D(t_hat)
    success = get_centered_certificate(Q, A_list, D)
