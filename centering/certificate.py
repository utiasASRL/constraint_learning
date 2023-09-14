import numpy as np
import scipy.linalg as la
import matplotlib

matplotlib.use("TkAgg")

from cert_tools.linalg_tools import get_nullspace
from cert_tools.eig_tools import solve_Eopt


def get_certificate(Q, A_list, x_hat):
    """
    Compute non-centered certificate.
    """
    L = np.concatenate([(A @ x_hat)[:, None] for A in A_list], axis=1)
    b = -Q @ x_hat

    null_basis, info = get_nullspace(L, method="qr")
    y, *_ = la.lstsq(L, b)

    Q_bar = Q + np.sum([Ai * yi for Ai, yi in zip(A_list, y)])
    A_list_bar = []
    for j in range(null_basis.shape[1]):
        A_bar = np.sum([Ai * ni for Ai, ni in zip(A_list, null_basis[:, j])])
        A_list_bar.append(A_bar)

    alphas, info = solve_Eopt(Q_bar, A_list_bar)
    return info["success"]


def get_centered_certificate(Q_bar, A_list):
    x_hat = np.zeros(Q_bar.shape[0])
    x_hat[0] = 1.0
    return get_certificate(Q_bar, A_list, x_hat)


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

    success = get_centered_certificate(Q, A_list, x_hat)
