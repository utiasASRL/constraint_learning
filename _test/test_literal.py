import itertools

import cvxpy as cp
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from lifters.plotting_tools import plot_matrices
from lifters.test_tools import all_lifters
from poly_matrix.poly_matrix import PolyMatrix


def sparsify(lifter, A_tilde, A_known=[]):
    ai_tilde_list = [lifter._get_vec(Ai_tilde).reshape((-1, 1)) for Ai_tilde in A_tilde]
    Ai_tilde_matrix = np.concatenate(ai_tilde_list, axis=1)
    N, n = Ai_tilde_matrix.shape
    basis_sparse = []
    for i in range(n):
        print(f"treating {i+1}/{n}")
        ai_tilde = Ai_tilde_matrix[:, i]  # N x 1

        alpha_i = cp.Variable((n, 1))
        ai = Ai_tilde_matrix @ alpha_i  # N x 1

        constraints = [lifter._get_vec(Ai_known) @ ai == 0 for Ai_known in A_known]
        constraints += [a.T @ ai == 0 for a in basis_sparse]
        prob = cp.Problem(
            cp.Minimize(cp.norm_inf(ai) + 10 * cp.norm1(ai - ai_tilde)),
            constraints=constraints,
        )
        prob.solve()

        ai_hat = Ai_tilde_matrix @ alpha_i.value
        basis_sparse.append(np.array(ai_hat))
        if i > 10:
            break
    A_sparse = lifter.generate_matrices(basis_sparse)
    return A_sparse


def sparsify_naive(lifter, Y, A_tilde, frac=8, tol=1e-5, axs=None):
    from copy import deepcopy

    import scipy.sparse as sp

    plt.show()
    A_sparse = []
    success = 0
    for A in A_tilde:
        A_new = deepcopy(A)
        A_new.data = (A.data * frac).round() / frac
        error = Y @ lifter._get_vec(A_new).T
        if axs is not None:
            axs[0].matshow(A.toarray())
            axs[1].matshow(A_new.toarray())
        if np.max(error) < tol:
            if np.allclose(A_new.data, A.data):
                print("didn't change anything")
            else:
                success += 1
            if len(A_sparse):
                assert A_new.shape == A_sparse[-1].shape
            A_sparse.append(A_new)
        else:
            if len(A_sparse):
                assert A.shape == A_sparse[-1].shape
            A_sparse.append(A)
    print(f"quantized {success}/{len(A_tilde)}")
    return A_sparse


def test_interpret():
    for lifter in all_lifters():
        A_known = lifter.get_A_known()

        data = []
        for A in A_known:
            A_poly = PolyMatrix()
            A_poly.init_from_sparse(A, lifter.var_dict)
            data.append(A_poly.interpret(lifter.var_dict))
        df = pd.DataFrame(data)
        print(df)


def test_sparsify():
    for lifter in all_lifters():
        A_known = lifter.get_A_known()
        print(len(A_known))
        lifter.test_constraints(A_known)

        Y = lifter.generate_Y()
        A_learned = lifter.get_A_learned(
            factor=2, eps=1e-7, method="qrp", Y=Y, A_known=A_known
        )
        print(len(A_learned))
        # fig, axs = plt.subplots(1, 2)
        A_learned_naive = sparsify_naive(lifter, Y=Y, A_tilde=A_learned, axs=None)
        A_learned_sparse = sparsify(lifter, A_learned, A_known)

        fig, ax = plot_matrices(A_known[:10], colorbar=False)
        fig.suptitle("known")
        fig, ax = plot_matrices(A_learned[:10], colorbar=False)
        fig.suptitle("learned")
        fig, ax = plot_matrices(A_learned_sparse[:10], colorbar=False)
        fig.suptitle("sparse optimization")
        fig, ax = plot_matrices(A_learned_naive[:10], colorbar=False)
        fig.suptitle("sparse naive")
        plt.show()


if __name__ == "__main__":
    test_interpret()
    # test_sparsify()
