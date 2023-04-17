import cvxpy as cp
import numpy as np

from lifters.test_tools import all_lifters


def sparsify(lifter, A_tilde, A_known=[]):
    ai_tilde_list = [lifter._get_vec(Ai_tilde)[:, None] for Ai_tilde in A_tilde]
    Ai_tilde_matrix = np.concatenate(ai_tilde_list, axis=1)
    N, n = Ai_tilde_matrix.shape
    basis_sparse = []
    for i in range(n):
        print(f"treating {i+1}/{n}")
        ai_tilde = Ai_tilde_matrix[:, i]

        alpha_i = cp.Variable(n)
        ai = Ai_tilde_matrix @ alpha_i

        constraints = [lifter._get_vec(Ai_known) @ ai == 0 for Ai_known in A_known]
        constraints += [a @ ai == 0 for a in basis_sparse]
        prob = cp.Problem(
            cp.Minimize(cp.norm_inf(ai) + 10 * cp.norm1(ai - ai_tilde)),
            constraints=constraints,
        )
        prob.solve()

        ai_hat = Ai_tilde_matrix @ alpha_i.value
        basis_sparse.append(ai_hat)
        if i > 10:
            break
    A_sparse = lifter.generate_matrices(basis_sparse)
    return A_sparse


def sparsify_naive(lifter, A_tilde):
    import scipy.sparse as sp

    ai_tilde_list = [lifter._get_vec(Ai_tilde)[:, None] for Ai_tilde in A_tilde]
    basis = np.concatenate(ai_tilde_list, axis=1)
    basis_sparse = sp.csr_array(basis.shape)
    for i in range(basis.shape[0]):
        ai = basis[[i], :]
        # TODO(FD) below might be easier by rounding
        tol = 1e-1
        ai[np.abs(ai - 1.0) < tol] = 1.0
        ai[np.abs(ai + 1.0) < tol] = -1.0
        ai[np.abs(ai - 0.5) < tol] = 0.5
        ai[np.abs(ai + 0.5) < tol] = -0.5
        ai[np.abs(ai) < tol] = 0.0
        basis_sparse[i, :] = ai
    A_sparse = lifter.generate_matrices(basis_sparse)
    return A_sparse


def test_literal():
    from lifters.plotting_tools import plot_matrices

    for lifter in all_lifters():
        A_known = lifter.get_A_known()
        A_learned = lifter.get_A_learned(
            factor=2, eps=1e-7, method="qrp"  # , A_known=A_known
        )
        A_learned_naive = sparsify_naive(lifter, A_learned)
        A_learned_sparse = sparsify(lifter, A_learned, A_known)

        fig, ax = plot_matrices(A_known[:10])
        ax.set_title("known")
        fig, ax = plot_matrices(A_learned[:10])
        ax.set_title("learned")
        fig, ax = plot_matrices(A_learned_sparse[:10])
        ax.set_title("sparse optimization")
        fig, ax = plot_matrices(A_learned_naive[:10])
        ax.set_title("sparse naive")
        lifter.print_constraints(A_known)


if __name__ == "__main__":
    test_literal()
