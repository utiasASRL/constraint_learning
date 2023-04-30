from abc import ABC, abstractmethod

import matplotlib.pylab as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

EPS = 1e-10  # threshold for nullspace (on eigenvalues)

# basis pursuit method, can be
# - qr: qr decomposition
# - qrp: qr decomposition with permutations (sparser)
# - svd: svd
METHOD = "qrp"

NORMALIZE = False  # normalize learned Ai or not, (True)
FACTOR = 2.0  # how much to oversample (>= 1)


class StateLifter(ABC):
    def __init__(self, theta_shape, M, L=0):
        self.theta_shape = theta_shape
        if len(theta_shape) > 1:
            self.N_ = np.multiply(*theta_shape)
        else:
            self.N_ = theta_shape[0]
        self.M_ = M
        self.L = L
        self.setup = None

        # fixing seed for testing purposes
        np.random.seed(1)
        self.generate_random_setup()

    # the property decorator creates by default read-only attributes.
    # can create a @dim_x.setter function to make it also writeable.
    @property
    def dim_x(self):
        return 1 + self.N + self.M + self.L

    @property
    def d(self):
        return self.d_

    @d.setter
    def d(self, var):
        assert var in [1, 2, 3]
        self.d_ = var

    @property
    def N(self):
        return self.N_

    @N.setter
    def N(self, var):
        self.N_ = var

    @property
    def M(self):
        return self.M_

    def get_Q(self, noise=1e-3):
        print("Warning: get_Q not implemented")
        return None, None

    @abstractmethod
    def generate_random_setup(self):
        return

    @abstractmethod
    def sample_theta(self):
        return

    @abstractmethod
    def get_x(self, theta) -> np.ndarray:
        return

    def get_vec(self, mat):
        """Convert NxN Symmetric matrix to (N+1)N/2 vectorized version that preserves inner product.

        :param mat: (spmatrix or ndarray) symmetric matrix
        :return: ndarray
        """
        from copy import deepcopy

        mat = deepcopy(mat)
        if isinstance(mat, sp.spmatrix):
            ii, jj = mat.nonzero()
            mat[ii, jj] *= np.sqrt(2.0)
            diag = ii == jj
            mat[ii[diag], jj[diag]] /= np.sqrt(2)
        else:
            mat *= np.sqrt(2.0)
            mat[range(mat.shape[0]), range(mat.shape[0])] /= np.sqrt(2)
        return np.array(mat[np.triu_indices(n=mat.shape[0])]).flatten()

        def sub2ind(rows, cols):
            return rows * (mat.shape[0] - np.cumsum(rows)) + cols

        vec_shape = int(mat.shape[0] * (mat.shape[0] + 1) / 2)

        if isinstance(mat, sp.spmatrix):
            ii, jj = mat.nonzero()
        elif isinstance(mat, np.ndarray):
            ii, jj = np.where(mat != 0)

        # multiply all elements by sqrt(2)
        vec_nnz = np.array(mat[ii, jj]).flatten() * np.sqrt(2)
        # undo operation for diagonal
        vec_nnz[ii == jj] /= np.sqrt(2)
        vec = np.zeros(vec_shape)
        vec[sub2ind(ii, jj)] = vec_nnz
        return vec

    def get_mat(self, vec, sparse=False, trunc_tol=1e-8):
        """Convert (N+1)N/2 vectorized matrix to NxN Symmetric matrix in a way that preserves inner products.

        In particular, this means that we divide the off-diagonal elements by sqrt(2).

        :param vec (ndarray): vector of upper-diagonal elements
        :return: symmetric matrix filled with vec.
        """
        triu = np.triu_indices(n=self.dim_x)
        mask = np.abs(vec) > trunc_tol
        triu_i_nnz = triu[0][mask]
        triu_j_nnz = triu[1][mask]
        vec_nnz = vec[mask]
        if sparse:
            # divide off-diagonal elements by sqrt(2)
            offdiag = triu_i_nnz != triu_j_nnz
            diag = triu_i_nnz == triu_j_nnz
            triu_i = triu_i_nnz[offdiag]
            triu_j = triu_j_nnz[offdiag]
            diag_i = triu_i_nnz[diag]
            vec_nnz_off = vec_nnz[offdiag] / np.sqrt(2)
            vec_nnz_diag = vec_nnz[diag]
            Ai = sp.csr_array(
                (
                    np.r_[vec_nnz_diag, vec_nnz_off, vec_nnz_off],
                    (np.r_[diag_i, triu_i, triu_j], np.r_[diag_i, triu_j, triu_i]),
                ),
                (self.dim_x, self.dim_x),
            )
        else:
            Ai = np.zeros((self.dim_x, self.dim_x))

            # divide all elements by sqrt(2)
            Ai[triu_i_nnz, triu_j_nnz] = vec_nnz / np.sqrt(2)
            Ai[triu_j_nnz, triu_i_nnz] = vec_nnz / np.sqrt(2)

            # undo operation for diagonal
            Ai[range(self.dim_x), range(self.dim_x)] *= np.sqrt(2)
        return Ai

    def get_A_known(self) -> list:
        return []

    def get_A_learned(
        self,
        factor=FACTOR,
        eps=EPS,
        method=METHOD,
        A_known=[],
        plot=False,
        Y=None,
        return_S=False,
        normalize=NORMALIZE,
    ) -> list:
        if Y is None:
            Y = self.generate_Y(factor=factor)

        S_known = [0] * len(A_known)

        basis, S = self.get_basis(Y, method=method, eps=eps, A_list=A_known)
        corank = basis.shape[0] - len(A_known)
        try:
            assert abs(S[-corank]) / eps < 1e-1  # 1e-1  1e-10
            assert abs(S[-corank - 1]) / eps > 10  # 1e-11 1e-10
        except:
            print(f"there might be a problem with the chosen threshold {eps}:")
            print(S[-corank], eps, S[-corank - 1])

        if plot:
            from lifters.plotting_tools import plot_singular_values

            plot_singular_values(S, eps=eps)

        if return_S:
            A_learned = self.generate_matrices(basis, normalize=normalize)
            if corank > 0:
                return A_learned, S_known + list(np.abs(S[-corank:]))
            else:
                # avoid S[-0] which gives unexpected result.
                return A_learned, S_known
        else:
            return self.generate_matrices(basis, normalize=normalize)

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around groudn truth.

        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def generate_Y(self, factor=3, ax=None):
        dim_Y = int(self.dim_x * (self.dim_x + 1) / 2)

        # need at least dim_Y different random setups
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = self.sample_theta()
            if seed < 10 and ax is not None:
                if np.ndim(self.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = self.get_x(theta)
            X = np.outer(x, x)

            Y[seed, :] = self.get_vec(X)
        return Y

    def get_basis(self, Y, A_list: list = [], eps=EPS, method=METHOD):
        """Generate basis from lifted state matrix Y.

        :param A_list: if given, will generate basis that is orthogonal to these given constraints.

        :return: basis with A_list as first elements (if given)
        """
        # if there is a known list of constraints, add them to the Y so that resulting nullspace is orthogonal to them
        if len(A_list) > 0:
            A = np.vstack([self.get_vec(a) for a in A_list])
            Y = np.vstack([Y, A])

        if method != "qrp":
            print("using a method other than qrp is not recommended.")

        if method == "svd":
            U, S, Vh = np.linalg.svd(
                Y
            )  # nullspace of Y is in last columns of V / last rows of Vh
            rank = np.sum(np.abs(S) > eps)
            basis = Vh[rank:, :]

            # test that it is indeed a null space
            np.testing.assert_allclose(Y @ basis.T, 0.0, atol=1e-5)
        elif method == "qr":
            # if Y.T = QR, the last n-r columns
            # of R make up the nullspace of Y.
            Q, R = np.linalg.qr(Y.T)
            S = np.abs(np.diag(R))
            sorted_idx = np.argsort(S)[::-1]
            S = S[sorted_idx]
            rank = np.where(S < eps)[0][0]
            # decreasing order
            basis = Q[:, sorted_idx[rank:]].T
        elif method == "qrp":
            # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.
            Q, R, p = la.qr(Y, pivoting=True, mode="economic")
            S = np.abs(np.diag(R))
            rank = np.sum(S > eps)
            R1, R2 = R[:rank, :rank], R[:rank, rank:]
            N = np.vstack([la.solve_triangular(R1, R2), -np.eye(R2.shape[1])])
            basis = np.zeros(N.T.shape)
            basis[:, p] = N.T
        else:
            raise ValueError(method)
        # Add known constraints to basis
        if len(A_list) > 0:
            basis = np.vstack([A, basis])

        return basis, S

    def generate_matrices(
        self,
        basis,
        normalize=NORMALIZE,
        sparse=True,
        trunc_tol=1e-10,
    ):
        """
        generate matrices from vectors
        """

        A_list = []
        for i in range(len(basis)):
            Ai = self.get_mat(basis[i], sparse=sparse, trunc_tol=trunc_tol)
            # Normalize the matrix
            if normalize and not sparse:
                # Ai /= np.max(np.abs(Ai))
                Ai /= np.linalg.norm(Ai, p=2)
            elif normalize and sparse:
                Ai /= sp.linalg.norm(Ai, ord="fro")
            # Sparsify and truncate
            if sparse:
                Ai.eliminate_zeros()
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0
            # add to list
            A_list.append(Ai)
        return A_list

    def test_constraints(self, A_list, tol: float = 1e-7, errors: str = "raise"):
        """
        :param errors: "raise" or "print" detected violations.
        """
        max_violation = -np.inf
        j_bad = []
        for j, A in enumerate(A_list):
            t = self.sample_theta()
            x = self.get_x(t)

            constraint_violation = abs(x.T @ A @ x)
            max_violation = max(max_violation, constraint_violation)
            if constraint_violation > tol:
                msg = f"big violation at {j}: {constraint_violation:.1e}"
                j_bad.append(j)
                if errors == "raise":
                    raise ValueError(msg)
                elif errors == "print":
                    print(msg)
                else:
                    raise ValueError(errors)
        return max_violation, j_bad

    def get_A0(self):
        from poly_matrix.poly_matrix import PolyMatrix

        A0 = PolyMatrix()
        A0["l", "l"] = 1.0
        return A0.get_matrix(self.var_dict)

    def get_A_b_list(self, A_list):
        return [(self.get_A0(), 1.0)] + [(A, 0.0) for A in A_list]

    def run(self, n_dual: int = 3, plot: bool = False, noise: float = 1e-4):
        """Convenience function to quickly test and debug lifter

        :param n_dual

        """
        import matplotlib.pylab as plt

        from lifters.plotting_tools import plot_matrices
        from solvers.common import find_local_minimum, solve_dual, solve_sdp_cvxpy

        Q, y = self.get_Q(noise=noise)

        print("find local minimum...")
        that, local_cost = find_local_minimum(
            self, y, delta=noise, n_inits=1, plot=False, verbose=True
        )

        A_known = self.get_A_known()
        A_list = self.get_A_learned(eps=EPS, method="qrp", plot=plot, A_known=A_known)
        print(
            f"number of constraints: known {len(A_known)}, redundant {len(A_list) - len(A_known)}, total {len(A_list)}"
        )

        max_error, j_bad = self.test_constraints(A_list, errors="print")
        for i, j in enumerate(j_bad):
            del A_list[j - i]

        # just a sanity check
        if len(j_bad):
            max_error, j_bad = self.test_constraints(A_list, errors="print")

        A_b_list = self.get_A_b_list(A_list)

        # check that learned constraints meat tolerance

        if plot:
            plot_matrices(A_list, n_matrices=5, start_idx=0)

        tol = 1e-10
        print("solve dual problems...")
        dual_costs = []
        # primal_costs = []
        n = min(n_dual, len(A_list))
        n_constraints = range(len(A_list) - n + 1, len(A_list) + 1)
        for i in n_constraints:
            print(f"{i}/{len(A_list)}")

            X, cost = solve_sdp_cvxpy(Q, A_b_list, primal=False, tol=tol, verbose=False)
            dual_costs.append(cost)

            # X, cost = solve_sdp_cvxpy(Q, A_b_list, primal=True, tol=tol)
            # primal_costs.append(cost)

            # cost, H, status = solve_dual(Q, A_list[:i])
            # dual_costs.append(cost)

        E, V = np.linalg.eigh(X)
        xhat = V[:, -1] * np.sqrt(E[-1])
        if abs(xhat[0] + 1) < 1e-10:  # xhat is close to -1
            xhat = -xhat
        elif abs(xhat[0] - 1) < 1e-10:
            pass
        else:
            ValueError("{xhat[0]:.4f} not 1!")
        theta = xhat[1 : self.N + 1]
        if E[-1] / E[-2] < 1e5:
            print(f"not rank 1! {E}")
        else:
            assert abs((self.get_cost(theta, y) - cost)) < 1e-7
        print(
            f"dual costs: {[np.format_float_scientific(d, precision=4) for d in dual_costs]}"
        )
        print(f"local cost: {np.format_float_scientific(local_cost, precision=4)}")

        if plot:
            fig, ax = plt.subplots()
            p_gt, a_gt = self.get_positions_and_landmarks(self.theta)
            ax.scatter(*p_gt.T, color="C0")
            ax.scatter(*a_gt.T, color="C0", marker="x")
            for i, (t, name) in enumerate(zip([theta, that], ["primal", "qcqp"])):
                cost = self.get_cost(t, y)
                p, a = self.get_positions_and_landmarks(t)
                ax.scatter(
                    *p.T, color=f"C{i+1}", marker="*", label=f"{name}, {cost:.1e}"
                )
                ax.scatter(
                    *a.T,
                    color=f"C{i+1}",
                    marker="+",
                )
                # 2 x K x d
                a_lines = np.concatenate([a_gt[None, :], a[None, :]], axis=0)
                ax.plot(a_lines[:, :, 0], a_lines[:, :, 1], color=f"C{i+1}")
                p_lines = np.concatenate([p_gt[None, :], p[None, :]], axis=0)
                ax.plot(p_lines[:, :, 0], p_lines[:, :, 1], color=f"C{i+1}")
            ax.legend(loc="best")

            plt.figure()
            plt.axhline(local_cost, label=f"QCQP cost {local_cost:.2e}")
            plt.scatter(n_constraints, dual_costs, label="dual costs")
            plt.xlabel("added constraints")
            plt.ylabel("cost")
            plt.legend()
            plt.show()
            plt.pause(1)
