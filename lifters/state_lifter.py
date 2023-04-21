from abc import abstractmethod

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


class StateLifter(object):
    def __init__(self, theta_shape, M, L=0):
        self.theta_shape = theta_shape
        if len(theta_shape) > 1:
            self.N = np.multiply(*theta_shape)
        else:
            self.N = theta_shape[0]
        self.M = M
        self.L = L
        self.setup = None

        # fixing seed for testing purposes
        np.random.seed(1)
        self.generate_random_setup()
        self.generate_random_unknowns()

    def generate_random_setup(self):
        print("Warning: nothing to setup")

    def generate_random_unknowns(self):
        raise NotImplementedError()

    def get_Q(self, noise=1e-3):
        print("Warning: get_Q not implemented")
        return None, None

    @abstractmethod
    def get_x(self, theta):
        raise NotImplementedError()

    def get_A_known(self):
        return []

    def get_vec(self, mat):
        """Convert NxN Symmetric matrix to (N+1)N/2 vectorized version that preserves inner product.

        Args:
            mat (spmatrix or ndarray): symmetric matrix

        Returns:
            ndarray
        """
        # Multiply off-diagonals by sqrt(2)
        mat *= np.sqrt(2)
        if isinstance(mat, sp.spmatrix):
            mat.setdiag(mat.diagonal() / np.sqrt(2))
        elif isinstance(mat, np.ndarray):
            ind = range(mat.shape[0])
            mat[ind, ind] = np.diag(mat) / np.sqrt(2)
        return mat[np.triu_indices(n=self.dim_X())].flatten()

    def get_mat(self, vec):
        """Convert (N+1)N/2 vectorized matrix to NxN Symmetric matrix in a way that preserves inner products.

        Args:
            vec (ndarray): symmetric matrix

        Returns:
            ndarray
        """
        Ai = np.zeros((self.dim_X(), self.dim_X()))
        triu = np.triu_indices(n=self.dim_X())
        Ai[triu] = vec
        Ai += Ai.T
        # Divide the off diagonal by root 2
        Ai /= np.sqrt(2)
        ind = range(self.dim_X())
        Ai[ind, ind] = np.diag(Ai) / np.sqrt(2)
        return Ai

    def get_vec_around_gt(self, delta=0):
        if type(self.unknowns) == np.ndarray:
            return self.unknowns + np.random.normal(
                size=self.unknowns.shape, scale=delta
            )
        elif type(self.unknowns) == tuple:
            vec = np.concatenate([*self.unknowns])
            return vec + np.random.normal(size=vec.shape, scale=delta)
            # return tuple(
            #    u + np.random.normal(size=u.shape, scale=delta) for u in self.unknowns
            # )

    def generate_Y(self, factor=3, ax=None):
        dim_X = self.dim_X()
        dim_Y = int(dim_X * (dim_X + 1) / 2)

        # need at least dim_Y different random setups
        n_seeds = dim_Y * factor
        Y = np.empty((n_seeds, dim_Y))
        self.generate_random_setup()
        for seed in range(n_seeds):
            np.random.seed(seed)
            self.generate_random_unknowns()
            theta = self.get_theta()
            if seed < 10 and ax is not None:
                if np.ndim(theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = self.get_x(theta)
            X = np.outer(x, x)

            Y[seed, :] = self.get_vec(X)

        return Y

    def get_basis(self, Y, A_list: list = [], eps=1e-10, method="qrp"):
        """
        generate basis from lifted state matrix Y
        """
        # if there is a known list of constraints, add them to the Y so that resulting nullspace is orthogonal to them
        if len(A_list) > 0:
            A = np.vstack([self.get_vec(a) for a in A_list])
            Y = np.vstack([Y, A])

        if method == "svd":
            U, S, Vh = np.linalg.svd(
                Y
            )  # nullspace of Y is in last columns of V / last rows of Vh
            rank = np.sum(np.abs(S) > eps)
            basis = Vh[rank:, :]
            # test that it is indeed a null space
            np.testing.assert_allclose(Y @ basis.T, 0.0, atol=1e-5)
        elif method == "qr":
            Q, R = np.linalg.qr(Y.T)
            S = np.diag(R)
            rank = np.sum(np.abs(S) > eps)
            basis = Q[:, rank:].T
        elif method == "qrp":
            # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.
            Q, R, p = la.qr(Y, pivoting=True, mode="economic")
            S = np.diag(R)
            rank = np.sum(np.abs(S) > eps)
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

    def dim_X(self):
        return 1 + self.N + self.M + self.L

    def generate_matrices(self, basis, normalize=True, sparse=True, trunc_tol=1e-10):
        """
        generate matrices from vectors
        """

        A_list = []
        vmax = -np.inf
        vmin = np.inf
        for i in range(basis.shape[0]):
            Ai = self.get_mat(basis[i, :])
            # Normalize the matrix
            if normalize:
                Ai /= np.max(np.abs(Ai))
            # Sparsify and truncate
            if sparse:
                Ai = sp.csr_array(Ai)
                Ai.data[np.abs(Ai.data) < trunc_tol] = 0.0
                Ai.eliminate_zeros()
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0
            # add to list
            A_list.append(Ai)

            vmax = max(vmax, np.max(Ai))
            vmin = min(vmin, np.min(Ai))

        return A_list
