from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from poly_matrix import PolyMatrix

EPS = 1e-10  # threshold for nullspace (on eigenvalues)

# basis pursuit method, can be
# - qr: qr decomposition
# - qrp: qr decomposition with permutations (sparser)
# - svd: svd
METHOD = "qrp"

NORMALIZE = False  # normalize learned Ai or not, (True)
FACTOR = 2.0  # how much to oversample (>= 1)

# maximum number of elements to use when doing incremental learning.
# we start with all possible pairs of variables (k=2) and go up to
# k=MAX_N_SUBSETS.
MAX_N_SUBSETS = 3


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
    def get_x(self, theta, var_subset=None) -> np.ndarray:
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

    def get_mat(self, vec, sparse=False, trunc_tol=1e-8, var_dict=None):
        """Convert (N+1)N/2 vectorized matrix to NxN Symmetric matrix in a way that preserves inner products.

        In particular, this means that we divide the off-diagonal elements by sqrt(2).

        :param vec (ndarray): vector of upper-diagonal elements
        :return: symmetric matrix filled with vec.
        """
        # len(vec) = k = n(n+1)/2 -> dim_x = n =
        dim_x = int(0.5 * (-1 + np.sqrt(1 + 8 * len(vec))))
        if var_dict is None:
            assert dim_x == self.dim_x

        triu = np.triu_indices(n=dim_x)
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
                (dim_x, dim_x),
            )
        else:
            Ai = np.zeros((dim_x, dim_x))

            # divide all elements by sqrt(2)
            Ai[triu_i_nnz, triu_j_nnz] = vec_nnz / np.sqrt(2)
            Ai[triu_j_nnz, triu_i_nnz] = vec_nnz / np.sqrt(2)

            # undo operation for diagonal
            Ai[range(dim_x), range(dim_x)] *= np.sqrt(2)

        if var_dict is None:
            return Ai
        # if var_dict is not None, then Ai corresponds to the subblock
        # defined by var_dict, of the full constraint matrix.
        Ai_poly, __ = PolyMatrix.init_from_sparse(Ai, var_dict)
        return Ai_poly.get_matrix(self.var_dict)

    def get_A_known(self) -> list:
        return []

    def extract_A_known(self, A_known, var_subset):
        sub_A_known = []
        for A in A_known:
            A_poly, var_dict = PolyMatrix.init_from_sparse(
                A, self.var_dict, symmetric=True
            )

            assert len(A_poly.get_variables()) > 0

            # if all of the non-zero elements of A_poly are in var_subset,
            # we can use this matrix.
            if np.all([v in var_subset for v in A_poly.get_variables()]):
                sub_A_known.append(
                    A_poly.get_matrix(
                        {k: v for k, v in var_dict.items() if k in var_subset}
                    )
                )
        return sub_A_known

    def get_augmented_dict(self, var_subset=None):
        import itertools

        if var_subset is None:
            var_subset = self.var_dict.keys()
        vectorized_var_list = list(
            itertools.combinations_with_replacement(var_subset, 2)
        )
        product_dict = {}
        for p in range(self.get_dim_P()):
            for zi, zj in vectorized_var_list:
                # for diagonal, we will only have half of the matrix
                if zi == zj:
                    size = int(self.var_dict[zi] * (self.var_dict[zj] + 1) / 2)
                else:
                    size = self.var_dict[zi] * self.var_dict[zj]
                param_name = "l" if p == 0 else f"p_{p-1}"
                product_dict[f"{param_name}.{zi}.{zj}"] = size
        return product_dict

    def get_A_learned(
        self,
        eps: float = EPS,
        A_known: list = [],
        plot: bool = False,
        Y: np.ndarray = None,
        factor: int = FACTOR,
        method: str = METHOD,
        normalize: bool = NORMALIZE,
        incremental: bool = False,
    ) -> list:
        """
        Learn the constraint matrices.

        :param A_known: if given, learn constraints that are orthogonal to these known
                        constraints.

        :return: list of learned matrices, nullspace basis, and S if return_S is True,
        """
        import itertools

        if incremental:
            if Y is not None:
                raise NotImplementedError(
                    "can't do incremental A learning with fixed Y yet."
                )

            # var_dict is always composed of
            # - "l": homogenization
            # - "x": fixed-size parameters
            # - "z_i", ..., "z_N": variable-size parameters
            var_subsets = []
            for k in range(MAX_N_SUBSETS):
                var_subsets.append(tuple(["l", "x"] + [f"z_{i}" for i in range(k)]))
            # var_subsets = list(itertools.combinations(self.var_dict.keys(), 2))
            # for k in range(3, MAX_N_SUBSETS + 1):
            #    var_subsets += list(itertools.combinations(self.var_dict.keys(), k))
        else:
            var_subsets = [list(self.var_dict.keys())]

        A_learned = []
        A_learned += A_known

        # keep track of current set of lin. independent constraints
        dim_Y = self.get_dim_X(var_subsets[-1]) * self.get_dim_P()

        if len(A_known):
            if incremental:
                raise ValueError(
                    "don't use A_known with incremental! this leads to worse performance"
                )
            basis_learned = np.vstack(
                [self.get_augmented_vec(self.get_vec(A)) for A in A_known]
            )
            assert basis_learned.shape[1] == dim_Y
            current_rank = np.linalg.matrix_rank(basis_learned)
        else:
            basis_learned = np.empty((0, dim_Y))
            current_rank = 0

        basis_dict = {}

        all_product_dict = self.get_augmented_dict(var_subset=var_subsets[-1])
        for var_subset in var_subsets:
            basis_dict[var_subset] = []
            # var_dict = {key: self.var_dict[key] for key in var_subset}

            Y = self.generate_Y(factor=factor, var_subset=var_subset)

            # extract subset of known matrices given the current variables
            sub_A_known = self.extract_A_known(A_known, var_subset)

            # TODO(FD) can we enforce lin. independance to previously found
            # matrices at this point?
            basis_new, S = self.get_basis(
                Y,
                method=method,
                eps=eps,
                A_known=sub_A_known,  # basis_known=basis_all
            )
            corank = basis_new.shape[0]

            if corank == 0:
                print(f"{var_subset}: no new learned matrices found")
                continue

            print(f"{var_subset}: {corank} learned matrices found")

            if corank > 1:
                try:
                    assert abs(S[-corank]) / eps < 1e-1  # 1e-1  1e-10
                    assert abs(S[-corank - 1]) / eps > 10  # 1e-11 1e-10
                except:
                    print(f"there might be a problem with the chosen threshold {eps}:")
                    print(S[-corank], eps, S[-corank - 1])

            if plot:
                from lifters.plotting_tools import plot_singular_values

                plot_singular_values(S, eps=eps)

            # find out which of the constraints are linearly dependant of the others.
            # TODO(FD): could potentially do below with a QRP decomposition.
            for i, bi_sub in enumerate(basis_new):
                # get the variable pairs that bi_sub corresponds to.
                sub_product_dict = self.get_augmented_dict(var_subset)
                assert sum([size for size in sub_product_dict.values()]) == len(bi_sub)

                bi_poly = PolyMatrix(symmetric=False)
                j = 0
                for key, size in sub_product_dict.items():
                    val = bi_sub[j : j + size]
                    if np.any(np.abs(val) > 1e-10):
                        bi_poly["l", key] = val[None, :]
                    j += size

                # created zero-padded bi
                bi = bi_poly.get_vector_dense(all_product_dict, i="l")[None, :]

                basis_learned_test = np.vstack([basis_learned, bi])
                new_rank = np.linalg.matrix_rank(basis_learned_test, tol=1e-10)
                if new_rank == current_rank + 1:
                    print(f"b{i} is valid basis vector.")
                    basis_dict[var_subset].append(bi_poly)
                    basis_learned = basis_learned_test
                    current_rank += 1
                elif new_rank == current_rank:
                    print(f"b{i} is linearly dependent.")
                else:
                    print(
                        f"Warning: invalid rank change from rank {current_rank} to {new_rank}"
                    )

        # given the learned matrices we need to generalize to all landmarks!
        basis_poly = PolyMatrix(symmetric=False)
        # ex: (l, x, z_0), [B_i]
        # for var_subset, bi_poly_list in basis_dict.items():
        var_subset = ("l", "x", "z_0")
        bi_poly_list = basis_dict[var_subset]
        m = 0
        # for each found constraint....
        for bi_poly in bi_poly_list:
            # if z_0 is in this constraint, repeat the constraint for each landmark.
            if np.any(["z_0" in key for key in bi_poly.variable_dict_j]):
                for i in range(self.n_landmarks):
                    for key in bi_poly.variable_dict_j:
                        key_i = key.replace("z_0", f"z_{i}").replace("p_0", f"p_{i}")
                        basis_poly[m, key_i] = bi_poly["l", key]
                    m += 1
            else:
                for key in bi_poly.variable_dict_j:
                    basis_poly[m, key] = bi_poly["l", key]
                m += 1

        var_subset = ("l", "x", "z_0", "z_1")
        bi_poly_list = basis_dict[var_subset]
        for bi_poly in bi_poly_list:
            # if z_0 and z_1 are in this constraint, repeat the constraint for each landmark.
            if np.any(
                [("z_0" in key) or ("z_1" in key) for key in bi_poly.variable_dict_j]
            ):
                for i, j in itertools.combinations(range(self.n_landmarks), 2):
                    print(f"should map 0 to {i} and 1 to {j}")
                    assert i != j
                    for key in bi_poly.variable_dict_j:
                        # need intermediate variables cause otherwise z_0 -> z_1 -> z_2 etc. can happen.
                        key_ij = key.replace("z_0", f"zi_{i}").replace("z_1", f"zi_{j}")
                        key_ij = key_ij.replace("p_0", f"pi_{i}").replace(
                            "p_1", f"pi_{j}"
                        )
                        key_ij = key_ij.replace("zi", "z").replace("pi", "p")
                        print("changed", key, "to", key_ij)
                        basis_poly[m, key_ij] = bi_poly["l", key]
                    m += 1
            else:  # below should actually never happen.
                for key in bi_poly.variable_dict_j:
                    basis_poly[m, key] = bi_poly["l", key]
                m += 1

        all_dict = self.get_augmented_dict()
        basis_learned = basis_poly.get_matrix((list(range(m)), all_dict))
        A_learned = self.generate_matrices(basis_learned, normalize=normalize)
        return A_learned, basis_learned, basis_poly

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around groudn truth.

        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def get_dim_x(self, var_subset=None):
        if var_subset is None:
            return self.get_dim_x(var_subset=self.var_dict)
        return sum([val for key, val in self.var_dict.items() if (key in var_subset)])

    def sample_parameters(self):
        """Default behavior: has no effect. Can add things like landmark coordintaes here, to learn dependencies."""
        return [1.0]

    def get_parameters(self):
        if self.parameters is None:
            self.parameters = self.sample_parameters()
        return self.parameters

    def get_dim_X(self, var_subset=None):
        dim_x = self.get_dim_x(var_subset)
        return int(dim_x * (dim_x + 1) / 2)

    def get_dim_P(self):
        return len(self.get_parameters())

    def generate_Y(self, factor=3, ax=None, var_subset=None):
        dim_X = self.get_dim_X(var_subset=var_subset)
        dim_P = self.get_dim_P()

        # need at least dim_Y different random setups
        dim_Y = int(dim_X * dim_P)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = self.sample_theta()
            parameters = self.sample_parameters()

            if seed < 10 and ax is not None:
                if np.ndim(self.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = self.get_x(theta, parameters, var_subset=var_subset)
            X = np.outer(x, x)

            # generates [1*x, a1*x, ..., aK*x]
            Y[seed, :] = np.kron(parameters, self.get_vec(X))
        return Y

    def get_basis(
        self,
        Y,
        A_known: list = [],
        basis_known: np.ndarray = None,
        eps=EPS,
        method=METHOD,
    ):
        """Generate basis from lifted state matrix Y.

        :param A_known: if given, will generate basis that is orthogonal to these given constraints.

        :return: basis
        """
        # if there is a known list of constraints, add them to the Y so that resulting nullspace is orthogonal to them
        if basis_known is not None:
            if len(A_known):
                print(
                    "Warning: ignoring given A_known because basis_all is also given."
                )
            Y = np.vstack([Y, basis_known.T])
        elif len(A_known):
            A = np.vstack([self.get_augmented_vec(self.get_vec(a)) for a in A_known])
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
        return basis, S

    def get_reduced_vec(self, bi, var_subset=None):
        if isinstance(bi, np.ndarray):
            len_b = len(bi)
        else:
            # bi can be a scipy sparse matrix,
            len_b = bi.shape[1]

        parameters = self.get_parameters()
        dim_X = self.get_dim_X(var_subset)
        n_parts = len_b / dim_X
        assert n_parts == len(parameters)

        ai = np.zeros(dim_X)
        for i, p in enumerate(parameters):
            if isinstance(bi, np.ndarray):
                ai += p * bi[i * dim_X : (i + 1) * dim_X]
            else:
                ai += p * bi[0, i * dim_X : (i + 1) * dim_X].toarray().flatten()
        return ai

    def get_augmented_vec(self, ai):
        n_parameters = self.get_dim_P()
        return np.hstack([ai, np.zeros((n_parameters - 1) * len(ai))])

    def generate_matrices(
        self, basis, normalize=NORMALIZE, sparse=True, trunc_tol=1e-10, var_dict=None
    ):
        """
        Generate constraint matrices from the rows of the nullspace basis matrix.
        """
        try:
            n_basis = len(basis)
        except:
            n_basis = basis.shape[0]
        A_list = []
        for i in range(n_basis):
            ai = self.get_reduced_vec(basis[i], var_dict)
            Ai = self.get_mat(ai, sparse=sparse, trunc_tol=trunc_tol, var_dict=var_dict)
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
            p = self.parameters
            x = self.get_x(t, p)

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
