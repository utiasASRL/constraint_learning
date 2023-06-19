from abc import ABC, abstractmethod
import itertools

import numpy as np
import matplotlib.pylab as plt
import scipy.linalg as la
import scipy.sparse as sp

from poly_matrix import PolyMatrix

# consider singular value zero below this
EPS_SVD = 1e-8

# tolerance for feasibility error of learned constraints
EPS_ERROR = 1e-8

PRUNE = True  # prune learned matrices to make sure they are all lin. indep.

# basis pursuit method, can be
# - qr: qr decomposition
# - qrp: qr decomposition with permutations (sparser)
# - svd: svd
METHOD = "qrp"

NORMALIZE = False  # normalize learned Ai or not, (True)
FACTOR = 2.0  # how much to oversample (>= 1)

# number of scalable elements to add.
MAX_N_SUBSETS = 2


class StateLifter(ABC):
    @staticmethod
    def create_symmetric(triu_vector, size):
        mat = np.zeros((size, size))
        mat[np.triu_indices(size)] = triu_vector
        mat += mat.T
        mat[range(size), range(size)] /= 2.0
        return mat

    @staticmethod
    def test_S_cutoff(S, corank, eps=EPS_SVD):
        if corank > 1:
            try:
                assert abs(S[-corank]) / eps < 1e-1  # 1e-1  1e-10
                assert abs(S[-corank - 1]) / eps > 10  # 1e-11 1e-10
            except:
                print(f"there might be a problem with the chosen threshold {eps}:")
                print(S[-corank], eps, S[-corank - 1])

    def __init__(self, theta_shape, M, L=0):
        self.theta_shape = theta_shape
        if len(theta_shape) > 1:
            self.N_ = np.multiply(*theta_shape)
        else:
            self.N_ = theta_shape[0]
        self.M_ = M
        self.L = L
        self.setup = None

        self.var_dict_ = None
        self.param_dict_ = None
        self.theta_ = None

        # fixing seed for testing purposes
        np.random.seed(1)
        self.generate_random_setup()

    @property
    def var_dict(self):
        return self.var_dict_

    @property
    def param_dict(self):
        return self.param_dict_

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
        try:
            len_vec = len(vec)
        except:
            len_vec = vec.shape[1]
        dim_x = int(0.5 * (-1 + np.sqrt(1 + 8 * len_vec)))
        assert dim_x == self.get_dim_x(var_dict)

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

    def get_labels(self, p, zi, zj):
        # for diagonal, we will only have half of the matrix
        labels = []
        size_i = self.var_dict[zi]
        size_j = self.var_dict[zj]
        if (size_i > 1) or (size_j > 1):
            if zi == zj:
                size = int(size_i * (size_i + 1) / 2)
                # only upper diagonal for i == j
                key_pairs = itertools.combinations_with_replacement(range(size_i), 2)
            else:
                size = size_i * size_j
                key_pairs = itertools.product(range(size_i), range(size_j))
            for i, j in key_pairs:
                labels.append(f"{self.param_dict[p]}.{zi}.{zj}:{i}-{j}")
        else:
            size = 1
            labels.append(f"{self.param_dict[p]}.{zi}.{zj}")
        assert len(labels) == size
        return labels

    def var_list_all(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict.keys()
        vectorized_var_list = list(
            itertools.combinations_with_replacement(var_subset, 2)
        )
        label_list = []
        for p in range(self.get_dim_P()):
            for zi, zj in vectorized_var_list:
                label_list += self.get_labels(p, zi, zj)
            assert len(label_list) == (p + 1) * self.get_dim_X(var_subset)
        return label_list

    def var_dict_all(self, var_subset=None):
        return {l: 1 for l in self.var_list_all(var_subset)}

    def get_basis_from_poly_rows(self, basis_poly_list, var_subset=None):
        if var_subset is not None:
            var_dict = {k: self.var_dict[k] for k in var_subset}
        else:
            var_dict = self.var_dict

        all_dict = {l: 1 for l in self.var_list_all(var_subset)}
        basis_reduced = np.empty((0, len(all_dict)))
        for i, bi_poly in enumerate(basis_poly_list):
            # test that this constraint holds

            bi = bi_poly.get_matrix((["l"], all_dict))

            if bi.shape[1] == self.get_dim_X(var_subset) * self.get_dim_P():
                ai = self.get_reduced_a(bi, var_subset=var_subset)
                Ai = self.get_mat(ai, var_dict=var_dict)
            elif bi.shape[1] == self.get_dim_X():
                Ai = self.get_mat(bi, var_subset=var_subset)

            err, idx = self.test_constraints([Ai], errors="print", tol=EPS_ERROR)
            if len(idx):
                print(f"found b{i} has error: {err[0]}")
                continue

            # test that this constraint is lin. independent of previous ones.
            basis_reduced_test = np.vstack([basis_reduced, bi.toarray()])
            rank = np.linalg.matrix_rank(basis_reduced_test)
            if rank == basis_reduced_test.shape[0]:
                basis_reduced = basis_reduced_test
            else:
                print(f"b{i} is linearly dependant after factoring out parameters.")
        print(f"left with {basis_reduced.shape} total constraints")
        return basis_reduced

        # there are still residual dependent vectors which only appear
        # after summing out the parameters. we want to remove those.
        import scipy.linalg as la

        __, r, p = la.qr(basis_learned, pivoting=True, mode="economic")
        rank = np.where(np.abs(np.diag(r)) > EPS_SVD)[0][-1] + 1
        if rank < len(basis_learned.shape[0]):
            # sanity check
            basis_learned = basis_learned[p[:rank], :]
            __, r, p = la.qr(basis_learned, pivoting=True, mode="economic")
            rank_new = np.where(np.abs(np.diag(r)) > EPS_SVD)[0][-1] + 1
            assert rank_new == rank
        return basis_learned

    def get_vector_dense(self, poly_row_sub):
        # complete the missing variables
        var_dict = self.var_dict_all()
        poly_row_all = poly_row_sub.get_matrix((["l"], var_dict), output_type="poly")
        vector = np.empty(0)
        for param in self.param_dict.values():
            # extract each block corresponding to a bigger matrix
            sub_mat = PolyMatrix(symmetric=True)
            for vari, varj in itertools.combinations_with_replacement(self.var_dict, 2):
                val = poly_row_all["l", f"{param}.{vari}.{varj}"]
                if np.ndim(val) > 0:
                    if vari != varj:
                        sub_mat[vari, varj] = val.reshape(
                            self.var_dict[vari], self.var_dict[varj]
                        )
                    else:
                        mat = self.create_symmetric(val, self.var_dict[vari])
                        sub_mat[vari, varj] = mat
                elif val != 0:
                    sub_mat[vari, varj] = val
            mat = sub_mat.get_matrix(self.var_dict).toarray()
            vector = np.r_[vector, mat[np.triu_indices(mat.shape[0])]]
        return np.array(vector)

    def get_A_learned(
        self,
        A_known: list = [],
        eps: float = EPS_SVD,
        plot: bool = False,
        Y: np.ndarray = None,
        factor: int = FACTOR,
        method: str = METHOD,
        incremental: bool = False,
        normalize: bool = NORMALIZE,
    ):
        if incremental:
            assert Y is None
            assert len(A_known) == 0
            basis_list = self.get_basis_list_incremental(
                eps=eps,
                plot=plot,
                factor=factor,
                method=method,
            )
            basis_list_all = self.augment_basis_list(basis_list, normalize=normalize)
            basis_learned = self.get_basis_from_poly_rows(basis_list_all)
            A_learned = self.generate_matrices(basis_learned, normalize=normalize)
        else:
            A_learned, basis_poly = self.get_basis_learned(
                A_known=A_known,
                eps=eps,
                plot=plot,
                Y=Y,
                factor=factor,
                method=method,
            )
        return A_learned, basis_poly

    def convert_b_to_poly(self, b, var_subset):
        var_dict = {k: v for k, v in self.var_dict.items() if k in var_subset}
        dim_X = self.get_dim_X(var_subset)
        dim_x = self.get_dim_x(var_subset)
        dim_P = self.get_dim_P()

        poly_all = PolyMatrix(symmetric=False)
        for p in range(dim_P):
            block = b[p * dim_X : (p + 1) * (dim_X)]
            mat = self.create_symmetric(block, dim_x)
            poly_mat, __ = PolyMatrix.init_from_sparse(mat, var_dict)
            for keyi, keyj in itertools.combinations_with_replacement(var_dict, 2):
                if keyi in poly_mat.matrix and keyj in poly_mat.matrix[keyi]:
                    val = poly_mat.matrix[keyi][keyj]
                    labels = self.get_labels(p, keyi, keyj)
                    if keyi != keyj:
                        vals = val.flatten()
                    else:
                        vals = val[np.triu_indices(val.shape[0])]
                    assert len(labels) == len(vals)
                    for l, v in zip(labels, vals):
                        if np.any(np.abs(v)) > 1e-10:
                            poly_all["l", l] = v

        return poly_all

    def zero_pad_subvector(self, b, var_subset, target_subset=None):
        """Add zero padding to b vector learned for a subset of variables(e.g. as obtained from learning method)"""
        poly_all = self.convert_b_to_poly(b, var_subset)
        var_dict = {k: v for k, v in self.var_dict.items() if k in var_subset}
        if target_subset is None:
            target_subset = self.var_dict
        all_dict = self.var_dict_all(target_subset)
        return poly_all.get_matrix((["l"], all_dict)).toarray()

        dim_P = self.get_dim_P()
        dim_X = self.get_dim_X(var_subset)
        dim_x = self.get_dim_x(var_subset)

        poly_all = PolyMatrix(symmetric=False)
        b_all = np.empty(0)
        for p in range(dim_P):
            block = b[p * dim_X : (p + 1) * (dim_X)]
            mat = self.create_symmetric(block, dim_x)
            poly_mat, __ = PolyMatrix.init_from_sparse(mat, var_dict)

            # TODO(FD) implement below using sparse matrices?
            mat = poly_mat.get_matrix(target_subset).toarray()
            bi_all = np.r_[bi_all, mat[np.triu_indices(mat.shape[0])]]

            for keyi, keyj in itertools.combinations_with_replacement(
                poly_mat.variable_dict_i, 2
            ):
                if keyi in poly_mat.matrix and keyj in poly_mat.matrix[keyi]:
                    val = poly_mat.matrix[keyi][keyj]
                    labels = self.get_labels(p, keyi, keyj)
                    assert len(labels) == np.size(val)
                    for l, v in zip(labels, val.flatten()):
                        if np.any(np.abs(v)) > 1e-10:
                            poly_all["l", l] = v

        assert len(bi_all) == self.get_dim_X(target_subset) * self.get_dim_P()
        return bi_all, poly_all

    def get_incremental_vars(self):
        return tuple(["l", "x"] + [f"z_{i}" for i in range(MAX_N_SUBSETS)])

    def get_basis_list_incremental(
        self,
        plot: bool = False,
    ):
        """Learn the constraint matrices."""
        var_subsets = []
        for k in range(MAX_N_SUBSETS + 1):
            var_subsets.append(tuple(["l", "x"] + [f"z_{i}" for i in range(k)]))

        # keep track of current set of lin. independent constraints
        dim_Y = self.get_dim_Y()
        basis_learned = np.empty((0, dim_Y))
        current_rank = 0

        basis_list = []
        for var_subset in var_subsets:
            Y = self.generate_Y(var_subset=var_subset)

            # extract subset of known matrices given the current variables
            # sub_A_known = self.extract_A_known(A_known, var_subset)

            # TODO(FD) can we enforce lin. independance to previously found
            # matrices at this point?
            basis_new, S = self.get_basis(Y)
            corank = basis_new.shape[0]

            if corank == 0:
                print(f"{var_subset}: no new learned matrices found")
                continue

            print(f"{var_subset}: {corank} learned matrices found")
            self.test_S_cutoff(S, corank)

            if plot:
                from lifters.plotting_tools import plot_singular_values

                plot_singular_values(S)

            # find out which of the constraints are linearly dependant of the others.
            # TODO(FD) sum out constraints to be reduce even more!
            for i, bi_sub in enumerate(basis_new):
                bi_sub[np.abs(bi_sub) < 1e-10] = 0.0
                bi, bi_poly = self.zero_pad_subvector(bi_sub, var_subset)

                basis_learned_test = np.vstack([basis_learned, bi])
                new_rank = np.linalg.matrix_rank(basis_learned_test, tol=1e-10)
                if new_rank == current_rank + 1:
                    print(f"b{i} is valid basis vector.")
                    basis_list.append(bi_poly)
                    basis_learned = basis_learned_test
                    current_rank += 1
                elif new_rank == current_rank:
                    print(f"b{i} is linearly dependent.")
                else:
                    print(
                        f"Warning: invalid rank change from rank {current_rank} to {new_rank}"
                    )
        return basis_list

    def get_basis_learned(
        self,
        A_known: list = [],
        eps: float = EPS_SVD,
        plot: bool = False,
        Y: np.ndarray = None,
        factor: int = FACTOR,
        method: str = METHOD,
        normalize: bool = NORMALIZE,
    ) -> dict:
        dim_Y = self.get_dim_Y()

        A_learned = []
        A_learned += A_known

        if len(A_known):
            basis_learned = np.vstack(
                [self.augment_using_zero_padding(self.get_vec(A)) for A in A_known]
            )
            assert basis_learned.shape[1] == dim_Y
            current_rank = np.linalg.matrix_rank(basis_learned)
        else:
            basis_learned = np.empty((0, dim_Y))
            current_rank = 0

        Y = self.generate_Y(factor=factor)
        basis_new, S = self.get_basis(
            Y,
            method=method,
            eps=eps,
            A_known=A_known,  # basis_known=basis_all
        )
        corank = basis_new.shape[0]

        print(f"{corank} learned matrices found")
        self.test_S_cutoff(S, corank, eps)

        if plot:
            from lifters.plotting_tools import plot_singular_values

            plot_singular_values(S, eps=eps)

        # find out which of the constraints are linearly dependant of the others.
        # TODO(FD) sum out constraints to be reduce even more!
        for i, bi in enumerate(basis_new):
            basis_learned_test = np.vstack([basis_learned, bi])
            new_rank = np.linalg.matrix_rank(basis_learned_test, tol=1e-10)
            if new_rank == current_rank + 1:
                # print(f"b{i} is valid basis vector.")
                basis_learned = basis_learned_test
                current_rank += 1
            elif new_rank == current_rank:
                print(f"b{i} is linearly dependent.")
            else:
                print(
                    f"Warning: invalid rank change from rank {current_rank} to {new_rank}"
                )
        A_learned = self.generate_matrices(basis_learned, normalize=normalize)

        basis_poly = PolyMatrix(symmetric=False)
        for i, bi in enumerate(basis_learned):
            bi_same, bi_poly = self.zero_pad_subvector(bi, self.var_dict)
            np.testing.assert_allclose(bi_same, bi)
            for key in bi_poly.variable_dict_j:
                basis_poly[i, key] = bi_poly["l", key]
        return A_learned, basis_poly

    def augment_basis_list(self, basis_list, var_subset, n_landmarks=5, verbose=False):
        # given the learned matrices we need to generalize to all landmarks!

        # TODO(FD) generalize below; but for now, this is easier to debug and understand.
        new_poly_rows = []
        for bi in basis_list:
            # find the number of variables that this touches.
            bi_poly = self.convert_b_to_poly(bi, var_subset)
            unique_idx = set()
            for key in bi_poly.variable_dict_j:
                vars = key.split(".")
                for var in vars:
                    var_base = var.split(":")[0]
                    if "_" in var_base:
                        i = int(var_base.split("_")[-1])
                        unique_idx.add(i)

            # if z_0 is in this constraint, repeat the constraint for each landmark.
            for idx in itertools.combinations(range(n_landmarks), len(unique_idx)):
                new_poly_row = PolyMatrix(symmetric=False)
                for key in bi_poly.variable_dict_j:
                    # need intermediate variables cause otherwise z_0 -> z_1 -> z_2 etc. can happen.
                    key_ij = key
                    for from_, to_ in zip(range(len(unique_idx)), idx):
                        key_ij = key_ij.replace(f"z_{from_}", f"zi_{to_}")
                        key_ij = key_ij.replace(f"p_{from_}", f"pi_{to_}")
                    key_ij = key_ij.replace("zi", "z").replace("pi", "p")
                    if verbose and (key != key_ij):
                        print("changed", key, "to", key_ij)
                    new_poly_row["l", key_ij] = bi_poly["l", key]
                new_poly_rows.append(new_poly_row)
        return new_poly_rows

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

    def get_dim_Y(self, var_subset=None):
        dim_X = self.get_dim_X(var_subset=var_subset)
        dim_P = self.get_dim_P()
        return int(dim_X * dim_P)

    def get_dim_X(self, var_subset=None):
        dim_x = self.get_dim_x(var_subset)
        return int(dim_x * (dim_x + 1) / 2)

    def get_dim_P(self):
        return len(self.get_parameters())

    def generate_Y(self, factor=3, ax=None, var_subset=None):
        # need at least dim_Y different random setups
        dim_Y = self.get_dim_Y(var_subset)
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
        eps=EPS_SVD,
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
            A = np.vstack(
                [self.augment_using_zero_padding(self.get_vec(a)) for a in A_known]
            )
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

    def get_reduced_a(self, bi, var_subset=None):
        """
        Extract first block of bi by summing over other blocks times the parameters.
        """
        if isinstance(bi, np.ndarray):
            len_b = len(bi)
        else:
            # bi can be a scipy sparse matrix,
            len_b = bi.shape[1]

        n_params = self.get_dim_P()
        dim_X = self.get_dim_X(var_subset)
        n_parts = len_b / dim_X
        assert (
            n_parts == n_params
        ), f"{len_b} does not not split in dim_P={n_params} parts of size dim_X={dim_X}"

        parameters = self.get_parameters()
        ai = np.zeros(dim_X)
        for i, p in enumerate(parameters):
            if isinstance(bi, np.ndarray):
                ai += p * bi[i * dim_X : (i + 1) * dim_X]
            else:
                ai += p * bi[0, i * dim_X : (i + 1) * dim_X].toarray().flatten()
        return ai

    def augment_using_zero_padding(self, ai):
        n_parameters = self.get_dim_P()
        return np.hstack([ai, np.zeros((n_parameters - 1) * len(ai))])

    def augment_using_parameters(self, x):
        parameters = self.get_parameters()
        return np.kron(parameters, x)

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
            ai = self.get_reduced_a(basis[i], var_dict)
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
        :param constraints: can be either list of sparse matrices, or the basis matrix.
        :param errors: "raise" or "print" detected violations.
        """
        import scipy.sparse as sp

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
            # else:
            #    print(f"no violation at {j}")
        return max_violation, j_bad

    def get_A0(self, var_subset=None):
        if var_subset is not None:
            var_dict = {k: self.var_dict[k] for k in var_subset}
        else:
            var_dict = self.var_dict
        A0 = PolyMatrix()
        A0["l", "l"] = 1.0
        return A0.get_matrix(var_dict)

    def get_A_b_list(self, A_list, var_subset=None):
        return [(self.get_A0(var_subset), 1.0)] + [(A, 0.0) for A in A_list]

    def run(self, n_dual: int = 3, plot: bool = False, noise: float = 1e-4):
        """Convenience function to quickly test and debug lifter

        :param n_dual

        """
        import matplotlib.pylab as plt

        from lifters.plotting_tools import plot_matrices
        from solvers.common import find_local_minimum, solve_sdp_cvxpy

        Q, y = self.get_Q(noise=noise)

        print("find local minimum...")
        that, local_cost = find_local_minimum(
            self, y, delta=noise, n_inits=1, plot=False, verbose=True
        )

        A_known = self.get_A_known()
        A_list = self.get_A_learned(
            eps=EPS_SVD, method="qrp", plot=plot, A_known=A_known
        )
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
