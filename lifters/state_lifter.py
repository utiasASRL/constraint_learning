import itertools

import numpy as np
import matplotlib.pylab as plt
import scipy.linalg as la
import scipy.sparse as sp

from utils.common import increases_rank
from utils.common import upper_triangular
from lifters.base_class import BaseClass

from poly_matrix import PolyMatrix

# consider singular value zero below this
EPS_SVD = 1e-8

# set elements below this threshold to zero.
EPS_SPARSE = 1e-10

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

# can we find a cheaper way to achieve what the below does?
N_CLEANING_STEPS = 1

# number of scalable elements to add.
MAX_N_SUBSETS = 2


class StateLifter(BaseClass):
    @staticmethod
    def get_variable_indices(var_subset):
        return [int(v.split("_")[-1]) for v in var_subset if v.startswith("z_")]

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

    def get_level_dims(self, n=1):
        assert (
            self.level == "no"
        ), "Need to overwrite get_level_dims to use level different than 'no'"
        return 0

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.generate_random_theta()
        return self.theta_

    @property
    def base_var_dict(self):
        var_dict = {}
        # var_dict.update({f"c_{i}": self.d for i in range(self.d)})
        # var_dict.update({"t": self.d})
        var_dict = {f"x": self.d**2 + self.d}
        return var_dict

    @property
    def sub_var_dict(self):
        level_dim = self.get_level_dims()[self.level]
        var_dict = {f"z_{k}": self.d + level_dim for k in range(self.n_landmarks)}
        return var_dict

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {"l": 1}
            self.var_dict_.update(self.base_var_dict)
            self.var_dict_.update(self.sub_var_dict)
        return self.var_dict_

    def get_var_dict(self, var_subset=None):
        if var_subset is not None:
            return {k: v for k, v in self.var_dict.items() if k in var_subset}
        return self.var_dict

    def get_p(self, parameters=None, var_subset=None):
        """
        :param parameters: list of all parameters
        :param var_subset: subset of variables tat we care about (will extract corresponding parameters)
        """
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        if self.param_level == "no":
            return np.array([1.0])

        landmarks = self.get_variable_indices(var_subset)
        if len(landmarks):
            all_p = parameters[1:].reshape((self.n_landmarks, self.d))
            sub_p = np.r_[1.0, all_p[landmarks, :].flatten()]
            if self.param_level == "p":
                return sub_p
            elif self.param_level == "ppT":
                return upper_triangular(sub_p)
        else:
            return np.array([1.0])

    def get_vec(self, mat, correct=True):
        """Convert NxN Symmetric matrix to (N+1)N/2 vectorized version that preserves inner product.

        :param mat: (spmatrix or ndarray) symmetric matrix
        :return: ndarray
        """
        from copy import deepcopy

        mat = deepcopy(mat)
        if correct:
            if isinstance(mat, sp.spmatrix):
                ii, jj = mat.nonzero()
                mat[ii, jj] *= np.sqrt(2.0)
                diag = ii == jj
                mat[ii[diag], jj[diag]] /= np.sqrt(2)
            else:
                mat *= np.sqrt(2.0)
                mat[range(mat.shape[0]), range(mat.shape[0])] /= np.sqrt(2)
        return np.array(mat[np.triu_indices(n=mat.shape[0])]).flatten()

    def get_mat(
        self, vec, sparse=False, trunc_tol=EPS_SPARSE, var_dict=None, correct=True
    ):
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
            offdiag = triu_i_nnz != triu_j_nnz
            diag = triu_i_nnz == triu_j_nnz
            triu_i = triu_i_nnz[offdiag]
            triu_j = triu_j_nnz[offdiag]
            diag_i = triu_i_nnz[diag]
            if correct:
                # divide off-diagonal elements by sqrt(2)
                vec_nnz_off = vec_nnz[offdiag] / np.sqrt(2)
            else:
                vec_nnz_off = vec_nnz[offdiag]
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

            if correct:
                # divide all elements by sqrt(2)
                Ai[triu_i_nnz, triu_j_nnz] = vec_nnz / np.sqrt(2)
                Ai[triu_j_nnz, triu_i_nnz] = vec_nnz / np.sqrt(2)
                # undo operation for diagonal
                Ai[range(dim_x), range(dim_x)] *= np.sqrt(2)
            else:
                Ai[triu_i_nnz, triu_j_nnz] = vec_nnz
                Ai[triu_j_nnz, triu_i_nnz] = vec_nnz

        if var_dict is None:
            return Ai
        # if var_dict is not None, then Ai corresponds to the subblock
        # defined by var_dict, of the full constraint matrix.
        Ai_poly, __ = PolyMatrix.init_from_sparse(Ai, var_dict)
        return Ai_poly.get_matrix(self.var_dict)

    def get_A_known(self) -> list:
        return []

    def extract_A_known(self, A_known, var_subset, output_type="csc"):
        if output_type == "dense":
            sub_A_known = np.empty((0, self.get_dim_Y(var_subset)))
        else:
            sub_A_known = []
        for A in A_known:
            A_poly, var_dict = PolyMatrix.init_from_sparse(A, self.var_dict)

            assert len(A_poly.get_variables()) > 0

            # if all of the non-zero elements of A_poly are in var_subset,
            # we can use this matrix.
            if np.all([v in var_subset for v in A_poly.get_variables()]):
                Ai = A_poly.get_matrix(
                    self.get_var_dict(var_subset), output_type=output_type
                )
                if output_type == "dense":
                    ai = self.augment_using_zero_padding(
                        self.get_vec(Ai, correct=True), var_subset
                    )
                    sub_A_known = np.r_[sub_A_known, ai[None, :]]
                else:
                    sub_A_known.append(Ai)
        return sub_A_known

    def get_labels(self, p, zi, zj):
        labels = []
        size_i = self.var_dict[zi]
        size_j = self.var_dict[zj]
        if zi == zj:
            # only upper diagonal for i == j
            key_pairs = itertools.combinations_with_replacement(range(size_i), 2)
        else:
            key_pairs = itertools.product(range(size_i), range(size_j))
        for i, j in key_pairs:
            label = f"{p}-"
            label += f"{zi}:{i}." if size_i > 1 else f"{zi}."
            label += f"{zj}:{j}" if size_j > 1 else f"{zj}"
            labels.append(label)
        return labels

    def var_list_row(self, var_subset=None):
        if var_subset is None:
            var_subset = list(self.var_dict.keys())
        elif type(var_subset) == dict:
            var_subset = list(var_subset.keys())

        label_list = []
        param_dict = self.get_param_idx_dict(var_subset)
        for idx, key in enumerate(param_dict.keys()):
            for i in range(len(var_subset)):
                zi = var_subset[i]
                sizei = self.var_dict[zi]
                for di in range(sizei):
                    keyi = f"{zi}:{di}" if sizei > 1 else f"{zi}"
                    for j in range(i, len(var_subset)):
                        zj = var_subset[j]
                        sizej = self.var_dict[zj]
                        if zi == zj:
                            djs = range(di, sizej)
                        else:
                            djs = range(sizej)

                        for dj in djs:
                            keyj = f"{zj}:{dj}" if sizej > 1 else f"{zj}"
                            label_list.append(f"{key}-{keyi}.{keyj}")
            # for zi, zj in vectorized_var_list:
            # label_list += self.get_labels(key, zi, zj)
            assert len(label_list) == (idx + 1) * self.get_dim_X(var_subset)
        return label_list

    def var_dict_row(self, var_subset=None):
        return {l: 1 for l in self.var_list_row(var_subset)}

    def get_basis_from_poly_rows(self, basis_poly_list, var_subset=None):
        var_dict = self.get_var_dict(var_subset=var_subset)

        all_dict = {l: 1 for l in self.var_list_row(var_subset)}
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
        var_dict = self.var_dict_row()
        poly_row_all = poly_row_sub.get_matrix((["l"], var_dict), output_type="poly")
        vector = np.empty(0)
        for param in self.get_param_idx_dict().keys():
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
        incremental: bool = False,
        normalize: bool = NORMALIZE,
        method: str = METHOD,
    ):
        if incremental:
            assert len(A_known) == 0
            var_subset = ["l", "x", "z_0"]
            basis_list = self.get_basis_list(
                var_subset=var_subset,
                A_known=A_known,
                eps_svd=eps,
                plot=plot,
                method=method,
            )
            basis_list_all = self.augment_basis_list(basis_list, normalize=normalize)
            basis_learned = self.get_basis_from_poly_rows(basis_list_all)
            A_learned = self.generate_matrices(basis_learned, normalize=normalize)
        else:
            basis_list = self.get_basis_list(
                var_subset=self.var_dict,
                A_known=A_known,
                eps_svd=eps,
                plot=plot,
                method=method,
            )
            A_learned = self.generate_matrices(basis_list, normalize=normalize)
        return A_learned

    def convert_poly_to_a(self, poly_row, var_subset=None):
        """Convert poly-row to reduced a.

        poly-row has elements with keys "pk:l.xi:m.xj:n",
        meaning this element corresponds to the l-th element of parameter i,
        and the m-n-th element of xj times xk.
        """
        var_dict = self.get_var_dict(var_subset)
        parameters = self.get_p(var_subset=var_subset)
        param_dict = self.get_param_idx_dict(var_subset)

        poly_mat = PolyMatrix(symmetric=True)
        for key in poly_row.variable_dict_j:
            param, var_keys = key.split("-")
            keyi_m, keyj_n = var_keys.split(".")
            if param in ["l", "l.l"]:
                poly_mat[keyi_m, keyj_n] = poly_row["l", key]
            else:
                poly_mat[keyi_m, keyj_n] += (
                    poly_row["l", key] * parameters[param_dict[param]]
                )

        mat_var_list = []
        for var, size in var_dict.items():
            if size == 1:
                mat_var_list.append(var)
            else:
                mat_var_list += [f"{var}:{i}" for i in range(size)]
        mat_sparse = poly_mat.get_matrix({m: 1 for m in mat_var_list})
        return np.array(mat_sparse[np.triu_indices(mat_sparse.shape[0])]).flatten()

    def convert_a_to_polyrow(
        self, a, var_subset=None, eps_sparse=EPS_SPARSE
    ) -> PolyMatrix:
        """Convert a array to poly-row."""
        if var_subset is None:
            var_subset = self.var_dict
        var_dict = self.get_var_dict(var_subset)
        dim_x = self.get_dim_x(var_subset)
        dim_X = self.get_dim_X(var_subset)
        assert len(a) == dim_X

        mat = self.create_symmetric(a, dim_x)
        poly_mat, __ = PolyMatrix.init_from_sparse(mat, var_dict)
        poly_row = PolyMatrix(symmetric=False)
        for keyi, keyj in itertools.combinations_with_replacement(var_dict, 2):
            if keyi in poly_mat.matrix and keyj in poly_mat.matrix[keyi]:
                val = poly_mat.matrix[keyi][keyj]
                labels = self.get_labels("l", keyi, keyj)
                if keyi != keyj:
                    vals = val.flatten()
                else:
                    # TODO: use get_vec instead?
                    vals = val[np.triu_indices(val.shape[0])]
                assert len(labels) == len(vals)
                for l, v in zip(labels, vals):
                    if np.any(np.abs(v) > eps_sparse):
                        poly_row["l", l] = v
        return poly_row

    def convert_b_to_polyrow(self, b, var_subset, tol=1e-10) -> PolyMatrix:
        """Convert (augmented) b array to poly-row."""
        assert len(b) == self.get_dim_Y(var_subset)
        poly_row = PolyMatrix(symmetric=False)
        var_list = self.var_list_row(var_subset)
        for key, val in zip(var_list, b):
            if abs(val) > tol:
                poly_row["l", key] = val
        return poly_row

    def zero_pad_subvector(self, b, var_subset, target_subset=None):
        b_row = self.convert_b_to_polyrow(b, var_subset)
        target_row_dict = self.var_dict_row(var_subset=target_subset)

        # find out if the relevant variables of b are a subset of target_subset.
        if set(b_row.variable_dict_j).issubset(target_row_dict):
            return self.zero_pad_subvector_old(b, var_subset, target_subset)
        else:
            return None

    def zero_pad_subvector_old(self, b, var_subset, target_subset=None):
        """Add zero padding to b vector learned for a subset of variables(e.g. as obtained from learning method)"""
        var_dict = self.get_var_dict(var_subset)
        if target_subset is None:
            target_subset = self.var_dict
        dim_X = self.get_dim_X(var_subset)
        dim_x = self.get_dim_x(var_subset)
        param_dict = self.get_param_idx_dict(var_subset)

        # default is that b is an augmented vector (with parameters)
        if len(b) != dim_X * len(param_dict):
            # can call this function with "a" vector (equivalent to b without parameters)
            b = self.augment_using_zero_padding(b)

        param_dict_target = self.get_param_idx_dict(target_subset)
        dim_X_target = self.get_dim_X(target_subset)
        bi_all = np.zeros(self.get_dim_Y(target_subset))
        for p, key in enumerate(param_dict.keys()):
            block = b[p * dim_X : (p + 1) * (dim_X)]
            mat_small = self.create_symmetric(block, dim_x)
            poly_mat, __ = PolyMatrix.init_from_sparse(mat_small, var_dict)
            mat_target = poly_mat.get_matrix(target_subset).toarray()

            pt = param_dict_target[key]
            bi_all[pt * dim_X_target : (pt + 1) * dim_X_target] = mat_target[
                np.triu_indices(mat_target.shape[0])
            ]

        # below doesn't pass. this would make for a cleaner implementation, consider fixing.
        # poly_row = self.convert_b_to_polyrow(b, var_subset)
        # row_target_dict = self.var_dict_all(target_subset)
        # bi_all_test = poly_row.get_matrix(
        #    (["l"], row_target_dict), output_type="dense"
        # ).flatten()
        # np.testing.assert_allclose(bi_all, bi_all_test)
        return bi_all

    def get_incremental_vars(self):
        return tuple(["l", "x"] + [f"z_{i}" for i in range(MAX_N_SUBSETS)])

    def get_basis_list(
        self,
        var_subset,
        A_known: list = [],
        plot: bool = False,
        eps_svd: float = EPS_SVD,
        eps_sparse: float = EPS_SPARSE,
        method: str = METHOD,
    ):
        """Generate list of PolyRow basis vectors"""
        basis_list = []

        Y = self.generate_Y(var_subset=var_subset)
        if len(A_known):
            A_known = self.extract_A_known(A_known, var_subset, output_type="dense")
            Y = np.r_[Y, A_known]
            for i in range(A_known.shape[0]):
                basis_list.append(self.convert_b_to_polyrow(A_known[i], var_subset))

        for i in range(N_CLEANING_STEPS + 1):
            # TODO(FD) can we enforce lin. independance to previously found
            # matrices at this point?
            basis_new, S = self.get_basis(Y, eps_svd=eps_svd, method=method)
            corank = basis_new.shape[0]

            if corank == 0:
                print(f"{var_subset}: no new learned matrices found")
                return basis_list
            print(f"{var_subset}: {corank} learned matrices found")
            self.test_S_cutoff(S, corank, eps=eps_svd)

            errors = np.abs(basis_new @ Y.T)  # Nb x n x n x Ns = Nb x Ns
            bad_bins = np.unique(np.argmax(errors, axis=1))
            if plot:
                fig, ax = plt.subplots()
                ax.semilogy(np.min(errors, axis=1))
                ax.semilogy(np.max(errors, axis=1))
                ax.semilogy(np.median(errors, axis=1))
                ax.semilogy(S[-corank:])

            Y = np.delete(Y, bad_bins, axis=0)

        if plot:
            from lifters.plotting_tools import plot_singular_values

            plot_singular_values(S, eps=eps_svd)

        # find out which of the constraints are linearly dependant of the others.
        current_basis = None
        for i, bi_sub in enumerate(basis_new):
            bi_poly = self.convert_b_to_polyrow(bi_sub, var_subset, tol=eps_sparse)
            ai = self.get_reduced_a(bi_sub, var_subset)
            if increases_rank(current_basis, ai):
                basis_list.append(bi_poly)
                if current_basis is None:
                    current_basis = ai[None, :]
                else:
                    current_basis = np.r_[current_basis, ai[None, :]]
        return basis_list

    def augment_basis_list(
        self, basis_list, var_subset, n_landmarks=None, verbose=False
    ):
        """
        Apply the learned patterns in basis_list to all landmarks.

        :param basis_list: list of sparse vectors or list of poly matrices.
        """

        if n_landmarks is None:
            n_landmarks = self.n_landmarks

        # TODO(FD) generalize below; but for now, this is easier to debug and understand.
        new_poly_rows = []
        for bi in basis_list:
            # find the number of variables that this touches.
            if isinstance(bi, PolyMatrix):
                bi_poly = bi
            else:
                bi_poly = self.convert_b_to_polyrow(bi, var_subset)
            unique_idx = set()
            for key in bi_poly.variable_dict_j:
                param, var_keys = key.split("-")
                vars = var_keys.split(".")
                vars += param.split(".")
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
        """Sample around ground truth.

        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def get_dim_x(self, var_subset=None):
        var_dict = self.get_var_dict(var_subset)
        return sum([val for val in var_dict.values()])

    def get_dim_Y(self, var_subset=None):
        dim_X = self.get_dim_X(var_subset=var_subset)
        dim_P = self.get_dim_P(var_subset=var_subset)
        return int(dim_X * dim_P)

    def get_dim_X(self, var_subset=None):
        dim_x = self.get_dim_x(var_subset)
        return int(dim_x * (dim_x + 1) / 2)

    def get_dim_P(self, var_subset=None):
        return len(self.get_p(var_subset=var_subset))

    def generate_Y(self, factor=FACTOR, ax=None, var_subset=None):
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
            p = self.get_p(parameters=parameters, var_subset=var_subset)
            Y[seed, :] = np.kron(p, self.get_vec(X))
        return Y

    def get_basis(
        self,
        Y,
        A_known: list = [],
        basis_known: np.ndarray = None,
        eps_svd=EPS_SVD,
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
            rank = np.sum(np.abs(S) > eps_svd)
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
            rank = np.where(S < eps_svd)[0][0]
            # decreasing order
            basis = Q[:, sorted_idx[rank:]].T
        elif method == "qrp":
            # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.
            Q, R, p = la.qr(Y, pivoting=True, mode="economic")
            S = np.abs(np.diag(R))
            rank = np.sum(S > eps_svd)
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
        elif isinstance(bi, PolyMatrix):
            bi = bi.get_matrix((["l"], self.var_dict_row(var_subset)))
            len_b = bi.shape[1]
        else:
            # bi can be a scipy sparse matrix,
            len_b = bi.shape[1]

        n_params = self.get_dim_P(var_subset)
        dim_X = self.get_dim_X(var_subset)
        n_parts = len_b / dim_X
        assert (
            n_parts == n_params
        ), f"{len_b} does not not split in dim_P={n_params} parts of size dim_X={dim_X}"

        parameters = self.get_p(var_subset=var_subset)
        ai = np.zeros(dim_X)
        for i, p in enumerate(parameters):
            if isinstance(bi, np.ndarray):
                ai += p * bi[i * dim_X : (i + 1) * dim_X]
            else:
                ai += p * bi[0, i * dim_X : (i + 1) * dim_X].toarray().flatten()
        return ai

    def augment_using_zero_padding(self, ai, var_subset=None):
        n_parameters = self.get_dim_P(var_subset=var_subset)
        return np.hstack([ai, np.zeros((n_parameters - 1) * len(ai))])

    def augment_using_parameters(self, x, var_subset=None):
        parameters = self.get_parameters()
        p = self.get_p(parameters, var_subset=var_subset)
        return np.kron(p, x)

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

        if type(var_dict) is list:
            var_dict = self.get_var_dict(var_dict)

        A_list = []
        for i in range(n_basis):
            ai = self.get_reduced_a(basis[i], var_dict)
            # Ai = self.create_symmetric(ai, size=self.get_dim_x(var_dict))
            Ai = self.get_mat(
                ai, sparse=sparse, trunc_tol=trunc_tol, var_dict=var_dict, correct=True
            )
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
        max_violation = -np.inf
        j_bad = []

        for j, A in enumerate(A_list):
            t = self.sample_theta()
            p = self.get_parameters()
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
                elif errors == "ignore":
                    pass
                else:
                    raise ValueError(errors)
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

    def get_param_idx_dict(self, var_subset=None):
        """
        Give the current subset of variables, extract the parameter dictionary to use.
        Example: var_subset = ['l', 'z_0']
        - if param_level == 'no': {'l': 0}
        - if param_level == 'p': {'l': 0, 'p_0:0': 1, ..., 'p_0:d-1': d}
        - if param_level == 'ppT': {'l': 0, 'p_0:0.p_0:0': 1, ..., 'p_0:d-1:.p_0:d-1': 1}
        """
        if self.param_level == "no":
            return {"l": 0}
        variables = self.get_variable_indices(var_subset)
        param_keys = ["l"] + [f"p_{i}:{d}" for i in variables for d in range(self.d)]
        if self.param_level == "p":
            param_dict = {p: i for i, p in enumerate(param_keys)}
        elif self.param_level == "ppT":
            i = 0
            param_dict = {}
            for pi, pj in itertools.combinations_with_replacement(param_keys, 2):
                if pi == pj == "l":
                    param_dict["l"] = i
                else:
                    param_dict[f"{pi}.{pj}"] = i
                i += 1
        return param_dict
