"""
The BaseClass contains all the functionalities that are required by StateLifter but that
are really tedious and uninteresting, such as converting between different formats.

Ideally, the user never has to look at this code, and we apologize in advance if you do
-- it is not super well documented.

"""

import itertools

import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix, unroll
from utils.common import upper_triangular


class BaseClass(object):
    # set elements below this threshold to zero.
    EPS_SPARSE = 1e-9

    @staticmethod
    def ravel_multi_index_triu(index_tuple, shape):
        """Equivalent of np.multi_index_triu, but using only the upper-triangular part of matrix."""
        ii, jj = index_tuple

        triu_mask = jj >= ii
        i_upper = ii[triu_mask]
        j_upper = jj[triu_mask]
        flat_indices = []
        for i, j in zip(i_upper, j_upper):
            # for i == 0: idx = j
            # for i == 1: idx = shape[0] + j
            # for i == 2: idx = shape[0] + shape[0]-1 + j
            idx = np.sum(range(shape[0] - i, shape[0])) + j
            flat_indices.append(idx)
        return flat_indices

    @staticmethod
    def unravel_multi_index_triu(flat_indices, shape):
        """Equivalent of np.multi_index_triu, but using only the upper-triangular part of matrix."""
        i_upper = []
        j_upper = []

        # for 4 x 4, this would give [4, 7, 9, 11]
        cutoffs = np.cumsum(list(range(1, shape[0] + 1))[::-1])
        for idx in flat_indices:
            i = np.where(idx < cutoffs)[0][0]
            if i == 0:
                j = idx
            else:
                j = idx - cutoffs[i - 1] + i
            i_upper.append(i)
            j_upper.append(j)
        return np.array(i_upper), np.array(j_upper)

    @staticmethod
    def get_variable_indices(var_subset, variable="z"):
        return np.unique(
            [int(v.split("_")[-1]) for v in var_subset if v.startswith(f"{variable}_")]
        )

    @staticmethod
    def create_symmetric(vec, correct=False, sparse=False):
        def get_dim_x(len_vec):
            return int(0.5 * (-1 + np.sqrt(1 + 8 * len_vec)))

        try:
            # vec is dense
            len_vec = len(vec)
            dim_x = get_dim_x(len_vec)
            triu = np.triu_indices(n=dim_x)
            mask = np.abs(vec) > BaseClass.EPS_SPARSE
            triu_i_nnz = triu[0][mask]
            triu_j_nnz = triu[1][mask]
            vec_nnz = vec[mask]
        except Exception:
            # vec is sparse
            len_vec = vec.shape[1]
            dim_x = get_dim_x(len_vec)
            vec.data[np.abs(vec.data) < BaseClass.EPS_SPARSE] = 0
            vec.eliminate_zeros()
            ii, jj = vec.nonzero()  # vec is 1 x jj
            triu_i_nnz, triu_j_nnz = BaseClass.unravel_multi_index_triu(
                jj, (dim_x, dim_x)
            )
            vec_nnz = np.array(vec[ii, jj]).flatten()
        # assert dim_x == self.get_dim_x(var_dict)

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
                dtype=float,
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
        return Ai

    ### Functionalities related to var_dict
    @property
    def var_dict_unroll(self):
        raise ValueError("use self.get_var_dict(unroll_keys=True)")

    def get_var_dict_unroll(self, var_subset=None):
        raise ValueError("use self.get_var_dict(unroll_keys=True)")

    def get_var_dict(self, var_subset=None, unroll_keys=False):
        if var_subset is not None:
            var_dict = {k: v for k, v in self.var_dict.items() if k in var_subset}
            if unroll_keys:
                return unroll(var_dict)
            else:
                return var_dict
        if unroll_keys:
            return unroll(self.var_dict)
        return self.var_dict

    def get_param_dict(self, param_subset=None):
        if param_subset is not None:
            return {k: v for k, v in self.param_dict.items() if k in param_subset}
        return self.param_dict

    ### Functionalities related to random setups
    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    @theta.setter
    def theta(self, t):
        assert (
            self.theta_ is None
        ), "The property self.theta is only meant to be set once!"
        self.theta_ = t

    @property
    def parameters(self):
        if self.parameters_ is None:
            self.parameters_ = self.sample_parameters()
            assert isinstance(self.parameters_, dict)
        return self.parameters_

    @parameters.setter
    def parameters(self, p):
        assert (
            self.parameters_ is None
        ), "The property self.parameters is only meant to be set once!"
        assert isinstance(p, dict)
        self.parameters_ = p

    def extract_A_known(self, A_known, var_subset, output_type="csc"):
        """
        Extract from the list of constraint matrices only the ones that
        touch only a subset of var_subset.
        """
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

    def get_param_idx_dict(self, var_subset=None):
        """
        Give the current subset of variables, extract the parameter dictionary to use.
        Example:
            var_subset = ['z_0', 'x_1']
            -> Parameters to include:  self.HOM (always), p_0, p_1
        - if param_level == 'no': {'l': 0}
        - if param_level == 'p': {'l': 0, 'p_0:0': 1, ..., 'p_0:d-1': d}
        - if param_level == 'ppT': {'l': 0, 'p_0:0.p_0:0': 1, ..., 'p_0:d-1:.p_0:d-1': 1}
        """
        # TODO(FD) change to use  param_subset instead.
        param_subset = [self.HOM] + [
            f"p_{i}" for i in self.get_variable_indices(var_subset)
        ]
        return unroll(self.get_param_dict(param_subset))

    def get_p(self, parameters=None, var_subset=None):
        """
        :param parameters: list of all parameters
        :param var_subset: subset of variables tat we care about (will extract corresponding parameters)
        """
        raise ValueError("deprecated, need to implement get_p from now on.")
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        if self.param_level == "no":
            return np.array([1.0])

        parameters_idx = self.get_variable_indices(var_subset)
        if len(parameters_idx):
            all_p = parameters[1:].reshape((self.n_parameters, self.d))
            sub_p = np.r_[1.0, all_p[parameters_idx, :].flatten()]
            if self.param_level == "p":
                return sub_p
            elif self.param_level == "ppT":
                return upper_triangular(sub_p)
        else:
            return np.array([1.0])

    @staticmethod
    def get_vec(mat, correct=True, sparse=False):
        """Convert NxN Symmetric matrix to (N+1)N/2 vectorized version that preserves inner product.

        :param mat: (spmatrix or ndarray) symmetric matrix
        :return: ndarray
        """
        from copy import deepcopy

        mat = deepcopy(mat)
        if correct:
            if isinstance(mat, sp.spmatrix) or isinstance(mat, sp.sparray):
                ii, jj = mat.nonzero()
                mat[ii, jj] *= np.sqrt(2.0)
                diag = ii == jj
                mat[ii[diag], jj[diag]] /= np.sqrt(2)
            else:
                mat *= np.sqrt(2.0)
                mat[range(mat.shape[0]), range(mat.shape[0])] /= np.sqrt(2)
        if sparse:
            # flat_indices = np.ravel_multi_index([i_upper, j_upper], mat.shape)
            ii, jj = mat.nonzero()
            if len(ii) == 0:
                # got an empty matrix -- this can happen depending on the parameter values.
                return None
            triu_mask = jj >= ii

            flat_indices = BaseClass.ravel_multi_index_triu(
                [ii[triu_mask], jj[triu_mask]], mat.shape
            )
            data = np.array(mat[ii[triu_mask], jj[triu_mask]]).flatten()
            vec_size = int(mat.shape[0] * (mat.shape[0] + 1) / 2)
            return sp.csr_matrix(
                (data, ([0] * len(flat_indices), flat_indices)), (1, vec_size)
            )
        else:
            return np.array(mat[np.triu_indices(n=mat.shape[0])]).flatten()

    def get_mat(self, vec, sparse=False, var_dict=None, correct=True):
        """Convert (N+1)N/2 vectorized matrix to NxN Symmetric matrix in a way that preserves inner products.

        In particular, this means that we divide the off-diagonal elements by sqrt(2).

        :param vec (ndarray): vector of upper-diagonal elements
        :return: symmetric matrix filled with vec.
        """
        # len(vec) = k = n(n+1)/2 -> dim_x = n =
        if var_dict is None:
            pass

        elif not isinstance(var_dict, dict):
            var_dict = {k: v for k, v in self.var_dict.items() if k in var_dict}

        Ai = BaseClass.create_symmetric(vec, correct=correct, sparse=sparse)
        if var_dict is None:
            return Ai

        # if var_dict is not None, then Ai corresponds to the subblock
        # defined by var_dict, of the full constraint matrix.
        Ai_poly, __ = PolyMatrix.init_from_sparse(Ai, var_dict, unfold=True)
        from poly_matrix.poly_matrix import augment

        augment_var_dict = augment(self.var_dict)
        all_var_dict = {key[2]: 1 for key in augment_var_dict.values()}
        return Ai_poly.get_matrix(all_var_dict)

    @staticmethod
    def get_labels(p, zi, zj, var_dict):
        labels = []
        size_i = var_dict[zi]
        size_j = var_dict[zj]
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

    def var_list_row(self, var_subset=None, force_parameters_off=False):
        if var_subset is None:
            var_subset = list(self.var_dict.keys())
        elif isinstance(var_subset, dict):
            var_subset = list(var_subset.keys())

        label_list = []
        if force_parameters_off:
            param_dict = {self.HOM: 0}
        else:
            param_dict = unroll(self.param_dict)  # self.get_param_idx_dict(var_subset)
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

    def var_dict_row(self, var_subset=None, force_parameters_off=False):
        return {
            label: 1
            for label in self.var_list_row(
                var_subset, force_parameters_off=force_parameters_off
            )
        }

    def get_basis_from_poly_rows(self, basis_poly_list, var_subset=None):
        var_dict = self.get_var_dict(var_subset=var_subset)

        all_dict = {label: 1 for label in self.var_list_row(var_subset)}
        basis_reduced = np.empty((0, len(all_dict)))
        for i, bi_poly in enumerate(basis_poly_list):
            # test that this constraint holds

            bi = bi_poly.get_matrix(([self.HOM], all_dict))

            if bi.shape[1] == self.get_dim_X(var_subset) * self.get_dim_P():
                ai = self.get_reduced_a(bi, var_subset=var_subset)
                Ai = self.get_mat(ai, var_dict=var_dict)
            elif bi.shape[1] == self.get_dim_X():
                Ai = self.get_mat(bi, var_subset=var_subset)

            err, idx = self.test_constraints([Ai], errors="print")
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

    def get_vector_dense(self, poly_row_sub):
        # complete the missing variables
        var_dict = self.var_dict_row()
        poly_row_all = poly_row_sub.get_matrix(
            ([self.HOM], var_dict), output_type="poly"
        )
        vector = np.empty(0)
        for param in self.get_param_idx_dict().keys():
            # extract each block corresponding to a bigger matrix
            sub_mat = PolyMatrix(symmetric=True)
            for vari, varj in itertools.combinations_with_replacement(self.var_dict, 2):
                val = poly_row_all[self.HOM, f"{param}.{vari}.{varj}"]
                if np.ndim(val) > 0:
                    if vari != varj:
                        sub_mat[vari, varj] = val.reshape(
                            self.var_dict[vari], self.var_dict[varj]
                        )
                    else:
                        mat = self.create_symmetric(val, correct=False)
                        sub_mat[vari, varj] = mat
                elif val != 0:
                    sub_mat[vari, varj] = val
            mat = sub_mat.get_matrix(self.var_dict).toarray()
            vector = np.r_[vector, mat[np.triu_indices(mat.shape[0])]]
        return np.array(vector)

    def convert_b_to_Apoly(self, new_template, var_dict):
        ai_sub = self.get_reduced_a(new_template, var_dict, sparse=True)
        Ai_sparse = self.get_mat(ai_sub, var_dict=var_dict, sparse=True)
        Ai, __ = PolyMatrix.init_from_sparse(Ai_sparse, self.var_dict, unfold=True)
        return Ai

    def convert_polyrow_to_Apoly(self, poly_row, correct=True):
        """Convert poly-row to reduced a.

        poly-row has elements with keys "pk:l.xi:m.xj:n",
        meaning this element corresponds to the l-th element of parameter i,
        and the m-n-th element of xj times xk.
        """
        parameters = self.get_p()
        param_dict = dict(zip(unroll(self.param_dict), parameters))

        poly_mat = PolyMatrix(symmetric=True)
        for key in poly_row.variable_dict_j:
            param, var_keys = key.split("-")
            param_val = param_dict[param]

            keyi_m, keyj_n = var_keys.split(".")
            m = keyi_m.split(":")[-1]
            n = keyj_n.split(":")[-1]

            # divide off-diagonal elements by sqrt(2)
            newval = poly_row[self.HOM, key] * param_val
            if correct and not ((keyi_m == keyj_n) and (m == n)):
                newval /= np.sqrt(2)

            poly_mat[keyi_m, keyj_n] += newval
        return poly_mat

    def convert_polyrow_to_Asparse(self, poly_row, var_subset=None):
        poly_mat = self.convert_polyrow_to_Apoly(poly_row, correct=False)

        var_dict = self.get_var_dict(var_subset)
        mat_var_list = []
        for var, size in var_dict.items():
            if size == 1:
                mat_var_list.append(var)
            else:
                mat_var_list += [f"{var}:{i}" for i in range(size)]
        mat_sparse = poly_mat.get_matrix({m: 1 for m in mat_var_list})
        return mat_sparse

    def convert_polyrow_to_a(self, poly_row, var_subset=None, sparse=False):
        """Convert poly-row to reduced a.

        poly-row has elements with keys "pk:l.xi:m.xj:n",
        meaning this element corresponds to the l-th element of parameter i,
        and the m-n-th element of xj times xk.
        """
        mat_sparse = self.convert_polyrow_to_Asparse(poly_row, var_subset)
        return self.get_vec(mat_sparse, correct=False, sparse=sparse)

    def convert_a_to_polyrow(
        self,
        a,
        var_subset=None,
    ) -> PolyMatrix:
        """Convert a array to poly-row."""
        if var_subset is None:
            var_subset = self.var_dict
        var_dict = self.get_var_dict(var_subset)
        dim_X = self.get_dim_X(var_subset)

        try:
            dim_a = len(a)
        except Exception:
            dim_a = a.shape[1]
        assert dim_a == dim_X

        mat = self.create_symmetric(a, sparse=True)
        poly_mat, __ = PolyMatrix.init_from_sparse(mat, var_dict)
        poly_row = PolyMatrix(symmetric=False)
        for keyi, keyj in itertools.combinations_with_replacement(var_dict, 2):
            if keyi in poly_mat.matrix and keyj in poly_mat.matrix[keyi]:
                val = poly_mat.matrix[keyi][keyj]
                labels = self.get_labels(self.HOM, keyi, keyj, self.var_dict)
                if keyi != keyj:
                    vals = val.flatten()
                else:
                    # TODO: use get_vec instead?
                    vals = val[np.triu_indices(val.shape[0])]
                assert len(labels) == len(vals)
                for label, v in zip(labels, vals):
                    if np.any(np.abs(v) > self.EPS_SPARSE):
                        poly_row[self.HOM, label] = v
        return poly_row

    def convert_b_to_polyrow(
        self, b, var_subset, param_subset=None, tol=1e-10
    ) -> PolyMatrix:
        """Convert (augmented) b array to poly-row."""
        if isinstance(b, PolyMatrix):
            raise NotImplementedError(
                "can't call convert_b_to_polyrow with PolyMatrix yet."
            )

        assert len(b) == self.get_dim_Y(var_subset, param_subset)
        poly_row = PolyMatrix(symmetric=False)
        mask = np.abs(b) > tol

        # get the variable names such as p_0:0-x:0.x:4 whch corresponds to p_0[0]*x[0]*x[4]
        var_list_row = self.var_list_row(var_subset)
        assert len(b) == len(var_list_row)

        for idx in np.where(mask == True)[0]:
            poly_row[self.HOM, var_list_row[idx]] = b[idx]
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
        """Add zero padding to b vector learned for a subset of variables
        (e.g. as obtained from learning method)
        """
        var_dict = self.get_var_dict(var_subset)
        if target_subset is None:
            target_subset = self.var_dict
        dim_X = self.get_dim_X(var_subset)
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
            mat_small = self.create_symmetric(block)
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
        #    (["h"], row_target_dict), output_type="dense"
        # ).flatten()
        # np.testing.assert_allclose(bi_all, bi_all_test)
        return bi_all

    def get_dim_x(self, var_subset=None):
        var_dict = self.get_var_dict(var_subset)
        return sum([val for val in var_dict.values()])

    def get_dim_p(self, param_subset=None):
        param_dict = self.get_param_dict(param_subset)
        return sum([val for val in param_dict.values()])

    def get_dim_Y(self, var_subset=None, param_subset=None):
        dim_X = self.get_dim_X(var_subset=var_subset)
        dim_P = self.get_dim_P(param_subset=param_subset)
        return int(dim_X * dim_P)

    def get_dim_X(self, var_subset=None):
        dim_x = self.get_dim_x(var_subset)
        return int(dim_x * (dim_x + 1) / 2)

    def get_dim_P(self, param_subset=None):
        return len(self.get_p(param_subset=param_subset))

    def get_reduced_a(self, bi, param_here=None, var_subset=None, sparse=False):
        """
        Extract first block of bi by summing over other blocks times the parameters.
        """
        if param_here is None:
            param_here = self.get_p()

        if isinstance(bi, np.ndarray):
            len_b = len(bi)
        elif isinstance(bi, PolyMatrix):
            bi = bi.get_matrix(([self.HOM], self.var_dict_row(var_subset)))
            len_b = bi.shape[1]
        else:
            # bi can be a scipy sparse matrix,
            len_b = bi.shape[1]

        n_params = self.get_dim_P()
        dim_X = self.get_dim_X(var_subset)
        n_parts = len_b / dim_X
        assert (
            n_parts == n_params
        ), f"{len_b} does not not split in dim_P={n_params} parts of size dim_X={dim_X}"

        ai = np.zeros(dim_X)
        for i, p in enumerate(param_here):
            if isinstance(bi, np.ndarray):
                ai += p * bi[i * dim_X : (i + 1) * dim_X]
            else:
                ai += p * bi[0, i * dim_X : (i + 1) * dim_X].toarray().flatten()
        if sparse:
            ai_sparse = sp.csr_array(ai[None, :])
            ai_sparse.eliminate_zeros()
            return ai_sparse
        else:
            return ai

    def augment_using_zero_padding(self, ai):
        n_parameters = self.get_dim_P()
        return np.hstack([ai, np.zeros((n_parameters - 1) * len(ai))])

    def augment_using_parameters(self, x):
        p = self.get_p()
        return np.kron(p, x)
