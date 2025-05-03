import itertools

import numpy as np
import scipy.sparse as sp


def upper_triangular(p):
    """Given vector, get the half kronecker product."""
    return np.outer(p, p)[np.triu_indices(len(p))]


def diag_indices(n):
    """Given the half kronecker product, return diagonal elements"""
    z = np.empty((n, n))
    z[np.triu_indices(n)] = range(int(n * (n + 1) / 2))
    return np.diag(z).astype(int)


def get_aggregate_sparsity(matrix_list_sparse):
    agg_ii = []
    agg_jj = []
    for i, A_sparse in enumerate(matrix_list_sparse):
        assert isinstance(A_sparse, sp.spmatrix)
        ii, jj = A_sparse.nonzero()
        agg_ii += list(ii)
        agg_jj += list(jj)
    return sp.csr_matrix(([1.0] * len(agg_ii), (agg_ii, agg_jj)), A_sparse.shape)


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


def create_symmetric(vec, eps_sparse, correct=False, sparse=False):
    """Create a symmetric matrix from the vectorized elements of the upper half"""

    def get_dim_x(len_vec):
        return int(0.5 * (-1 + np.sqrt(1 + 8 * len_vec)))

    try:
        # vec is dense
        len_vec = len(vec)
        dim_x = get_dim_x(len_vec)
        triu = np.triu_indices(n=dim_x)
        mask = np.abs(vec) > eps_sparse
        triu_i_nnz = triu[0][mask]
        triu_j_nnz = triu[1][mask]
        vec_nnz = vec[mask]
    except Exception:
        # vec is sparse
        len_vec = vec.shape[1]
        dim_x = get_dim_x(len_vec)
        vec.data[np.abs(vec.data) < eps_sparse] = 0
        vec.eliminate_zeros()
        ii, jj = vec.nonzero()  # vec is 1 x jj
        triu_i_nnz, triu_j_nnz = unravel_multi_index_triu(jj, (dim_x, dim_x))
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

        flat_indices = ravel_multi_index_triu([ii[triu_mask], jj[triu_mask]], mat.shape)
        data = np.array(mat[ii[triu_mask], jj[triu_mask]]).flatten()
        vec_size = int(mat.shape[0] * (mat.shape[0] + 1) / 2)
        return sp.csr_matrix(
            (data, ([0] * len(flat_indices), flat_indices)), (1, vec_size)
        )
    else:
        return np.array(mat[np.triu_indices(n=mat.shape[0])]).flatten()


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
