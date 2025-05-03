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
