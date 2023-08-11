import numpy as np


def upper_triangular(p):
    # given vector, get the half kronecker product.
    return np.outer(p, p)[np.triu_indices(len(p))]


def diag_indices(n):
    # given the half kronecker product, return diagonal elements
    z = np.empty((n, n))
    z[np.triu_indices(n)] = range(int(n * (n + 1) / 2))
    return np.diag(z).astype(int)
