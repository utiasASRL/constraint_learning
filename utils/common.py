import numpy as np


def upper_triangular(p):
    """Given vector, get the half kronecker product."""
    return np.outer(p, p)[np.triu_indices(len(p))]


def diag_indices(n):
    """Given the half kronecker product, return diagonal elements"""
    z = np.empty((n, n))
    z[np.triu_indices(n)] = range(int(n * (n + 1) / 2))
    return np.diag(z).astype(int)


def rank_project(X, p=1, tol=1e-10):
    """Project X to matrices of rank p"""
    E, V = np.linalg.eigh(X)
    if p is None:
        p = np.sum(np.abs(E) > tol)
    x = V[:, -p:] * np.sqrt(E[-p:])

    X_hat = np.outer(x, x)
    info = {
        "error X": np.linalg.norm(X_hat - X), 
        "error eigs": np.sum(np.abs(E[:p]))
    }
    return x, info
    
