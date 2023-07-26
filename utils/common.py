import numpy as np

import os.path


def get_fname(name, extension="pkl"):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_results", f"{name}.{extension}")
    )


def upper_triangular(p):
    # given vector, get the half kronecker product.
    return np.outer(p, p)[np.triu_indices(len(p))]

def diag_indices(n):
    # given the half kronecker product, return diagonal elements
    z = np.empty((n, n))
    z[np.triu_indices(n)] = range(int(n * (n+1)/2))
    return np.diag(z).astype(int)

def increases_rank(mat, new_row):
    import scipy.sparse as sp
    # TODO(FD) below is not the most efficient way of checking lin. indep.
    if mat is None:
        return True
    try:
        new_row = new_row.flatten()
        mat_test = np.vstack([mat, new_row[None, :]])
        new_rank = np.linalg.matrix_rank(mat_test)
    except:
        raise ValueError("don't use this function, it's inefficient.")
        return True

    # if the new matrix is not full row-rank then the new row was
    # actually linealy dependent.
    if new_rank == mat_test.shape[0]:
        return True
    return False


def setup_tex():
    import shutil
    import matplotlib.pylab as plt

    usetex = True if shutil.which("latex") else False
    print("found latex:", usetex)
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )
    import matplotlib

    # matplotlib.use("ps")
    plt.rc("text.latex", preamble=r"\usepackage{bm}\usepackage{color}")
    return plt
