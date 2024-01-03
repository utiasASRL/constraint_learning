from cvxopt import spmatrix, printing
from cvxopt import amd
import chompack as cp
import numpy as np
import scipy.sparse as sp

from utils.plotting_tools import import_plt

plt = import_plt()

printing.options["dformat"] = "%3.1f"


def symb2scipy(symb, shape, reordered=False):
    pattern = symb.sparsity_pattern(reordered=reordered, symmetric=True)
    I = np.array(pattern.I).flatten()
    J = np.array(pattern.J).flatten()
    return sp.csr_array((np.ones(len(I)), (I, J)), shape=shape, dtype=float)


def get_aggregate_sparsity(Q, A_b_list):
    return (np.abs(Q) >= 1e-10) + sum([np.abs(A_b[0]) >= 1e-10 for A_b in A_b_list])


def investigate_sparsity(mask):
    I, J = mask.nonzero()
    data = mask.data.astype(float)
    diag = I[I == J]
    missing = list(set(range(mask.shape[0])).difference(diag))
    I = np.hstack([I, missing]).astype(int)
    J = np.hstack([J, missing]).astype(int)
    data = np.hstack([data, np.zeros(len(missing))])
    M = spmatrix(data, I, J, mask.shape)

    p = cp.maxcardsearch(M)
    # p = amd.order(M)

    symb = cp.symbolic(M, p=p)  # amd.order)
    L = cp.cspmatrix(symb)
    L += M

    mask1 = symb2scipy(symb, mask.shape, reordered=False)
    plt.matshow(mask1.toarray())
    plt.title("filled-in mask")

    diff = (mask1 - mask) > 0
    plt.matshow(diff.toarray())
    plt.title("fill-in")

    if cp.peo(M, p):
        print("mask was already chordal")
    else:
        print("mask was not chordal!")
        print("fill in:")
        print(symb.fill)

    print("cliques:")
    print("i, len clique, len overlap")
    fig, ax = plt.subplots()
    ax.axhline(mask.shape[0], color="k", label="number of nodes")
    for i, (c, s) in enumerate(zip(symb.cliques(), symb.separators())):
        print(f"{i}, {len(c):4.0f}, {len(s):4.0f}")
        ax.scatter([i], len(c), marker="x", color="red")
        ax.scatter([i], len(s), marker="x", color="blue")

    ax.scatter([], [], color="red", marker="x", label="clique length")
    ax.scatter([], [], color="blue", marker="x", label="overlap length")
    ax.grid()
    ax.legend()
    return fig, ax