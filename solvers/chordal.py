import chompack as cp
import numpy as np
import scipy.sparse as sp
from cvxopt import amd, printing, spmatrix

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


def investigate_sparsity(mask, ax=None):
    I, J = mask.nonzero()
    try:
        data = mask.data.astype(float)
    except AttributeError:
        data = mask[I, J].flatten()
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
    if ax is None:
        fig1, ax1 = plt.subplots()
        ax1.matshow(mask1.toarray())
        ax1.set_title("chordal extension")
    else:
        ax1 = ax
        fig1 = plt.gcf()

    fig2, ax2 = plt.subplots()
    diff = (mask1 - mask) > 0
    try:
        ax2.matshow(diff.toarray())
    except AttributeError:
        ax2.matshow(diff)
    ax2.set_title("fill-in")

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
    num = len(symb.cliques(reordered=False))
    for i, (c, s) in enumerate(
        zip(symb.cliques(reordered=False), symb.separators(reordered=False))
    ):
        print(f"{i}, {len(c):4.0f}, {len(s):4.0f}")

        import itertools

        ri = -0.5 + i / num
        for cii, cjj in itertools.product(c, c):
            ax1.scatter(cii + ri, cjj + ri, color=f"C{i % 10}", s=5, marker="s")

        ax.scatter([i], len(c), marker="x", color="red")
        ax.scatter([i], len(s), marker="x", color="blue")

    ax.scatter([], [], color="red", marker="x", label="clique length")
    ax.scatter([], [], color="blue", marker="x", label="overlap length")
    ax.grid()
    ax.legend()
    return mask1.toarray()
