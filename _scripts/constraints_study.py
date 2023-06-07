# %%
# intialization

from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from lifters.plotting_tools import savefig
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from solvers.common import find_local_minimum, solve_sdp_cvxpy
from solvers.sparse import solve_lambda

# assume strong duality when absolute gap is smaller
TOL_ABS_GAP = 1e-5

# tolerance for nullspace basis vectors
EPS_LEARNED = 1e-7

# SOLVER = "CVXOPT"
SOLVER = "MOSEK"
NORMALIZE = False
METHOD = "qrp"

# remove Q[0,0] for optimization, and normalize entries
# TODO(FD) for some reason this is not helping for 1d problem
ADJUST = False

n_landmarks = 3
noise = 1e-2
level = "urT"
seed = 0
d = 3

save_pairs = [
    ("original", "increase"),
    ("original", "decrease"),
    ("optimization", "decrease"),
    ("opt-force", "decrease"),
    # ("singular", "decrease"),
    # ("sparsity", "increase"),
]

parameter_list = [
    dict(  # fully learned
        add_known_redundant=False,
        use_known=True,
        incremental=False,
        appendix="learned",
    ),
    dict(  # fully known
        add_known_redundant=True,
        use_known=True,
        incremental=False,
        appendix="known",
    ),
    dict(  # incremental
        add_known_redundant=False,
        use_known=False,
        incremental=True,
        appendix="incremental",
    ),
]

params = parameter_list[2]
# for params in parameter_list:

root = Path(__file__).resolve().parents[1]
fname_root = str(root / f"_results/experiments_{d}d_{params['appendix']}")

if d == 1:
    lifter = Stereo1DLifter(n_landmarks=n_landmarks)
elif d == 2:
    lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level)
elif d == 3:
    lifter = Stereo3DLifter(n_landmarks=n_landmarks, level=level)

# %%
# generate constraints
np.random.seed(seed)
lifter.generate_random_setup()
if (params["use_known"]) and (not params["incremental"]):
    A_known = lifter.get_A_known(add_known_redundant=params["add_known_redundant"])
else:
    A_known = []

A_all, S = lifter.get_A_learned(
    A_known=A_known,
    eps=EPS_LEARNED,
    normalize=NORMALIZE,
    return_S=True,
    method=METHOD,
    plot=False,
    incremental=params["incremental"],
)
errs, idxs = lifter.test_constraints(A_all, errors="print")
for idx in idxs:
    del A_all[idx]
    del S[idx]
n_learned = len(A_all) - len(A_known)

# %%
# solve locally
names = [f"A{i}:known" for i in range(len(A_known))]
names += [f"A{len(A_known) + i}:learned" for i in range(n_learned)]

np.random.seed(seed)
Q, y = lifter.get_Q(noise=noise)
# increase how many constraints we add to the problem
qcqp_that, qcqp_cost = find_local_minimum(lifter, y=y)
if qcqp_cost is None:
    print("Warning: could not solve local.")
elif qcqp_cost < 1e-7:
    print("Warning: too low qcqp cost, numerical issues.")


# %%
# compute lamdas using optimization
A_b_list_all = lifter.get_A_b_list(A_all)
xhat = lifter.get_x(qcqp_that)
H, lamdas_force = solve_lambda(Q, A_b_list_all, xhat, force_first=1 + len(A_known))
if lamdas_force is None:
    print("Warning: problem doesn't have feasible solution!")
    lamdas_force = np.zeros(len(A_b_list_all))
else:
    lamdas_force = np.abs(lamdas_force)

H, lamdas = solve_lambda(Q, A_b_list_all, xhat, force_first=1)
if lamdas is None:
    print("Warning: problem doesn't have feasible solution!")
    lamdas = np.zeros(len(A_b_list_all))
else:
    lamdas = np.abs(lamdas)

plt.figure()
plt.semilogy(lamdas_force, label=f"forced {1 + len(A_known)}")
plt.axvline(1 + len(A_known), color="k", ls=":")
plt.semilogy(lamdas, label="free")
plt.legend()

# %%
# compute lambas by solving (tight) dual problem
X, info = solve_sdp_cvxpy(Q, A_b_list_all, adjust=ADJUST)  # , rho_hat=qcqp_cost)
if info["cost"] is None:
    print("Warning: infeasible?")
    yvals = np.zeros(len(A_b_list_all))
else:
    yvals = np.abs(info["yvals"])
    if abs(qcqp_cost - info["cost"]) > TOL_ABS_GAP:
        print("Warning: no strong duality.")
        print(f"qcqp cost: {qcqp_cost}")
        print(f"dual cost: {info['cost']}")


# %%
# compute other scores
# compute sparsity score
sparsities = np.array([A.nnz for A in A_all])

# compute other scores?
# will order decreasingly
order_dicts = {
    "original": np.arange(len(A_all)),
    "sparsity": sparsities,
    "optimization": lamdas[1:],
    "opt-force": lamdas_force[1:],
    "singular": S,
}
results = {
    "dual": yvals[1:],
    "name": names,
    "seed": np.full(len(names), seed),
}
results.update(order_dicts)
df = pd.DataFrame(results)
if fname_root != "":
    fname = fname_root + "_values.pkl"
    pd.to_pickle(df, fname)
    print("saved values as", fname)

# %%
# compute tightness

from progressbar import ProgressBar

# order_dicts = {"sparsity": sparsities}
# create tightness study for this particular case, mostly as a sanity check.
data_tight = []

# for (order, order_type), df_sub in df_tight.groupby(["order", "type"]):
for order_name, order_type in save_pairs:
    print(f"{order_name} {order_type}")
    # for order_name, order_arrays in order_dicts.items():
    step = 1 if order_type == "increase" else -1
    order_arrays = order_dicts[order_name]

    order = np.argsort(order_arrays)[::step]
    A_b_here = [A_b_list_all[0]] + [A_b_list_all[s + 1] for s in order]
    names_here = ["A0"] + [names[s] for s in order]
    values_here = [1] + [order_arrays[s] for s in order]

    tight_counter = 0
    min_number = None

    p = ProgressBar(max_value=len(order + 2))
    for i in range(1, len(order) + 1):
        p.update(i)
        X, info = solve_sdp_cvxpy(Q, A_b_here[: i + 1], adjust=ADJUST)

        tight = False
        if info["cost"] is not None:
            error = abs(qcqp_cost - info["cost"])
            if error < TOL_ABS_GAP:
                tight = True
                if min_number is None:
                    min_number = i
                tight_counter += 1
        else:
            print(f"Did not solve at constraint {i}")
        if tight_counter > 10:
            break

        # sanity check on ordering
        if order_name == "sparsity":
            assert A_b_here[i][0].nnz == values_here[i]
        data_tight.append(
            {
                "order_name": order_name,
                "value": values_here[i],
                "H": info["H"],
                "A": A_b_here[i][0],
                "order_type": order_type,
                "cost": info["cost"],
                "tight": tight,
                "name": names_here[i],
            }
        )

df_tight = pd.DataFrame(data_tight)

# %%
if fname_root != "":
    fname = fname_root + "_tight.pkl"
    pd.to_pickle(df_tight, fname)
    print("saved values as", fname)

# %%
# plot the tightness for different orders
fig, ax = plt.subplots()
ax.axhline(qcqp_cost, color="k")
for (order_name, order_type), df_sub in df_tight.groupby(["order_name", "order_type"]):
    try:
        number = np.where(df_sub.tight.values)[0][0]
        label = f"{order_name} {order_type}: {number}"
    except IndexError:
        number = None
        label = f"{order_name} {order_type}: never tight"
    ax.semilogy(range(1, len(df_sub) + 1), df_sub.cost.values, label=label)
ax.legend()
ax.grid()
if fname_root != "":
    savefig(fig, fname_root + f"_tightness.png")

from math import ceil

# %%
# plot the matrices
from lifters.plotting_tools import plot_matrix

# for (order, order_type), df_sub in df_tight.groupby(["order", "type"]):
for order_name, order_type in save_pairs:
    df_sub = df_tight[
        (df_tight.order_name == order_name) & (df_tight.order_type == order_type)
    ]

    n_cols = 10
    n_rows = ceil(len(df_sub) / n_cols)
    fig, axs = plt.subplots(n_rows * 2, n_cols)
    fig.set_size_inches(n_cols, n_rows * 2)
    for j, matrix_type in enumerate(["A", "H"]):
        matrices = df_sub[matrix_type].values
        names_here = df_sub.name.values
        costs = df_sub.cost.values
        values = df_sub.value.values

        # make sure it's symmetric
        vmin = np.min([np.min(A) for A in matrices if (A is not None)])
        vmax = np.max([np.max(A) for A in matrices if (A is not None)])
        vmin = min(vmin, -vmax)
        vmax = max(vmax, -vmin)

        import itertools
        from math import ceil

        i = 0
        for row, col in itertools.product(range(n_rows), range(n_cols)):
            ax = axs[row * 2 + j, col]
            if i < len(matrices):
                if matrices[i] is None:
                    continue
                title = f"{matrix_type}{i}"
                if matrix_type == "A":
                    title += f"\n{names_here[i]}"
                else:
                    title += f"\nc={costs[i]:.2e}\nv={values[i]:.2e}"

                plot_matrix(
                    ax=ax,
                    Ai=matrices[i],
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    colorbar=False,
                )
                ax.set_title(title, fontsize=5)
            else:
                ax.axis("off")
            i += 1

    fname = fname_root + f"_{order_name}_{order_type}.pdf"
    savefig(fig, fname)

    fname = fname_root + f"_{order_name}_{order_type}.png"
    savefig(fig, fname)
    # f_root = fname_root + f"_{order.replace(' ','_')}_{order_type}"
    # partial_plot_and_save(
    #    lifter, Q=None, A_list=matrices, fname_root=f_root, title=title
    # )
    # fig, ax = plot_matrices(
    #    df_sub["matrix_type"].values, title=matrix_type, vmin=vmin, vmax=vmax
    # )

# H[0, 0] = 0.0
# figi, axi = plot_matrix(H, nticks=3, title=title, vmin=vmin, vmax=vmax)


# %%
# create a new dataframe that is more readible than the expressions.
from lifters.interpret import get_known_variables
from poly_matrix.poly_matrix import PolyMatrix

# expressions = []
landmark_dict = get_known_variables(lifter)
print("known values:", landmark_dict.keys())
data_math = []
for i, A in enumerate(A_all):
    # print(names[i])
    try:
        A_poly, var_dict = PolyMatrix.init_from_sparse(A, lifter.var_dict, unfold=True)
    except Exception as e:
        print(f"error at {i}:", e)
        continue
    sparse_series = A_poly.interpret(var_dict)
    data_math.append(sparse_series)

df_math = pd.DataFrame(data=data_math, dtype="Sparse[object]")
for name, values in order_dicts.items():
    df_math.loc[:, name] = values
df_math.loc[:, "name"] = names


def sort_fun(series):
    return series.isna()


df_math.dropna(axis=1, how="all", inplace=True)
df_math.sort_values(
    key=sort_fun,
    by=list(df_math.columns),
    axis=0,
    na_position="last",
    inplace=True,
)

if fname_root != "":
    fname = fname_root + "_math.pkl"
    pd.to_pickle(df_math, fname)
    print("saved math as", fname)
    fname = fname_root + "_math.csv"
    df_math.to_csv(fname)
    print("saved math as", fname)


# %%
