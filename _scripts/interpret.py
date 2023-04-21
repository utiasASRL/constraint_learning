# %%
import shutil

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from IPython.display import display

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

matplotlib.use("ps")

plt.rc("text.latex", preamble=r"\usepackage{bm}\usepackage{color}")
figsize = 7

pd.set_option("display.max_columns", None)

# %%
from lifters.plotting_tools import savefig
from lifters.stereo2d_lifter import Stereo2DLifter
from poly_matrix.poly_matrix import PolyMatrix

lifter = Stereo2DLifter(n_landmarks=3)
A_known = lifter.get_A_known()
A_learned_unknown = lifter.get_A_learned(method="qrp", eps=1e-7)

A_learned = lifter.get_A_learned(method="qrp", A_known=A_known, eps=1e-7)
assert (A_known[-1] != A_learned[-1]).nnz == 0
assert len(A_learned) == len(A_learned_unknown)


data = []
for A in A_learned:
    A_poly, var_dict = PolyMatrix.init_from_sparse(A, lifter.var_dict, unfold=True)
    sparse_series = A_poly.interpret(var_dict)
    data.append(sparse_series)
df = pd.DataFrame(data, dtype="Sparse[object]")


# %%

# first, sort dataframe by values and drop na columns.


def sort_fun(series):
    return series.isna()


df.dropna(axis=1, how="all", inplace=True)
# df.take(df.index < len(A_known)).sort_values(
#    key=sort_fun,
#    by=list(df.columns),
#    axis=0,
#    na_position="last",
#    inplace=True,
# )
# display(df)
pd.reset_option("display.max_columns")

# %%

names = {0: "x", 1: "y", 2: "z"}
landmark_dict = {
    np.round(lifter.landmarks[k, d], 4): f"$\\textcolor{{red}}{{a_{k}^{names[d]}}}$"
    for k in range(lifter.n_landmarks)
    for d in range(lifter.d)
}
print(landmark_dict.keys())

strings = []
for i, row in df.iterrows():
    string = ""

    multiindex = pd.MultiIndex.from_arrays([np.zeros(len(row)), row.index])
    row.index = multiindex
    row_coo = row.astype(pd.SparseDtype(float, fill_value=0)).sparse.to_coo()[0]
    for idx in row_coo.col:
        key = row[0].index[idx]
        val = row[0][key]
        # check if there is a landmark coordinate in this value.
        key_print = "$" + key.replace(".", "\\cdot ") + "$"
        val_key = np.round(val, 4)
        if val_key in landmark_dict:
            string += f" +{landmark_dict[val_key]} {key_print}"
        elif -val_key in landmark_dict:
            string += f" -{landmark_dict[-val_key]} {key_print}"
        elif np.mod(val, 1) < 1e-10:
            # string += f" + {val_key:.2f} {key}"
            string += f" {val:+.0f} {key_print}"
        elif np.mod(val * 10, 1) < 1e-10:
            # string += f" + {val_key:.2f} {key}"
            string += f" {val:+.1f} {key_print}"
        else:
            print("random:", val)
            string += f" +$\\alpha$ {key_print}"

    string = string[1:]
    print(string)
    strings.append(string + "\n")

# %%
fname = f"_results/{lifter}_constraints.txt"
with open(fname, "w") as f:
    f.writelines(strings)

print(f"wrote as {fname}")

# %%
from lifters.plotting_tools import savefig

fig, ax = plt.subplots()
space = 0.04
for h, string in enumerate(strings[::-1]):
    ax.text(0.0, h * space, string, fontsize=6)
# ax.plot([0.0, 0.0], [space, h * space])
ax.axis("off")
plt.tight_layout()
savefig(fig, f"_results/{lifter}_constraints.ps")
savefig(fig, f"_results/{lifter}_constraints.png")

# %%
