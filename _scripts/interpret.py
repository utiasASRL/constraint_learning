# %%
import pandas as pd
from utils.common import setup_tex

plt = setup_tex()

pd.set_option("display.max_columns", None)
figsize = 7

# %%
from lifters.plotting_tools import savefig
from lifters.stereo2d_lifter import Stereo2DLifter

lifter = Stereo2DLifter(n_landmarks=3)

# TODO(FD) get dataframe from leanred matrices (see Learner for an up-to-date implementation)
# df =

from lifters.interpret import convert_series_to_string, get_known_variables

landmark_dict = get_known_variables(lifter)
strings = []
for i, row in df.iterrows():
    string = convert_series_to_string(row, landmark_dict=landmark_dict)
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
pd.reset_option("display.max_columns")
