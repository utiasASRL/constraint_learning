import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

# USE_METHODS = ["SDP", "dSDP", "ADMM"]
# USE_METHODS = ["local", "dSDP", "ADMM", "pADMM"]
# USE_METHODS = ["ADMM", "pADMM"]
USE_METHODS = [
    "local",
    "SDP",
    "SDP-redun",
    "dSDP",
    "dSDP-redun",
    "pADMM",
    "pADMM-redun",
]
ADD_REDUNDANT = True

RESULTS_READ = "_results_server"
RESULTS_WRITE = "_results"

if __name__ == "__main__":
    overwrite = True
    n_threads_list = [10]

    # n_params_list = np.logspace(1, 2, 10).astype(int)
    # appendix = "time"

    # goes up to 3000 (after that we have memory error).
    n_params_list = np.logspace(1, 6, 21).astype(int)[:11]
    appendix = "alltime"

    # n_params_list = [10, 20, 30]
    # n_threads_list = [2]
    # appendix = "test"

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    for lifter in [lifter_ro, lifter_mat]:
        try:
            assert overwrite is False
            fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
            df = pd.read_pickle(fname)
        except (FileNotFoundError, AssertionError):
            fname = f"{RESULTS_WRITE}/{lifter}_{appendix}.pkl"
            add_redundant_constr = (
                True if isinstance(lifter, MatWeightLocLifter) else False
            )
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                fname=fname,
                use_methods=USE_METHODS,
                noise_list=[lifter.NOISE],
                add_redundant_constr=add_redundant_constr,
                n_threads_list=n_threads_list,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

    for lifter in [lifter_mat, lifter_ro]:
        fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
        df = pd.read_pickle(fname)
        for label, plot in zip(["t", "cost"], [sns.scatterplot, sns.barplot]):
            value_vars = [f"{label} {m}" for m in USE_METHODS]
            value_vars = set(value_vars).intersection(df.columns.unique())
            df_long = df.melt(
                id_vars=["n params"],
                value_vars=value_vars,
                value_name=label,
                var_name="solver type",
            )
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 4)
            plot(df_long, x="n params", y=label, hue="solver type", ax=ax)
            ax.set_yscale("log")
            if label != "cost":
                ax.set_xscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}.png"))
