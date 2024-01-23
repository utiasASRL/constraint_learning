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
USE_METHODS = ["dSDP", "ADMM"]

if __name__ == "__main__":
    # n_params_list = np.logspace(1, 2, 10).astype(int)
    # appendix = "time"

    # n_params_list = np.logspace(1, 3, 9).astype(int)
    # appendix = "large"

    n_params_list = np.logspace(3, 6, 9).astype(int)
    appendix = "beyond"

    # n_params_list = [10, 20]
    # appendix = "test"
    overwrite = True

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    # for lifter in [lifter_ro, lifter_mat]:
    for lifter in [lifter_mat]:
        lifter.ALL_PAIRS = False
        lifter.CLIQUE_SIZE = 2

        fname = f"_results/{lifter}_{appendix}.pkl"

        try:
            assert overwrite is False
            df = pd.read_pickle(fname)
        except (FileNotFoundError, AssertionError):
            df = generate_results(lifter, n_params_list=n_params_list, fname=fname)
            df.to_pickle(fname)
            print("saved final as", fname)

        for label, plot in zip(["t", "cost"], [sns.scatterplot, sns.barplot]):
            value_vars = [
                f"{label} local",
                f"{label} dSDP",
                f"{label} SDP",
                f"{label} ADMM",
            ]
            value_vars = set(value_vars).intersection(df.columns.unique())
            df_long = df.melt(
                id_vars=["n params"],
                value_vars=value_vars,
                value_name=label,
                var_name="solver type",
            )
            fig, ax = plt.subplots()
            plot(df_long, x="n params", y=label, hue="solver type", ax=ax)
            ax.set_yscale("log")
            if label != "cost":
                ax.set_xscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}.png"))

        fig, ax = plt.subplots()
        value_vars = ["evr SDP", "evr dSDP", "evr ADMM"]
        value_vars = set(value_vars).intersection(df.columns.unique())
        for v in value_vars:
            ax.loglog(df["n params"], df[v], label=v.strip("evr "))
        ax.legend()
        ax.grid("on")
        ax.set_ylabel("EVR")
        ax.set_xlabel("n params")
        savefig(fig, fname.replace(".pkl", f"_evr.png"))
        print("done")
