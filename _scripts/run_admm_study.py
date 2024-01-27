import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

USE_METHODS = ["local", "pADMM"]

RESULTS_READ = "_results_server"
RESULTS_WRITE = "_results"

if __name__ == "__main__":
    # n_params_list = np.logspace(1, 6, 6).astype(int)
    n_params_list = [10, 100]
    n_threads_list = (3 * 2 ** np.arange(0, 7)).astype(int)
    appendix = "admm"
    overwrite = False

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    costs_dict = {}
    for lifter in [lifter_mat, lifter_ro]:
        try:
            assert overwrite is False
            fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
            df = pd.read_pickle(fname)
        except (FileNotFoundError, AssertionError):
            fname = f"{RESULTS_WRITE}/{lifter}_{appendix}.pkl"
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                n_threads_list=n_threads_list,
                fname=fname,
                use_methods=USE_METHODS,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

        df_long = df.melt(
            id_vars=["n params", "n threads"],
            value_vars=[f"t pADMM-{n}" for n in n_threads_list],
            var_name="time",
        )
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 4)
        sns.scatterplot(data=df_long, x="n threads", y="time", hue="n params", ax=ax)
        savefig(fig, fname.replace(".pkl", f"_time.png"))

        n_threads = n_threads_list[0]
        cost_history = df.loc[
            df["n threads"] == n_threads, [f"cost history pADMM-{n_threads}"]
        ].values
        costs_dict[lifter] = cost_history

    fig, ax = plt.subplots()
    for label, costs in costs_dict.items():
        ax.plot(costs, label=label)
    ax.set_ylabel("cost")
    ax.set_xlabel("it")
    ax.legend(loc="upper right")
    savefig(fig, fname.replace(".pkl", f"_cost.png"))
