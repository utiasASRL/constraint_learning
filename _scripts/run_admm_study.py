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
    overwrite = False
    n_params_list = np.logspace(1, 3, 3).astype(int)
    n_threads_list = np.logspace(0, 7, 8, base=2).astype(int)
    appendix = "admm"

    np.random.seed(0)
    costs_all = []
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
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                n_threads_list=n_threads_list,
                fname=fname,
                use_methods=USE_METHODS,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

    for lifter in [lifter_ro, lifter_mat]:
        fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
        df = pd.read_pickle(fname)
        n_threads_list = sorted(
            [
                int(l.strip("t pADMM-"))
                for l in df.columns
                if l.startswith("t pADMM-") and ("total" not in l)
            ]
        )

        labels = (
            [f"t pADMM-{n}" for n in n_threads_list]
            if len(n_threads_list) > 1
            else ["t pADMM"]
        )
        df = df[df["n params"] > 10]
        df_long = df.melt(
            id_vars=["n params", "n threads"],
            value_vars=labels,
            value_name="time",
            var_name="label",
        )
        df_long.loc[:, "n threads"] = df_long.apply(
            lambda row: int(row["label"].strip("t pADMM-")), axis=1
        )

        MARKERS = {100: "x", 1000: "+"}
        COLORS = {100: "C0", 1000: "C1"}
        fig, ax = plt.subplots()
        fig.set_size_inches(3.5, 3)
        for n_params, df_plot in df_long.groupby("n params"):
            ax.scatter(
                df_plot["n threads"],
                df_plot["time"],
                marker=MARKERS[n_params],
                color=COLORS[n_params],
                label=f"N={n_params}",
            )

        ax.set_xlabel("number of threads")
        ax.set_ylabel("time [s]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_ylim(1, 1e)
        ax.legend(loc="upper right", ncol=3, columnspacing=0.1)
        ax.grid()
        savefig(fig, fname.replace(".pkl", f"_time.png"))

        n_threads = n_threads_list[0]
        fig, ax = plt.subplots()
        fig.set_size_inches(3.5, 3)
        cost_history = df[["n params", f"cost history pADMM-{n_threads}", "cost local"]]
        for n_params, df_plot in cost_history.groupby("n params"):
            assert len(df_plot) == 1
            row = df_plot.iloc[0]
            cost_history = (
                np.abs(row[f"cost history pADMM-{n_threads}"] - row["cost local"])
                / row["cost local"]
            )
            ax.scatter(
                np.arange(1, len(cost_history) + 1),
                cost_history,
                label=f"N={n_params}",
                color=COLORS[n_params],
                marker=MARKERS[n_params],
            )
        ax.set_ylabel("relative error")
        ax.set_xlabel("iteration index")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.grid()
        savefig(fig, fname.replace(".pkl", f"_conv.png"))

        n_threads = n_threads_list[0]
        cost_history = df[["n params", f"cost history pADMM-{n_threads}"]]
        cost_history.loc[:, "lifter"] = str(lifter)
        costs_all.append(cost_history)

    df_all = pd.concat(costs_all)
    fig, ax = plt.subplots()
    for (l, n_params), df_plot in df_all.groupby(["lifter", "n params"], sort=False):
        assert len(df_plot) == 1
        row = df_plot.iloc[0]
        ax.plot(row[f"cost history pADMM-{n_threads}"], label=f"{l}-{n_params}")
    ax.set_ylabel("cost")
    ax.set_xlabel("it")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    savefig(fig, f"{RESULTS_READ}/admm_convergence.png")
