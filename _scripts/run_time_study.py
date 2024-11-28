from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import USE_METHODS, USE_METHODS_MW, USE_METHODS_RO, savefig

ADD_REDUNDANT = True
RESULTS_DIR = "_results"

N_SEEDS = 10


def custom_plot(ax, x, y, data, **unused):
    data_methods = data["solver type"].unique()
    for method in USE_METHODS.keys():
        if method in data_methods:
            df_sub = data[data["solver type"] == method]
            with_label = deepcopy(USE_METHODS[method])
            with_label["marker"] = None
            ax.plot([], [], **with_label)

            no_label = deepcopy(USE_METHODS[method])
            no_label["marker"] = None
            no_label["label"] = None
            no_label["ls"] = ""
            ax.plot(df_sub[x], df_sub[y], **no_label)

            df_median = df_sub.groupby("n params")["t"].median()
            quantiles = np.vstack(
                [
                    np.zeros_like(df_median.values),
                    df_sub.groupby("n params")["t"].max().values - df_median.values,
                ]
            )
            no_label["marker"] = "o"
            no_label["ls"] = "-"
            ax.errorbar(df_median.index, df_median.values, yerr=quantiles, **no_label)
        else:
            continue


def plot_timing(df, xlabel="", fname="", use_methods=USE_METHODS):
    for label, plot in zip(["t", "cost"], [custom_plot, sns.barplot]):
        value_vars = [f"{label} {m}" for m in use_methods]
        value_vars = set(value_vars).intersection(df.columns.unique())
        df_long = df.melt(
            id_vars=["n params"],
            value_vars=value_vars,
            value_name=label,
            var_name="solver type",
        )
        df_long.loc[:, "solver type"] = [
            l[len(label) + 1 :] for l in df_long["solver type"]
        ]
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 3)
        plot(
            data=df_long,
            x="n params",
            y=label,
            ax=ax,
            hue="solver type",
            hue_order=use_methods,
            palette={m: USE_METHODS[m]["color"] for m in use_methods},
        )
        for group, kwargs in zip(ax.containers, USE_METHODS.values()):
            for bar in group:
                try:
                    bar.set_alpha(kwargs["alpha"])
                except:
                    continue
        ax.set_yscale("log")
        if label not in ["cost", "RDG", "error"]:
            ax.set_xscale("log")
            ax.plot(
                df_long["n params"].unique(),
                df_long[label].min() * 1e-1 * df_long["n params"].unique(),
                color="k",
                alpha=0.5,
                ls=":",
            )
            # ax.annotate(xy=(1000, 4), text="$N$", alpha=0.5)
            ax.plot(
                df_long["n params"].unique(),
                df_long[label].min() * 1e-3 * df_long["n params"].unique() ** 3,
                color="k",
                alpha=0.5,
                ls=":",
            )
            # ax.annotate(xy=(40, 200), text="$N^3$", alpha=0.5)
            ax.set_ylim(df_long[label].min(), df_long[label].max())
            ax.legend(loc="upper left")
            ax.set_ylabel("time [s]")
        else:
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [USE_METHODS[l]["label"] for l in labels]
            new_handles = [h for l, h in zip(new_labels, handles) if l is not None]
            new_labels = [l for l in new_labels if l is not None]
            ax.legend(new_handles, new_labels)
            ax.set_ylabel(label)
        ax.grid("on")
        ax.set_xlabel(xlabel)
        savefig(fig, fname.replace(".pkl", f"_{label}.png"))
        savefig(fig, fname.replace(".pkl", f"_{label}.pdf"))


def run_time_study(results_dir=RESULTS_DIR, overwrite=False, debug=False):
    if debug:
        appendix = "timetest"
        n_params_list = np.logspace(1, 6, 21).astype(int)[:2]
        n_threads_list = [10]
        n_seeds = 2
    else:
        appendix = "time"
        n_params_list = np.logspace(1, 6, 21).astype(int)[:11]
        n_threads_list = [10]
        n_seeds = N_SEEDS

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    if overwrite:
        for lifter in [lifter_ro, lifter_mat]:
            fname = f"{results_dir}/{lifter}_{appendix}.pkl"
            add_redundant_constr = (
                True if isinstance(lifter, MatWeightLocLifter) else False
            )
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                fname=fname,
                use_methods=list(USE_METHODS.keys()),
                noise_list=[lifter.NOISE],
                add_redundant_constr=add_redundant_constr,
                n_threads_list=n_threads_list,
                n_seeds=n_seeds,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

    fname = f"{results_dir}/{lifter_ro}_{appendix}.pkl"
    df = pd.read_pickle(fname)
    xlabel = "number of positions"
    plot_timing(df, xlabel=xlabel, fname=fname, use_methods=USE_METHODS_RO)

    xlabel = "number of poses"
    fname = f"{results_dir}/{lifter_mat}_{appendix}.pkl"
    df = pd.read_pickle(fname)
    plot_timing(df, xlabel=xlabel, fname=fname, use_methods=USE_METHODS_MW)


if __name__ == "__main__":
    run_time_study(overwrite=True)
