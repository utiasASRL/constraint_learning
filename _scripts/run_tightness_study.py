import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import USE_METHODS, savefig

RESULTS_READ = "_results_server"
RESULTS_WRITE = "_results"


def plot_this_vs_other(df_long, ax, other="EVR", this="noise"):
    # show individual points
    sns.stripplot(
        df_long,
        x=this,
        y=other,
        hue="solver type",
        dodge=0.05,
        ax=ax,
        zorder=1,
        legend=False,
        hue_order={m: kwargs["color"] for m, kwargs in USE_METHODS.items()},
    )
    # show means
    sns.pointplot(
        df_long,
        x=this,
        y=other,
        hue="solver type",
        dodge=0.8 - 0.8 / 6,
        ax=ax,
        markersize=4,
        linestyle="none",
        markers="d",
        errorbar=None,
        hue_order={m: f"C{i}" for i, m in enumerate(USE_METHODS)},
    )
    labels = [f"{eval(l._text):.2e}" for l in ax.get_xticklabels()]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2e}"))


if __name__ == "__main__":
    n_params_list = [100]
    sparsity_list = [1.0]  # np.linspace(0.5, 1.0, 6)[::-1]
    n_seeds = 5
    appendix = "noise"
    seed = 0
    overwrite = False

    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    if overwrite:
        np.random.seed(seed)
        noise_list = np.logspace(0, 1, 5)  # from 1 pixel to 10 pixels
        fname = f"{RESULTS_WRITE}/{lifter_mw}_{appendix}.pkl"
        df = generate_results(
            lifter_mw,
            n_params_list=n_params_list,
            fname=fname,
            noise_list=noise_list,
            sparsity_list=sparsity_list,
            n_seeds=n_seeds,
            use_methods=USE_METHODS,
            add_redundant_constr=True,
        )
        df.to_pickle(fname)
        print("saved final as", fname)

        np.random.seed(seed)
        noise_list = np.logspace(-2, 0, 5)  # from 1cm to 1m
        fname = f"{RESULTS_WRITE}/{lifter_ro}_{appendix}.pkl"
        df = generate_results(
            lifter_ro,
            n_params_list=n_params_list,
            fname=fname,
            noise_list=noise_list,
            sparsity_list=sparsity_list,
            n_seeds=n_seeds,
            use_methods=USE_METHODS,
            add_redundant_constr=False,
        )
        df.to_pickle(fname)
        print("saved final as", fname)

    for lifter in [lifter_mw, lifter_ro]:
        fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
        df = pd.read_pickle(fname)
        print(f"read {fname}")
        for label in ["EVR", "RDG"]:
            value_vars = [f"{label} local"] + [
                f"{label} {method}" for method in USE_METHODS
            ]
            value_vars = set(value_vars).intersection(df.columns.unique())
            # create long form for plotting
            df_long = df.melt(
                id_vars=["noise", "sparsity"],
                value_vars=value_vars,
                value_name=label,
                var_name="solver type",
            )
            # rename EVR SDP -> SDP etc.
            df_long.loc[:, "solver type"] = [
                f.strip(f"{label} ") for f in df_long["solver type"]
            ]

            chosen_sparsity = 1.0
            df_long_here = df_long[df_long["sparsity"] == chosen_sparsity]
            print(f"plotting at sparsity={chosen_sparsity}")

            fig, ax = plt.subplots()
            fig.set_size_inches(5, 5)
            methods = df_long_here["solver type"].unique()
            for m, kwargs in USE_METHODS.items():
                if m in methods:
                    rows = df_long_here[df_long_here["solver type"] == m]
                    # plot error bars
                    sns.pointplot(
                        x="noise",
                        y=label,
                        data=rows,
                        ax=ax,
                        color=kwargs["color"],
                        marker=None,
                        alpha=0.5 if "redun" in m else 1.0,
                        errorbar=("sd", 0.8),
                        log_scale=True,
                        label=kwargs["label"],
                    )
                    # plot all points
                    sns.stripplot(
                        x="noise",
                        y=label,
                        data=rows,
                        ax=ax,
                        color=kwargs["color"],
                        marker=kwargs["marker"],
                        alpha=0.5 if "redun" in m else 1.0,
                        dodge=0.05,
                        label=None,
                    )
                    pass
            ax.grid("on")
            ax.legend()
            ax.set_xticklabels(
                [f"{eval(l.get_text()):.2f}" for l in ax.get_xticklabels()]
            )
            ax.set_xlabel("noise")
            savefig(fig, fname.replace(".pkl", f"_{label}_noise.png"))
        print("done")
