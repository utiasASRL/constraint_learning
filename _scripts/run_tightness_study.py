import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

USE_METHODS = [
    "local",
    "SDP",
    "SDP-redun",
    "dSDP",
    "dSDP-redun",
    "pADMM",
    "pADMM-redun",
]


def plot_this_vs_other(df_long, ax, label="EVR", this="noise", other="sparsity"):
    # show individual points
    sns.stripplot(
        df_long,
        x=this,
        y=label,
        hue="solver type",
        dodge=0.05,
        ax=ax,
        alpha=0.25,
        zorder=1,
        legend=False,
        hue_order={m: f"C{i}" for i, m in enumerate(USE_METHODS)},
    )
    # show means
    sns.pointplot(
        df_long,
        x=this,
        y=label,
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
    n_seeds = 1
    appendix = "noise"
    seed = 0

    np.random.seed(seed)
    noise_list = np.logspace(0, 1, 5) # from 1 pixel to 10 pixels
    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    fname = f"_results/{lifter_mw}_{appendix}.pkl"
    df = generate_results(
        lifter_mw,
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

    np.random.seed(seed)
    noise_list = np.logspace(-2, 0, 5) # from 1cm to 1m
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    fname = f"_results/{lifter_ro}_{appendix}.pkl"
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
        fname = f"_results/{lifter}_{appendix}.pkl"
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
            fig.set_size_inches(7, 4)
            plot_this_vs_other(
                df_long_here, ax, label=label, this="noise", other="sparsity"
            )
            ax.set_yscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}_noise.png"))

            chosen_noise = df_long.noise.min()
            df_long_here = df_long[df_long["noise"] == chosen_noise]
            print(f"plotting at noise={chosen_noise}")

            fig, ax = plt.subplots()
            fig.set_size_inches(7, 4)
            plot_this_vs_other(
                df_long_here, ax, label=label, this="sparsity", other="noise"
            )
            ax.set_yscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}_sparsity.png"))

        print("done")
