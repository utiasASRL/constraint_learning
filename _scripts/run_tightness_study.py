import itertools

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import (
    TAB10_TO_RGB,
    USE_METHODS,
    USE_METHODS_MW,
    USE_METHODS_RO,
    savefig,
)

RESULTS_DIR = "_results"

SEED = 0
N_SEEDS = 5


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


def plot_success(fname, use_methods=USE_METHODS):
    label = "success"
    df = pd.read_pickle(fname)
    print(f"read {fname}")
    value_vars = [f"{label} local"] + [f"{label} {method}" for method in use_methods]
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

    df_long_here = (
        df_long_here.groupby(["noise", "solver type"]).sum()
        / df_long_here.groupby(["noise", "solver type"]).count()
    ) * 100
    df_long_here.reset_index(inplace=True)

    df_long_here["solver type"] = pd.Categorical(
        df_long_here["solver type"], categories=use_methods
    )
    df_long_here = df_long_here.sort_values("solver type")

    fig, axs = plt.subplots(1, len(df_long_here.noise.unique()), sharey=True)
    fig.set_size_inches(5, 5)
    for ax, (noise, df) in zip(axs, df_long_here.groupby("noise")):
        sns.stripplot(
            df,
            y="solver type",
            x=label,
            ax=ax,
            # hue=label,
            # palette="coolwarm",
            # palette={0: (255, 0, 0), 100: (0, 0, 255)}.update({i*10: (0, 255, 0) for i in range(1, 10)}),
            # dodge=True,
            # jitter=False,
        )
        ax.set_title(f"$\\sigma$={noise:.1f}")
        ax.grid()
        ax.set_xlabel("")
    axs[0].set_xlabel("success rate [\%]", loc="left")
    savefig(fig, fname.replace(".pkl", f"_{label}.png"))


def plot_boxplots(fname, label="error", log=True, use_methods=USE_METHODS):
    df = pd.read_pickle(fname)
    print(f"read {fname}")
    value_vars = [f"{label} local"] + [f"{label} {method}" for method in use_methods]
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

    fig, ax = plt.subplots()
    fig.set_size_inches(2.5, 5)
    colors = {
        m["label"]: list(TAB10_TO_RGB[m["color"]]) + [m["alpha"]]
        for v, m in USE_METHODS.items()
    }
    df_long_here["solver type"] = pd.Categorical(
        df_long_here["solver type"], categories=use_methods
    )
    df_long_here = df_long_here.sort_values("solver type")
    df_long_here.loc[:, "solver type"] = [
        USE_METHODS[m]["label"] for m in df_long_here["solver type"]
    ]

    sns.boxplot(
        df_long_here,
        x="noise",
        y=label,
        hue="solver type",
        ax=ax,
        palette=colors,
        hue_order=[USE_METHODS[m]["label"] for m in use_methods],
    )
    ax.legend(
        # loc="lower center",
        # bbox_to_anchor=[0.5, 1.0],
        # ncol=2,
        columnspacing=0.5,
        loc="upper left",
        bbox_to_anchor=[1.0, 1.0],
    )
    if log:
        ax.set_yscale("log")
    noises = sorted(df_long_here.noise.unique().round(2))
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f"{n:.2f}" for n in noises])
    ax.grid(axis="y")
    savefig(fig, fname.replace(".pkl", f"_{label}.png"))


def plot_tightness_study(fname, ylabels=["EVR", "RDG"], use_methods=USE_METHODS):
    df = pd.read_pickle(fname)
    print(f"read {fname}")
    for label in ylabels:
        value_vars = [f"{label} local"] + [
            f"{label} {method}" for method in use_methods
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

        fig, ax = plt.subplots()
        fig.set_size_inches(2.5, 5)
        methods = df_long_here["solver type"].unique()
        for m in use_methods:
            kwargs = USE_METHODS[m]
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
                    errorbar=("sd", 0.8),
                    log_scale=True,
                    label=kwargs["label"],
                    linestyles=kwargs["ls"],
                )
                # plot all points
                sns.stripplot(
                    x="noise",
                    y=label,
                    data=rows,
                    ax=ax,
                    color=kwargs["color"],
                    marker=kwargs["marker"],
                    dodge=0.05,
                    label=None,
                )
                pass
        ax.grid("on")
        ax.legend(
            # loc="lower center", bbox_to_anchor=[0.5, 1.0], columnspacing=0.5, ncol=2
            loc="upper left",
            bbox_to_anchor=[1.0, 1.0],
            columnspacing=0.5,
        )
        try:
            new_ticks = [f"{eval(l.get_text()):.2f}" for l in ax.get_xticklabels()]
            ax.set_xticks(range(len(new_ticks)))
            ax.set_xticklabels(new_ticks)
        except SyntaxError:
            continue
        ax.set_xlabel("noise")
        savefig(fig, fname.replace(".pkl", f"_{label}.png"))


def run_tightness_study(
    results_dir=RESULTS_DIR, overwrite=False, n_seeds=N_SEEDS, appendix="noise"
):
    if appendix == "noise":
        n_params_list = [100]
        sparsity_list = [1.0]
        n_threads_list = [10]
        n_noises = 5
    elif appendix == "noisetest":
        n_params_list = [100]
        sparsity_list = [1.0]
        n_threads_list = [2]
        n_noises = 2

    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    if overwrite:

        np.random.seed(SEED)
        noise_list = np.logspace(0, 1, n_noises)  # from 1 pixel to 10 pixels
        fname = f"{results_dir}/{lifter_mw}_{appendix}.pkl"
        df = generate_results(
            lifter_mw,
            n_params_list=n_params_list,
            fname=fname,
            noise_list=noise_list,
            sparsity_list=sparsity_list,
            n_seeds=n_seeds,
            use_methods=USE_METHODS,
            add_redundant_constr=True,
            n_threads_list=n_threads_list,
        )
        df.to_pickle(fname)
        print("saved final as", fname)

        np.random.seed(SEED)
        noise_list = np.logspace(-2, 0, n_noises)  # from 1cm to 1m
        fname = f"{results_dir}/{lifter_ro}_{appendix}.pkl"
        df = generate_results(
            lifter_ro,
            n_params_list=n_params_list,
            fname=fname,
            noise_list=noise_list,
            sparsity_list=sparsity_list,
            n_seeds=n_seeds,
            use_methods=USE_METHODS,
            add_redundant_constr=False,
            n_threads_list=n_threads_list,
        )
        df.to_pickle(fname)
        print("saved final as", fname)

    fname = f"{results_dir}/{lifter_mw}_{appendix}.pkl"
    plot_tightness_study(fname=fname, use_methods=USE_METHODS_MW)

    fname = f"{results_dir}/{lifter_ro}_{appendix}.pkl"
    plot_tightness_study(fname=fname, use_methods=USE_METHODS_RO)


def plot_accuracy_study_all(results_dir=RESULTS_DIR, appendix="noise"):
    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    fname = f"{results_dir}/{lifter_mw}_{appendix}.pkl"
    plot_boxplots(fname=fname, label="error", use_methods=USE_METHODS_MW)

    fname = f"{results_dir}/{lifter_ro}_{appendix}.pkl"
    plot_boxplots(fname=fname, label="error", use_methods=USE_METHODS_RO)


def plot_success_study_all(results_dir=RESULTS_DIR, appendix="noise"):
    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    for lifter in [lifter_mw, lifter_ro]:
        fname = f"{results_dir}/{lifter}_{appendix}.pkl"
        plot_success(fname=fname)


if __name__ == "__main__":
    run_tightness_study(overwrite=True, n_seeds=N_SEEDS)
