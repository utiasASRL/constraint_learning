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
from utils.plotting_tools import TAB10_TO_RGB, USE_METHODS, savefig

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


def plot_boxplots(fname, label="error"):
    df = pd.read_pickle(fname)
    print(f"read {fname}")
    value_vars = [f"{label} local"] + [f"{label} {method}" for method in USE_METHODS]
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
    # for m, kwargs in USE_METHODS.items():
    #    if m in methods:
    # rows = df_long_here[df_long_here["solver type"] == m]
    colors = {
        v: list(TAB10_TO_RGB[m["color"]]) + [m["alpha"]] for v, m in USE_METHODS.items()
    }
    df_long_here["solver type"] = pd.Categorical(
        df_long_here["solver type"], categories=USE_METHODS.keys()
    )
    df_long_here = df_long_here.sort_values("solver type")

    sns.boxplot(
        df_long_here,
        x="noise",
        y="error",
        hue="solver type",
        ax=ax,
        palette=colors,
        hue_order=USE_METHODS,
    )
    # ordered by noise, solver type
    for i, (solver, noise) in enumerate(
        itertools.product(
            df_long_here["solver type"].unique(),
            df_long_here["noise"].unique(),
        )
    ):
        print(noise, solver)
        patch = ax.patches[i]
        patch.set_facecolor(colors[solver])

    h, l = ax.get_legend_handles_labels()
    new_h_l = [(hi, li) for hi, li in zip(h, l) if "redun" not in li]
    ax.legend([h_l[0] for h_l in new_h_l], [h_l[1] for h_l in new_h_l])

    ax.set_yscale("log")
    ax.set_xticklabels(df_long_here.noise.unique().round(2))
    ax.grid(axis="y")
    savefig(fig, fname.replace(".pkl", f"_{label}_noise.png"))
    print("done")
    # print(rows)


def plot_tightness_study(fname, ylabels=["EVR", "RDG"]):
    df = pd.read_pickle(fname)
    print(f"read {fname}")
    for label in ylabels:
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
                    alpha=0.5 if "redun" in m else 1.0,
                    dodge=0.05,
                    label=None,
                )
                pass
        ax.grid("on")
        ax.legend()
        try:
            ax.set_xticklabels(
                [f"{eval(l.get_text()):.2f}" for l in ax.get_xticklabels()]
            )
        except SyntaxError:
            ax.set_xticklabels([l.get_text() for l in ax.get_xticklabels()])
        ax.set_xlabel("noise")
        savefig(fig, fname.replace(".pkl", f"_{label}_noise.png"))


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

    for lifter in [lifter_mw, lifter_ro]:
        fname = f"{results_dir}/{lifter}_{appendix}.pkl"
        plot_tightness_study(fname=fname)


def plot_accuracy_study_all(results_dir=RESULTS_DIR, appendix="noise"):
    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    for lifter in [lifter_mw, lifter_ro]:
        fname = f"{results_dir}/{lifter}_{appendix}.pkl"
        plot_boxplots(fname=fname, label="error")


def plot_success_study_all(results_dir=RESULTS_DIR, appendix="noise"):
    lifter_mw = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    for lifter in [lifter_mw, lifter_ro]:
        fname = f"{results_dir}/{lifter}_{appendix}.pkl"
        plot_boxplots(fname=fname, label="success")


if __name__ == "__main__":
    run_tightness_study(overwrite=True, n_seeds=N_SEEDS)
