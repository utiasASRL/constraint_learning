import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import matshow_list, savefig

USE_METHODS = ["SDP", "SDP-redun", "dSDP", "dSDP-redun", "ADMM", "ADMM-redun"]


def plot_this_vs_other(df_long, ax, label="EVR", this="noise", other="sparsity"):
    chosen_other = df_long[other].values[-1]
    print(f"plotting at {other}={chosen_other}")
    df_long = df_long[df_long[other] == chosen_other]
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
    xticklabels = [f"{n:2.0e}" for n in df_long[this].unique()]
    ax.set_xticks(range(len(xticklabels)), xticklabels)


if __name__ == "__main__":
    n_params_list = [10]
    noise_list = np.logspace(-2, 1, 7)
    sparsity_list = [0.8]  # np.linspace(0.5, 1.0, 6)[::-1]
    n_seeds = 3

    appendix = "all"
    overwrite = True

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2, level="no"
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    for lifter in [lifter_ro]:
        lifter.ALL_PAIRS = False
        lifter.CLIQUE_SIZE = 2

        fname = f"_results/{lifter}_{appendix}.pkl"

        try:
            assert overwrite is False
            df = pd.read_pickle(fname)
        except (FileNotFoundError, AssertionError):
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                fname=fname,
                noise_list=noise_list,
                sparsity_list=sparsity_list,
                n_seeds=n_seeds,
                use_methods=USE_METHODS,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

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
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 4)
            plot_this_vs_other(df_long, ax, label=label, this="noise", other="sparsity")
            ax.set_yscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}_noise.png"))

            fig, ax = plt.subplots()
            fig.set_size_inches(7, 4)
            plot_this_vs_other(df_long, ax, label=label, this="sparsity", other="noise")
            ax.set_yscale("log")
            ax.grid("on")
            savefig(fig, fname.replace(".pkl", f"_{label}_sparsity.png"))

        print("done")
