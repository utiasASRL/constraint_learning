import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    apply_autotemplate_base,
    apply_autotight_base,
    plot_autotemplate_time,
)
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import FIGSIZE, add_lines, savefig

n_positions = 3
n_landmarks = 10
d = 3

RESULTS_DIR = "_results"
# RESULTS_DIR = "_results_server"


def apply_autotight(results_dir=RESULTS_DIR):
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    # plots = []
    plots = [
        "svd",
        "matrices",
        "matrix",
        "tightness",
        "templates",
    ]  # ["svd", "matrices"]

    for level in ["no", "quad"]:
        np.random.seed(seed)
        lifter = RangeOnlyLocLifter(
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list="all",
        )
        learner = Learner(
            lifter=lifter,
            variable_list=lifter.variable_list,
            apply_templates=False,
            n_inits=1,
        )
        fname_root = f"{results_dir}/{lifter}_seed{seed}"
        apply_autotight_base(
            learner,
            fname_root,
            plots,
        )


def apply_autotemplate(n_seeds, recompute, results_dir=RESULTS_DIR):
    n_positions_list = [10, 15, 20, 25, 30]
    for level in ["no", "quad"]:
        print(f"=========== RO {level} autotemplate ===========")
        variable_list = None  # use the default one for the first step.
        np.random.seed(0)
        lifter = RangeOnlyLocLifter(
            d=d,
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            level=level,
            variable_list=variable_list,
        )
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        df = apply_autotemplate_base(
            learner,
            param_list=n_positions_list,
            n_seeds=n_seeds,
            recompute=recompute,
            results_folder=results_dir,
        )
        if df is None:
            continue

        df = df[df.type != "original"]
        fname_root = f"{results_dir}/autotemplate_{learner.lifter}"

        df_sub = df[df.type != "from scratch"]["t solve SDP"]
        fig, axs = plot_autotemplate_time(df, log=True, start="t ", legend_idx=1)

        axs[0].set_xticks(df.N.unique(), [f"{x:.0f}" for x in df.N.unique()])
        add_lines(axs[0], df.N.unique(), start=df["t create constraints"].min())
        add_lines(axs[1], df.N.unique(), start=df["t solve SDP"].min())

        savefig(fig, fname_root + f"_t.pdf")


def run_all(
    n_seeds, recompute, autotight=True, autotemplate=True, results_dir=RESULTS_DIR
):
    if autotemplate:
        apply_autotemplate(
            recompute=recompute, n_seeds=n_seeds, results_dir=results_dir
        )
    if autotight:
        apply_autotight(results_dir=results_dir)


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=True)
