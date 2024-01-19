import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    plot_scalability,
    run_oneshot_experiment,
    run_scalability_new,
)
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

n_positions = 3
n_landmarks = 10
d = 3

DEBUG = False

RESULTS_DIR = "_results"
# RESULTS_DIR = "_results_server"


def range_only_tightness():
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
        fname_root = f"{RESULTS_DIR}/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner,
            fname_root,
            plots,
        )


def range_only_scalability_new(n_seeds, recompute):
    if DEBUG:
        n_positions_list = [10, 15]
    else:
        n_positions_list = [10, 15, 20, 25, 30]
    for level in ["no", "quad"]:
        print(f"=========== RO {level} scalability ===========")
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
        df = run_scalability_new(
            learner,
            param_list=n_positions_list,
            n_seeds=n_seeds,
            recompute=recompute,
            results_folder=RESULTS_DIR,
        )
        if df is None:
            continue

        df = df[df.type != "original"]
        fname_root = f"{RESULTS_DIR}/scalability_{learner.lifter}"

        df_sub = df[df.type != "from scratch"]["t solve SDP"]
        fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=1)

        # [ax.set_ylim(10, 1000) for ax in axs.values()]

        fig.set_size_inches(4, 3)
        axs[-1].legend(
            loc="upper right", fontsize=10, framealpha=1.0
        )  # , bbox_to_anchor=[1.0, 1.0])
        savefig(fig, fname_root + f"_t.pdf")

        # tex_name = fname_root + f"_n.tex"
        # save_table(df, tex_name)


def run_all(n_seeds, recompute, tightness=True, scalability=True):
    if scalability:
        range_only_scalability_new(recompute=recompute, n_seeds=n_seeds)
    if tightness:
        range_only_tightness()


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=False)
