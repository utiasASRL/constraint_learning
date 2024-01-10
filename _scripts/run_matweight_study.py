import numpy as np
from mwcerts.lifters import MatWeightLocLifter

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    plot_scalability,
    run_oneshot_experiment,
    run_scalability_new,
)
from utils.plotting_tools import savefig

# matplotlib.use("TkAgg")
# plt.ion()
# matplotlib.use('Agg') # non-interactive
# plt.ioff()


DEBUG = False

RESULTS_DIR = "_results"
# RESULTS_DIR = "_results_server"


def mw_loc_tightness(n_landmarks, n_poses):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    seed = 0
    np.random.seed(seed)

    # TODO(FD) continue here: fix variable list.
    variable_list = [["h", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
    plots = ["tightness", "matrix", "templates", "svd"]
    lifter = MatWeightLocLifter(
        n_landmarks=n_landmarks, n_poses=n_poses, variable_list=variable_list
    )

    learner = Learner(
        lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
    )
    fname_root = f"{RESULTS_DIR}/{lifter}_seed{seed}"
    run_oneshot_experiment(learner, fname_root, plots)


def mw_loc_scalability_new(n_seeds, recompute):
    if DEBUG:
        n_landmarks_list = [10, 15]
    else:
        n_landmarks_list = [10, 15, 20, 25, 30]

    n_landmarks = 5
    n_poses = 4

    # variable_list = [["h", "x"] + [f"z_{i}" for i in range(n_landmarks)]] runs out of memory for d=3
    variable_list = None  # use the default one for the first step.
    np.random.seed(0)
    lifter = MatWeightLocLifter(n_landmarks=n_landmarks, n_poses=n_poses)
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

    df = run_scalability_new(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        recompute=recompute,
        results_folder=RESULTS_DIR,
    )
    if df is None:
        return

    fname_root = f"{RESULTS_DIR}/scalability_{learner.lifter}"

    fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=1)
    # [ax.set_ylim(10, 1000) for ax in axs.values()]
    [ax.set_ylim(2, 8000) for ax in axs]

    fig.set_size_inches(4, 3)
    axs[1].legend(loc="upper right", fontsize=10, framealpha=1.0)
    savefig(fig, fname_root + f"_t.pdf")

    # fig, ax = plot_scalability(df, log=True, start="n ")
    # fig.set_size_inches(5, 5)
    # savefig(fig, fname_root + f"_n.pdf")

    # tex_name = fname_root + f"_n.tex"
    # save_table(df, tex_name)


def run_all(n_seeds, recompute, tightness=True, scalability=True):
    if scalability:
        print("========== MatWeightLoc scalability ===========")
        mw_loc_scalability_new(n_seeds=n_seeds, recompute=recompute)
    if tightness:
        print("========== MatWeightLoc tightness ===========")
        mw_loc_tightness()


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=True)
