import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    apply_autotemplate_base,
    apply_autotight_base,
    plot_autotemplate_time,
)
from lifters.matweight_lifter import MatWeightLocLifter
from utils.plotting_tools import savefig

# matplotlib.use("TkAgg")
# plt.ion()
# matplotlib.use('Agg') # non-interactive
# plt.ioff()


DEBUG = False

RESULTS_DIR = "_results"
# RESULTS_DIR = "_results_server"


def mw_loc_autotight(n_landmarks=5, n_poses=2):
    """
    Find the set of minimal constraints required for autotight for stereo problem.
    """
    seed = 0
    np.random.seed(seed)

    trans_frame = "local"
    label = "t0" if trans_frame == "world" else "t"

    # TODO(FD) continue here: fix variable list.
    variable_list = [
        ["h"] + [l for i in range(n_poses) for l in [f"xC_{i}", f"x{label}_{i}"]]
    ]
    plots = ["autotight", "matrix", "svd"]
    lifter = MatWeightLocLifter(
        n_landmarks=n_landmarks,
        n_poses=n_poses,
        variable_list=variable_list,
        trans_frame=trans_frame,
    )

    learner = Learner(
        lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
    )
    fname_root = f"{RESULTS_DIR}/{lifter}_seed{seed}"
    apply_autotight_base(learner, fname_root, plots)


def mw_loc_autotemplate_new(n_seeds, recompute):
    if DEBUG:
        n_landmarks_list = [10, 15]
    else:
        n_landmarks_list = [10, 15, 20, 25, 30]

    n_landmarks = 5
    n_poses = 4

    np.random.seed(0)
    lifter = MatWeightLocLifter(n_landmarks=n_landmarks, n_poses=n_poses)
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

    df = apply_autotemplate_base(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        recompute=recompute,
        results_folder=RESULTS_DIR,
    )
    if df is None:
        return

    fname_root = f"{RESULTS_DIR}/autotemplate_{learner.lifter}"

    fig, axs = plot_autotemplate_time(df, log=True, start="t ", legend_idx=1)
    # [ax.set_ylim(10, 1000) for ax in axs.values()]
    # [ax.set_ylim(2, 8000) for ax in axs]

    fig.set_size_inches(4, 3)
    axs[1].legend(loc="upper right", fontsize=10, framealpha=1.0)
    savefig(fig, fname_root + f"_t.pdf")

    # fig, ax = plot_autotemplate(df, log=True, start="n ")
    # fig.set_size_inches(5, 5)
    # savefig(fig, fname_root + f"_n.pdf")

    # tex_name = fname_root + f"_n.tex"
    # save_table(df, tex_name)


def run_all(n_seeds, recompute, autotight=True, autotemplate=True):
    if autotight:
        print("========== MatWeightLoc autotight ===========")
        mw_loc_autotight()
    if autotemplate:
        print("========== MatWeightLoc autotemplate ===========")
        mw_loc_autotemplate_new(n_seeds=n_seeds, recompute=recompute)


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=False)
