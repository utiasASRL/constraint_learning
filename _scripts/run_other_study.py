import numpy as np

from experiments import plot_scalability, save_table
from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter
from utils.plotting_tools import savefig

from _scripts.stereo_study import (
    run_oneshot_experiment,
    run_scalability_new,
)

RECOMPUTE = True

def lifter_tightness(Lifter=MonoLifter, robust: bool = False, d: int = 2, n_landmarks=4, n_outliers=0):
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    plots = [
        "tightness",
    ]
    if robust:
        levels = ["xwT"]  # ["xxT"] #["xwT", "xxT"]
    else:
        levels = ["no"]

    for level in levels:
        np.random.seed(seed)
        lifter = Lifter(
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list="all",
            robust=robust,
            n_outliers=n_outliers if robust else 0
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner,
            fname_root,
            plots,
            tightness="rank",
            add_original=True,
            use_bisection=False,
            use_known=True
        )


def lifter_scalability_new(Lifter, d, n_landmarks, n_outliers, robust):
    level = "xwT"
    variable_list = None  # use the default one for the first step.

    if d == 2:
        n_landmarks_list = [5, 10, 15, 20, 25, 30]
        n_seeds = 1
    elif d == 3:
        n_landmarks_list = [10, 11, 12, 13, 14, 15]  # , 20, 25, 30]
        n_seeds = 1

    np.random.seed(0)
    lifter = Lifter(
        n_landmarks=n_landmarks,
        d=d,
        level=level,
        variable_list=variable_list,
        robust=robust,
        n_outliers=n_outliers
    )
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
    df = run_scalability_new(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        use_last=None,
        recompute=RECOMPUTE,
        use_bisection=True,
        add_original=False,
        tightness="cost",
        use_known=False
    )

    fname_root = f"_results/scalability_{learner.lifter}"
    fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=1)
    [ax.set_ylim(10, 1000) for ax in axs.values()]

    fig.set_size_inches(8, 3)
    #axs["t solve SDP"].legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
    savefig(fig, fname_root + f"_t.pdf")
    
    #fig, ax = plot_scalability(df, log=True, start="n ")
    #axs["t solve SDP"].legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
    fig.set_size_inches(5, 3)
    savefig(fig, fname_root + f"_n.pdf")

    tex_name = fname_root + f"_n.tex"
    save_table(df, tex_name)

if __name__ == "__main__":
    from lifters.mono_lifter import MonoLifter
    from lifters.wahba_lifter import WahbaLifter

    d = 3

    # study of non-robust lifters (scales well)
    robust = False
    lifter_tightness(WahbaLifter, d=d, n_landmarks=3, robust=robust)
    lifter_tightness(MonoLifter, d=d, n_landmarks=5, robust=robust)

    
    # study of robust lfiters (scales poorly --- use incremental only)
    robust = True
    n_outliers = 1
    lifter_scalability_new(WahbaLifter, d=d, n_landmarks=3+n_outliers, robust=robust, n_outliers=n_outliers)
    lifter_scalability_new(MonoLifter, d=d, n_landmarks=5+n_outliers, robust=robust, n_outliers=n_outliers)