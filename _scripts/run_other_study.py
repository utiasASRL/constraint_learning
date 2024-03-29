import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    plot_scalability,
    run_oneshot_experiment,
    run_scalability_new,
    save_table,
)
from lifters.mono_lifter import MonoLifter
from utils.plotting_tools import savefig

RESULTS_DIR = "_results"
# RESULTS_DIR = "_results_server"


def lifter_tightness(
    Lifter=MonoLifter,
    robust: bool = False,
    d: int = 2,
    n_landmarks=4,
    n_outliers=0,
    results_dir=RESULTS_DIR,
):
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    plots = ["tightness"]
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
            n_outliers=n_outliers if robust else 0,
        )
        learner = Learner(
            lifter=lifter,
            variable_list=lifter.variable_list,
            apply_templates=False,
            n_inits=1,
        )
        fname_root = f"{results_dir}/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner,
            fname_root,
            plots,
        )


def lifter_scalability_new(
    Lifter,
    d,
    n_landmarks,
    n_outliers,
    robust,
    n_seeds,
    recompute,
    results_dir=RESULTS_DIR,
):
    if robust:
        level = "xwT"
    else:
        level = "no"
    variable_list = None  # use the default one for the first step.

    n_landmarks_list = [10, 11, 12, 13, 14, 15]

    np.random.seed(0)
    lifter = Lifter(
        n_landmarks=n_landmarks,
        d=d,
        level=level,
        variable_list=variable_list,
        robust=robust,
        n_outliers=n_outliers,
    )
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)

    df = run_scalability_new(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        recompute=recompute,
        results_folder=results_dir,
    )
    if df is None:
        return

    fname_root = f"{results_dir}/scalability_{learner.lifter}"
    fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=1)
    [ax.set_ylim(10, 1000) for ax in axs]

    fig.set_size_inches(4, 3)
    axs[1].legend(loc="upper right", fontsize=10, framealpha=1.0)
    savefig(fig, fname_root + f"_t.pdf")

    # fig, ax = plot_scalability(df, log=True, start="n ")
    # axs["t solve SDP"].legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
    # fig.set_size_inches(5, 3)
    # savefig(fig, fname_root + f"_n.pdf")

    # tex_name = fname_root + f"_n.tex"
    # save_table(df, tex_name)


def run_wahba(
    n_seeds, recompute, tightness=True, scalability=True, results_dir=RESULTS_DIR
):
    from lifters.wahba_lifter import WahbaLifter

    d = 3
    n_outliers = 1

    print("================= Wahba study ==================")

    if tightness:
        lifter_tightness(
            WahbaLifter, d=d, n_landmarks=4, robust=False, results_dir=results_dir
        )
    if scalability:
        lifter_scalability_new(
            WahbaLifter,
            d=d,
            n_landmarks=4,
            robust=False,
            n_outliers=0,
            n_seeds=n_seeds,
            recompute=recompute,
            results_dir=results_dir,
        )
        lifter_scalability_new(
            WahbaLifter,
            d=d,
            n_landmarks=4 + n_outliers,
            robust=True,
            n_outliers=n_outliers,
            n_seeds=n_seeds,
            recompute=recompute,
            results_dir=results_dir,
        )


def run_mono(
    n_seeds, recompute, tightness=True, scalability=True, results_dir=RESULTS_DIR
):
    from lifters.mono_lifter import MonoLifter

    d = 3
    n_outliers = 1

    print("================= Mono study ==================")

    if tightness:
        lifter_tightness(
            MonoLifter, d=d, n_landmarks=5, robust=False, results_dir=results_dir
        )
    if scalability:
        lifter_scalability_new(
            MonoLifter,
            d=d,
            n_landmarks=5,
            robust=False,
            n_outliers=0,
            n_seeds=n_seeds,
            recompute=recompute,
            results_dir=results_dir,
        )
        lifter_scalability_new(
            MonoLifter,
            d=d,
            n_landmarks=5 + n_outliers,
            robust=True,
            n_outliers=n_outliers,
            n_seeds=n_seeds,
            recompute=recompute,
            results_dir=results_dir,
        )


def run_all(
    n_seeds, recompute, tightness=True, scalability=True, results_dir=RESULTS_DIR
):
    run_mono(
        n_seeds,
        recompute,
        tightness=tightness,
        scalability=scalability,
        results_dir=results_dir,
    )
    run_wahba(
        n_seeds,
        recompute,
        tightness=tightness,
        scalability=scalability,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=True)
    # run_mono(n_seeds=1, recompute=True)
    # run_wahba(n_seeds=1, recompute=True)
