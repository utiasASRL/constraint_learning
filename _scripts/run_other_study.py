import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    apply_autotemplate_base,
    apply_autotight_base,
    plot_autotemplate_time,
)
from lifters.mono_lifter import MonoLifter
from utils.plotting_tools import FIGSIZE, add_lines, savefig

RESULTS_DIR = "_results_server_v3"
# RESULTS_DIR = "_results_server"

ONLY_ROBUST = False


def apply_autotight(
    Lifter=MonoLifter,
    robust: bool = False,
    d: int = 2,
    n_landmarks=4,
    n_outliers=0,
    results_dir=RESULTS_DIR,
):
    """
    Find the set of minimal constraints required for autotight for range-only problem.
    """
    seed = 0
    plots = ["templates", "templates-full"]
    if robust:
        level = "xwT"
    else:
        level = "no"

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
    apply_autotight_base(
        learner,
        fname_root,
        plots,
    )


def apply_autotemplate(
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

    df = apply_autotemplate_base(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        recompute=recompute,
        results_folder=results_dir,
    )
    if df is None:
        return

    fname_root = f"{results_dir}/autotemplate_{learner.lifter}"
    fig, axs = plot_autotemplate_time(df, log=True, start="t ", legend_idx=1)
    [ax.set_ylim(10, 1000) for ax in axs]

    axs[0].set_xticks(df.N.unique(), [f"{x:.0f}" for x in df.N.unique()])
    add_lines(axs[0], df.N.unique(), start=df["t create constraints"].min(), facs=[3])
    add_lines(axs[1], df.N.unique(), start=df["t solve SDP"].min(), facs=[3])
    savefig(fig, fname_root + f"_t.pdf")


def run_wahba(
    n_seeds, recompute, autotight=True, autotemplate=True, results_dir=RESULTS_DIR
):
    from lifters.wahba_lifter import WahbaLifter

    d = 3
    n_outliers = 1

    print("================= Wahba study ==================")

    if autotight and not ONLY_ROBUST:
        apply_autotight(
            WahbaLifter, d=d, n_landmarks=4, robust=False, results_dir=results_dir
        )
    if autotemplate:
        if not ONLY_ROBUST:
            apply_autotemplate(
                WahbaLifter,
                d=d,
                n_landmarks=4,
                robust=False,
                n_outliers=0,
                n_seeds=n_seeds,
                recompute=recompute,
                results_dir=results_dir,
            )
        apply_autotemplate(
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
    n_seeds, recompute, autotight=True, autotemplate=True, results_dir=RESULTS_DIR
):
    from lifters.mono_lifter import MonoLifter

    d = 3
    n_outliers = 1

    print("================= Mono study ==================")

    if autotight and not ONLY_ROBUST:
        apply_autotight(
            MonoLifter, d=d, n_landmarks=5, robust=False, results_dir=results_dir
        )
    if autotemplate:
        if not ONLY_ROBUST:
            apply_autotemplate(
                MonoLifter,
                d=d,
                n_landmarks=5,
                robust=False,
                n_outliers=0,
                n_seeds=n_seeds,
                recompute=recompute,
                results_dir=results_dir,
            )
        apply_autotemplate(
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
    n_seeds, recompute, autotight=True, autotemplate=True, results_dir=RESULTS_DIR
):
    run_wahba(
        n_seeds,
        recompute,
        autotight=autotight,
        autotemplate=autotemplate,
        results_dir=results_dir,
    )
    run_mono(
        n_seeds,
        recompute,
        autotight=autotight,
        autotemplate=autotemplate,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    run_all(n_seeds=1, recompute=True, autotemplate=True, autotight=False)
    # run_mono(n_seeds=1, recompute=True)
    # run_wahba(n_seeds=1, recompute=True)
