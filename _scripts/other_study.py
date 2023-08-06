import numpy as np

from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter

from _scripts.stereo_study import (
    run_oneshot_experiment,
    run_scalability_new,
)

N_OUTLIERS = 0

def lifter_tightness(Lifter=MonoLifter, robust: bool = False, d: int = 2):
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    plots = [
        # "svd",
        # "matrices",
        # "matrix",
        "tightness",
        "templates",
    ]

    if d == 2:
        n_landmarks = 4  # we have d**2 + d = 6 unknowns --> need n>=3 landmarks
    elif d == 3:
        n_landmarks = 8  # we have d**2 + d = 12 unknowns --> need n>=6 landmarks
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
            n_outliers=N_OUTLIERS if robust else 0
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
            use_last=30,
            use_bisection=False,
        )


def lifter_scalability_new(Lifter, d: int = 2):
    level = "xwT"
    robust = True
    variable_list = None  # use the default one for the first step.

    if d == 2:
        n_landmarks = 4
        n_landmarks_list = [5, 10, 15, 20, 25, 30]
        n_seeds = 1
    elif d == 3:
        n_landmarks = 8 + N_OUTLIERS if robust else 8
        n_landmarks_list = [10, 11, 12, 13, 14, 15]  # , 20, 25, 30]
        n_seeds = 1

    np.random.seed(0)
    lifter = Lifter(
        n_landmarks=n_landmarks,
        d=d,
        level=level,
        variable_list=variable_list,
        robust=robust,
        n_outliers=N_OUTLIERS if robust else 0
    )
    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
    run_scalability_new(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        use_last=None,
        recompute=False,
        use_bisection=True,
        add_original=False,
        tightness="cost",
        use_known=False
    )


if __name__ == "__main__":
    from lifters.mono_lifter import MonoLifter
    from lifters.wahba_lifter import WahbaLifter

    d = 3
    #lifter_scalability_new(WahbaLifter, d=d)
    lifter_scalability_new(MonoLifter, d=d)

    #robust = False
    #for Lifter in [MonoLifter, WahbaLifter]:
    #    lifter_tightness(Lifter, d=d, robust=robust)