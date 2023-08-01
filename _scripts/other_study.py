import itertools

import numpy as np
import pandas as pd

from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter
from utils.plotting_tools import savefig

from _scripts.stereo_study import (
    run_oneshot_experiment,
    plot_scalability,
    run_scalability_new,
)

def lifter_tightness(Lifter = MonoLifter, robust:bool = False, d:int=2):
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    plots = [
        #"svd",
        #"matrices",
        #"matrix",
        "tightness",
        "templates",
    ] 

    if d == 2:
        n_landmarks = 4
    elif d == 3:
        n_landmarks = 8
    if robust:
        levels = ["xwT", "xxT"]
    else:
        levels = ["no"]

    for level in levels:
        np.random.seed(seed)
        lifter = Lifter(
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list="all",
            robust=robust
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner, fname_root, plots, tightness="rank", add_original=True, use_last=20
        )

def lifter_scalability_new(Lifter, d:int=2):
    if d == 2:
        n_landmarks = 4
        n_landmarks_list = [4, 6, 8]#, 20, 25, 30]
        n_seeds = 1  # 10
    elif d == 3:
        n_landmarks = 8
        n_landmarks_list = [9, 10]
        n_seeds = 1

    level = "xwT"
    robust = True

    variable_list = None  # use the default one for the first step.
    np.random.seed(0)
    lifter = Lifter(
        n_landmarks=n_landmarks,
        d=d,
        level=level,
        variable_list=variable_list,
        robust=robust
    )

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
    run_scalability_new(
        learner, param_list=n_landmarks_list, n_seeds=n_seeds, use_last=None, recompute=True#vmin=0.1, vmax=50
    )


if __name__ == "__main__":
    #from lifters.mono_lifter import MonoLifter as Lifter
    from lifters.wahba_lifter import WahbaLifter as Lifter

    #for d, robust in itertools.product([2, 3], [False, True]):
    #    lifter_tightness(Lifter, d=d, robust=robust)

    for d in [2]: #, 3]:
        lifter_scalability_new(Lifter, d=d)