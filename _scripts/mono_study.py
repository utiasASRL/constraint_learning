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

n_landmarks = 8 # minimum required number: 8
d = 3

#n_landmarks = 4 # minimum required number: 4
#d = 2
robust = True

def mono_tightness():
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

    if robust:
        levels = ["xwT"]#, "xxT"]
    else:
        levels = ["no"]

    for level in levels:
        np.random.seed(seed)
        lifter = MonoLifter(
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
            learner, fname_root, plots, tightness="rank", add_original=True, use_last=100
        )

def mono_scalability_new(d=2):
    if d == 2:
        n_landmarks = 4
        n_landmarks_list = [10, 15, 20, 25, 30]
        n_seeds = 3  # 10
    elif d == 3:
        n_landmarks = 8
        n_landmarks_list = [5, 10]
        n_seeds = 1

    level = "xwT"
    robust = True

    variable_list = None  # use the default one for the first step.
    np.random.seed(0)
    lifter = MonoLifter(
        n_landmarks=n_landmarks,
        d=d,
        level=level,
        variable_list=variable_list,
        robust=robust
    )

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
    run_scalability_new(
        learner, param_list=n_landmarks_list, n_seeds=n_seeds, use_last=50#vmin=0.1, vmax=50
    )


if __name__ == "__main__":
    #mono_tightness()
    mono_scalability_new()