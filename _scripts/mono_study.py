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

n_landmarks = 4
d = 2

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

    for level in ["no"]:  # ["xxT", "xwT"]:

        np.random.seed(seed)
        lifter = MonoLifter(
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list="all",
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner, fname_root, plots, tightness="rank", add_original=True
        )


if __name__ == "__main__":
    mono_tightness()
    # range_only_scalability_new()
