import itertools

import numpy as np
import pandas as pd

from lifters.learner import Learner
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

from _scripts.stereo_study import run_oneshot_experiment, plot_scalability, run_scalability_new

n_positions = 3
n_landmarks = 10
d = 3

def range_only_tightness():
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    #plots = [] 
    plots = ["svd", "matrices", "matrix", "tightness", "templates"]  # ["svd", "matrices"]

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
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(learner, fname_root, plots, tightness="cost", add_original=True)

def range_only_scalability_new():
    n_positions_list = [10, 15, 20, 25, 30]
    n_seeds = 1
    for level in ["no", "quad"]: 
        variable_list = None  # use the default one for the first step.
        np.random.seed(0)
        lifter = RangeOnlyLocLifter(
            d=d,
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            level=level,
            variable_list=variable_list,
        )
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
        run_scalability_new(learner, param_list=n_positions_list, n_seeds=n_seeds, tightness="rank", recompute=False)

if __name__ == "__main__":
    range_only_tightness()
    #range_only_scalability_new()
