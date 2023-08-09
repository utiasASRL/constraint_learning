import itertools
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import matplotlib

matplotlib.use("TkAgg")
plt.ion()
# matplotlib.use('Agg') # non-interactive
# plt.ioff()

from experiments import run_scalability_new, run_oneshot_experiment
from experiments import plot_scalability, save_table
from utils.plotting_tools import savefig

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from utils.plotting_tools import savefig


def stereo_tightness(d=2, n_landmarks=None):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    if n_landmarks is None:
        n_landmarks = d + 1
    seed = 0

    # parameter_levels = ["ppT"] #["no", "p", "ppT"]
    levels = ["no", "urT"]
    param_level = "no"
    for level in levels:
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        if d == 2:
            plots = ["tightness"]#, "matrix"]
            tightness = "cost"

            #plots = ["matrices", "templates", "svd"]
            #tightness = "cost"

            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )
        elif d == 3:
            plots = ["tightness"]
            tightness = "cost"

            # plots = ["matrices", "templates", "svd"]
            # tightness = "cost"

            lifter = Stereo3DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )

        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"

        run_oneshot_experiment(
            learner, fname_root, plots, tightness=tightness, add_original=True
        )


def stereo_scalability_new(d=2):
    if d == 2:
        n_landmarks_list = [5, 10, 15, 20, 25, 30]
        n_seeds = 10
    elif d == 3:
        n_landmarks_list = [10, 15, 20, 25, 30]
        n_seeds = 10 
    level = "urT"
    param_level = "ppT"

    n_landmarks = d + 1

    # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]] runs out of memory for d=3
    variable_list = None  # use the default one for the first step.
    np.random.seed(0)
    if d == 2:
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
    elif d == 3:
        lifter = Stereo3DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
    #run_scalability_plot(learner)
    df = run_scalability_new(
        learner, param_list=n_landmarks_list, n_seeds=n_seeds, recompute=False, use_bisection=True, use_known=False, add_original=False
    )

    fname_root = f"_results/scalability_{learner.lifter}"

    fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=0)

    #[ax.set_ylim(10, 1000) for ax in axs.values()]

    fig.set_size_inches(8, 3)
    axs["t create constraints"].legend(loc="lower right")
    savefig(fig, fname_root + f"_t.pdf")

    #fig, ax = plot_scalability(df, log=True, start="n ")
    #fig.set_size_inches(5, 5)
    #savefig(fig, fname_root + f"_n.pdf")

    tex_name = fname_root + f"_n.tex"
    save_table(df, tex_name)

if __name__ == "__main__":
    # import warnings
    # with warnings.catch_warnings():
    #    warnings.simplefilter("error")

    #stereo_scalability_new(d=2)
    #stereo_scalability_new(d=3)

    stereo_tightness(d=2)
    stereo_tightness(d=3)