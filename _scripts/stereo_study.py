import itertools

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg') # non-interactive
#plt.ioff()

from experiments import run_scalability_new, run_oneshot_experiment, tightness_study, plot_scalability, save_table

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from utils.plotting_tools import savefig


def stereo_tightness(d=2):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    if d == 2:
        n_landmarks = 3
    elif d == 3:
        n_landmarks = 4
    seed = 0
    # plots = ["tightness", "svd", "matrices", "templates"]
    #plots = ["matrices", "templates"]
    #plots = ["matrices", "templates", "svd", "tightness"]
    plots = ["matrices"]
    levels = ["no", "urT"]

    # parameter_levels = ["ppT"] #["no", "p", "ppT"]
    param_level = "no"
    for level in levels:
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
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

        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"

        if d == 2:
            run_oneshot_experiment(learner, fname_root, plots, tightness="cost", add_original=True)
        elif d == 3:
            run_oneshot_experiment(learner, fname_root, plots, tightness="cost", add_original=False)


def stereo_scalability_new(d=2):
    n_landmarks_list = [10, 15, 20]
    n_seeds = 3
    level = "urT"
    param_level = "ppT"

    n_landmarks = d+1

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
    run_scalability_new(learner, param_list=n_landmarks_list, n_seeds=n_seeds)


def stereo_scalability(d=2):
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
    """
    # n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    n_seeds = 1
    n_landmarks_list = [4, 5, 6]

    level = "urT"
    param_level = "no"
    # param_level = "ppT"

    # variable_list = None
    df_data = []
    for seed, n_landmarks in itertools.product(range(n_seeds), n_landmarks_list):
        print(f"================== N={n_landmarks},seed={seed} ======================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
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
        fname_root = f"_results/{lifter}"
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )

        # just use cost tightness because rank tightness was not achieved even in toy example
        times = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
        t_dict = times[0]

        idx_subset_original, idx_subset_reorder = tightness_study(
            learner, plot=False, tightness="cost"
        )
        t_dict["n req1"] = len(idx_subset_original)
        t_dict["n req2"] = len(idx_subset_reorder)
        t_dict["N"] = n_landmarks
        df_data.append(t_dict)

    df = pd.DataFrame(df_data)

    fig, ax = plot_scalability(df, log=True)
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability.pdf")

    fig, ax = plot_scalability(df, log=True, start="n ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability_n.pdf")

    tex_name = fname_root + "_scalability.tex"
    save_table(df, tex_name)


if __name__ == "__main__":
    # import warnings
    # with warnings.catch_warnings():
    #    warnings.simplefilter("error")

    #stereo_scalability(d=3)
    stereo_tightness(d=3)
    stereo_tightness(d=2)
    stereo_scalability_new(d=2)

    # with open("temp.pkl", "rb") as f:
    #    learner = pickle.load(f)
    # learner.apply_templates()
