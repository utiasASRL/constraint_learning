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

n_landmarks = 3
d = 2


def mono_tightness():
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    plots = [
        "svd",
        "matrices",
        "matrix",
        "tightness",
        "templates",
    ]  # ["svd", "matrices"]

    for level in ["xwT"]:  # ["xxT", "xwT"]:
        all_variables = ["l", "x"] + [f"w_{i}" for i in range(n_landmarks)]
        if level == "xxT":
            all_variables.append("z")
        else:
            all_variables += [f"z_{i}" for i in range(n_landmarks)]
        variable_list = [all_variables]

        np.random.seed(seed)
        lifter = MonoLifter(
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list=variable_list,
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(
            learner, fname_root, plots, tightness="rank", add_original=True
        )


def range_only_scalability():
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
    """
    n_seeds = 3
    n_landmarks = 10
    n_positions_list = [3, 4, 5]
    # n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    # level = "no" # for range-only
    d = 3

    for level in ["quad", "no"]:
        df_data = []
        for seed, n_positions in itertools.product(range(n_seeds), n_positions_list):
            variable_list = [
                ["l"]
                + [f"x_{n}" for n in range(n_positions)]
                + [f"z_{n}" for n in range(n_positions)]
            ]
            print(f"===== {n_positions} ====")
            np.random.seed(seed)
            lifter = RangeOnlyLocLifter(
                n_positions=n_positions,
                n_landmarks=n_landmarks,
                d=d,
                level=level,
                W=None,
                variable_list=variable_list,
            )
            fname_root = f"_results/{lifter}"
            learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

            times = learner.run(verbose=True, use_known=False, plot=False)
            for t_dict in times:
                t_dict["N"] = n_positions
                df_data.append(t_dict)

        df = pd.DataFrame(df_data)

        fig, ax = plot_scalability(df)
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_scalability.pdf")


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
        run_scalability_new(
            learner,
            param_list=n_positions_list,
            n_seeds=n_seeds,
            vmin=1e-2,
            vmax=5,
            tightness="rank",
        )


if __name__ == "__main__":
    mono_tightness()
    # range_only_scalability_new()
