import pickle

import numpy as np
import matplotlib.pylab as plt

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter


def find_minimal_constraints(lifter, fname_root):
    """
    Find the set of minimal constraints required for tightness, using the one-shot approach.
    """
    try:
        with open(fname_root + ".pkl", "rb") as f:
            learner = pickle.load(f)

        assert isinstance(learner, Learner)
        # learner.save_tightness(fname_root)
        # learner.save_patterns(fname_root)

        minimal_subset = learner.generate_minimal_subset(reorder=False)
        learner.save_patterns(
            b_tuples=minimal_subset, fname_root=""  # fname_root + "_subset"
        )
        print("done")

    except FileNotFoundError:
        print(f"running experiment {fname_root}")
        variable_list = [
            ["l", "x"] + [f"z_{i}" for i in range(j)] for j in range(n_landmarks + 1)
        ]
        learner = Learner(lifter=lifter, variable_list=variable_list)
        learner.run()
        learner.save_tightness(fname_root="")
        learner.save_patterns(fname_root="")
        with open(fname_root + ".pkl", "wb") as f:
            pickle.dump(learner, f)
        print(f"saved as {fname_root}.pkl")


if __name__ == "__main__":
    import itertools
    max_vars = 2  # n_landmarks
    #seeds = range(10) 
    seeds = [0]
    d = 2

    n_landmarks = 10
    lifter_type = "stereo"
    level = "urT" # for stereo
    parameter_levels = ["ppT"] #["no", "p", "ppT"]

    #n_landmarks = 30
    #n_positions = 10
    #lifter_type = "range"
    #level = "no" # for range-only
    #level = "quad" # for range-only
    #parameter_levels = ["no"]

    for seed, param_level in itertools.product(seeds, parameter_levels):
        print(f"============= seed {seed} parameters {param_level} ================")
        np.random.seed(seed)
        if lifter_type == "stereo":
            if d == 2:
                lifter = Stereo2DLifter(
                    n_landmarks=n_landmarks, level=level, param_level=param_level
                )
            elif d == 3:
                lifter = Stereo3DLifter(
                    n_landmarks=n_landmarks, level=level, param_level=param_level
                )
        elif lifter_type == "range":
            lifter = RangeOnlyLocLifter(n_positions=n_positions, n_landmarks=n_landmarks, d=d, level=level, W=None)
        else:
            raise ValueError(lifter_type)
        learner = Learner(lifter=lifter, variable_list=lifter.VARIABLE_LIST)

        fname_root = f"_results/{lifter}_seed{seed}"
        learner.run(verbose=True, use_known=False)

        df = learner.get_sorted_df()
        title = f"{param_level} parameters, {level} level, seed {seed}"
        learner.save_sorted_patterns(df, fname_root=fname_root, title=title)
        learner.save_tightness(fname_root=fname_root, title=title)
        #learner.save_patterns(
        #    fname_root=fname_root, title=title, with_parameters=False
        #)

    plt.show()
    print("done")
