import pickle

import numpy as np

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter


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
    n_landmarks = 3
    max_vars = 3  # n_landmarks
    n_seeds = 1

    # doesn't get to tightness
    variable_list = [
        ["l", "x"] + [f"z_{i}" for i in range(j)] for j in range(max_vars + 1)
    ]

    # gets to tightness!
    variable_list = [
        ["l"] + [f"z_{i}" for i in range(j)] for j in range(1, max_vars + 1)
    ]

    # parameter_dict = {"without": False}
    # parameter_levels = ["no", "p", "ppT"]  # , "p"]  # ["no", "p", "ppT"]
    parameter_levels = ["no", "p", "ppT"]
    for seed in range(n_seeds):
        for param_level in parameter_levels:
            print(f"============= parameters {param_level} ================")
            np.random.seed(seed)
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks, level="urT", param_level=param_level
            )
            learner = Learner(lifter=lifter, variable_list=variable_list)

            fname_root = f"_results/{lifter}_seed{seed}"
            learner.run()

            df = learner.get_sorted_df()
            title = f"{param_level} parameters, {variable_list[-1]}, seed {seed}"
            learner.save_df(df, fname_root=fname_root, title=title)
            learner.save_tightness(fname_root=fname_root, title=title)
            learner.save_patterns(
                fname_root=fname_root, title=title, with_parameters=False
            )

    import matplotlib.pylab as plt

    plt.show()
    print("done")
