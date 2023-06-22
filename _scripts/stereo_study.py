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

    for seed in range(2):
        # lifter = Stereo2DLifter(n_landmarks=n_landmarks, level="urT", add_parameters=False)
        # fname_root = f"_results/{lifter}_oneshot"
        # find_minimal_constraints(lifter, fname_root)

        max_vars = 1  # n_landmarks
        variable_list = [
            ["l", "x"] + [f"z_{i}" for i in range(j)] for j in range(max_vars + 1)
        ]
        # parameter_dict = {"without": False}
        parameter_dict = {"with": True, "without": False}
        for name, add_parameters in parameter_dict.items():
            np.random.seed(seed)
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks, level="urT", add_parameters=add_parameters
            )
            print(lifter.landmarks)
            learner = Learner(lifter=lifter, variable_list=variable_list)

            fname_root = f"_results/{lifter}_{name}_seed{seed}"
            learner.run()

            df = learner.get_sorted_df()
            title = f"{name} parameters, {variable_list[-1]}, seed {seed}"
            fig, ax = learner.save_df(df, fname_root=fname_root, title=title)

            # fig, ax = learner.save_patterns(fname_root=fname_root)
            # ax.set_title(f"{name} parameters, {variable_list[-1]}, seed {seed}")

            matrix_symbols = set(learner.patterns_poly.get_matrix().data.round(4))
            param_symbols = set(lifter.landmarks[:max_vars, :].flatten().round(4))

            lm = lifter.landmarks[:max_vars, :]
            lm_outer = np.outer(lm, lm)[np.triu_indices(lm.shape[1])].round(4)
            param_symbols_higher = set(lm_outer).union(-lm_outer)
            constants = np.array(
                [np.sqrt(2), 1.0, 1 / np.sqrt(2), 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]
            ).round(4)
            param_symbols_higher = (
                param_symbols_higher.union(param_symbols)
                .union(set(constants))
                .union(set(-constants))
            )
            inexplained = sorted(matrix_symbols.difference(param_symbols_higher))
            print(lifter.landmarks)
            print(inexplained)
            print(len(inexplained))

            print(
                f"============= {variable_list} {name} parameters: found {len(learner.A_matrices)} constraints ==============="
            )
    print("done")
