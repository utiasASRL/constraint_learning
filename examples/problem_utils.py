import numpy as np
import pickle


def save_test_problem(Q, Constraints, x_cand=None, fname="", **kwargs):
    test_problem_dict = {}
    test_problem_dict["Q"] = Q
    test_problem_dict["Constraints"] = Constraints
    test_problem_dict.update(kwargs)

    if x_cand is not None:
        test_problem_dict["x_cand"] = np.atleast_2d(x_cand).reshape((-1, 1))
    else:
        x_cand = np.zeros((Q.shape[0], 1))
        x_cand[0, 0] = 1.0
        test_problem_dict["x_cand"] = x_cand

    with open(fname, "wb") as f:
        pickle.dump(test_problem_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved testproblem as {fname}.")
