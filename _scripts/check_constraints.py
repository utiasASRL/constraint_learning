import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from lifters.stereo2d_lifter import Stereo2DLifter
from solvers.common import find_local_minimum, solve_sdp_cvxpy

if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    noise = 1e-3
    n_landmarks = 2
    n_shuffles = 20
    n_random_noise = 3

    lifter = Stereo2DLifter(n_landmarks=n_landmarks)
    A_known = lifter.get_A_known()
    A_all = lifter.get_A_learned(A_known=A_known)
    n_learned = len(A_all) - len(A_known)
    names = [f"A{i}_l" for i in range(n_learned)]
    names += [f"A{i}_k" for i in range(len(A_known))]
    shuffle_idx = np.arange(len(A_all))

    data = []
    for shuffle_seed in range(n_shuffles):
        print(f"shuffle {shuffle_seed+1}/{n_shuffles}")
        np.random.seed(shuffle_seed)
        if shuffle_seed > 0:
            np.random.shuffle(shuffle_idx)

        A_shuffled = [A_all[s] for s in shuffle_idx]
        for noise_seed in range(n_random_noise):
            print(f"   noise {noise_seed+1}/{n_random_noise}")
            np.random.seed(shuffle_seed)
            Q, y = lifter.get_Q(noise=noise)

            # increase how many constraints we add to the problem
            qcqp_xhat, qcqp_cost = find_local_minimum(lifter, y=y)

            current_cost = None
            current_num = len(A_all) + 1
            current_left = 0

            # perform bisection search on number of constraints.
            counter = 0
            while counter < len(A_all):  # safety timeout
                dual_Xhat, dual_cost = solve_sdp_cvxpy(
                    Q, lifter.get_A_b_list(A_shuffled[:current_num]), verbose=False
                )
                # at the start, or when we find a cost that is close to current max, look in the left half.
                if (current_cost is None) or (
                    (dual_cost - current_cost) / current_cost < 1e-6
                ):
                    current_right = current_num
                # else, look in the right half.
                else:
                    current_left = current_num

                counter += 1
                current_num = int(round((current_right - current_left) / 2))
                if current_num in [current_right, current_left]:
                    break

            rank = np.linalg.matrix_rank(dual_Xhat)
            gap = qcqp_cost - dual_cost
            results_dict = dict(
                shuffle_seed=shuffle_seed,
                random_noise_seed=noise_seed,
                dual_cost=dual_cost,
                qcqp_cost=qcqp_cost,
                rank=rank,
                gap=gap,
                num_constraints=current_num,
            )
            results_dict.update({names[i]: 1 for i in shuffle_idx[:current_num]})
            data.append(results_dict)
    df = pd.DataFrame(data)
    fname = str(root / "_results/check_constraints.pkl")
    pd.to_pickle(df, fname)
    print("wrote as", fname)
