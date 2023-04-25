import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from lifters.stereo2d_lifter import Stereo2DLifter
from solvers.common import find_local_minimum, solve_sdp_cvxpy

# what method to use to find number of constraints.
# bnb: branch and abound
# all: brute force
# sparse: use sparse optimization

# METHOD = "bnb"
# METHOD = "all"
METHOD = "sparse"

TOL_SPARSE_LAMBDA = 1e-10


def get_tightness_study(
    lifter, n_shuffles, n_random_noise, noise, verbose=True, fname=""
):
    def branch_and_bound(left, right, costs):
        """perform branch and abound on number of constraints."""

        if verbose:
            print("window:", left, right)

        for idx in [left, right]:
            if costs[idx] == 0:
                dual_Xhat, dual_cost = solve_sdp_cvxpy(
                    Q, lifter.get_A_b_list(A_shuffled[:idx]), verbose=False
                )
                costs[idx] = dual_cost

        # tighten window until we have a good starting point.
        if costs[right] is None:
            right -= 1
            return branch_and_bound(left, right, costs)
        elif costs[left] is None:
            left += 1
            return branch_and_bound(left, right, costs)

        eps = 1e-8
        half = int(round((right + left) / 2))
        if half in [left, right]:
            return half
        # emergency stop
        if all([(v is None) or (v > 0) for v in costs.values()]):
            return None

        if costs[half] == 0:
            dual_Xhat, dual_cost = solve_sdp_cvxpy(
                Q, lifter.get_A_b_list(A_shuffled[:half]), verbose=False
            )
            costs[half] = dual_cost

        if costs[half] is None:
            # by default, search right first (arbitrary)
            r = branch_and_bound(half, right, costs)
            if r > half:
                return r
            assert r == half
            return branch_and_bound(left, r, costs)
        elif costs[half] > (costs[right] - eps):
            # return best index in left half
            return branch_and_bound(left, half, costs)
        else:
            # return best index in right half
            return branch_and_bound(half, right, costs)

    def try_all(left, right, costs):
        """perform branch and abound on number of constraints."""
        for idx in range(left, right):
            dual_Xhat, dual_cost = solve_sdp_cvxpy(
                Q, lifter.get_A_b_list(A_shuffled[:idx]), verbose=False
            )
            costs[idx] = dual_cost

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
        A_b_list = lifter.get_A_b_list(A_shuffled)
        for noise_seed in range(n_random_noise):
            print(f"   noise {noise_seed+1}/{n_random_noise}")
            np.random.seed(noise_seed)
            Q, y = lifter.get_Q(noise=noise)

            # increase how many constraints we add to the problem
            qcqp_that, qcqp_cost = find_local_minimum(lifter, y=y)

            costs = {
                i + 1: 0 for i in range(0, len(A_all) + 1)
            }  # key: number of constraints
            min_number = 1
            max_number = len(A_all) + 1

            if METHOD == "bnb":
                current_num = branch_and_bound(min_number, max_number, costs)
                A_b_list_here = A_b_list[:current_num]
            elif METHOD == "all":
                current_num = try_all(min_number, max_number, costs)
                A_b_list_here = A_b_list[:current_num]
            elif METHOD == "sparse":
                from solvers.sparse import solve_lambda

                xhat = lifter.get_x(qcqp_that)
                H, lamda = solve_lambda(Q, A_b_list, xhat)
                indices = np.argsort(np.abs(lamda))[::-1]
                A_b_list_here = [A_b_list[i] for i in indices]

            # solve again to get rank etc.
            dual_Xhat, dual_cost = solve_sdp_cvxpy(Q, A_b_list_here, verbose=False)
            rank = np.linalg.matrix_rank(dual_Xhat)

            gap = qcqp_cost - dual_cost
            results_dict = dict(
                shuffle_seed=shuffle_seed,
                random_noise_seed=noise_seed,
                dual_cost=dual_cost,
                costs=costs,
                qcqp_cost=qcqp_cost,
                rank=rank,
                gap=gap,
                num_constraints=current_num,
            )
            results_dict.update({names[i]: 1 for i in shuffle_idx[:current_num]})
            data.append(results_dict)

        if fname != "":
            df = pd.DataFrame(data)
            pd.to_pickle(df, fname)
            print("saved intermediate as", fname)
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    n_landmarks = 2
    noise = 1e-3
    n_shuffles = 20
    n_random_noise = 3

    levels = ["urT", "no"]
    for level in levels:
        lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level)

        fname = str(root / f"_results/check_constraints_{lifter}.pkl")
        df = get_tightness_study(
            lifter,
            n_shuffles=n_shuffles,
            n_random_noise=n_random_noise,
            noise=noise,
            fname=fname,
        )
        pd.to_pickle(df, fname)
        print("saved final as", fname)
