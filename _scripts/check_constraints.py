import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from noise_study import run_noise_study

from lifters.plotting_tools import plot_matrices, plot_tightness, savefig
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from solvers.common import find_local_minimum, solve_sdp_cvxpy

# what method to use to find number of constraints.
# bnb: branch and abound
# all: brute force
# sparse: use sparse optimization

METHODS = ["all", "bnb"]

TOL_SPARSE_LAMBDA = 1e-10

# assume strong duality when relative gap is smaller
TOL_REL_GAP = 1e-3
TOL_ABS_GAP = 1e-5

# tolerance for nullspace basis vectors
EPS_LEARNED = 1e-7

# SOLVER = "CVXOPT"
SOLVER = "MOSEK"


def compare_constraints(lifter, fname=""):
    A_known = lifter.get_A_known()
    A_incr = lifter.get_A_learned(A_known=A_known, method="qrp", normalize=False)
    A_learned = lifter.get_A_learned()

    noises = np.logspace(-3, -1, 3)
    n_seeds = 3
    n_shuffles = 1
    for A_list, name in zip([A_incr[::-1], A_learned], ["incremental", "learned"]):
        print(f"checking {name} constraints...")
        lifter.test_constraints(A_list, errors="raise")

        for noise in noises:
            params = dict(
                lifter=lifter,
                A_list=A_list,
                noise=noise,
                n_seeds=n_seeds,
                n_shuffles=n_shuffles,
                fname="",
            )
            df = run_noise_study(**params, verbose=False, solver=SOLVER)
            fig, ax = plot_tightness(df)
            ax.set_title(f"{name}, noise {noise:.1e}")
            ax.set_ylim(1e-8, 1e-1)
            if fname != "":
                savefig(
                    fig, fname + f"_{name}_{str(noise).replace('.','-')}_{SOLVER}.png"
                )

    from math import ceil

    n_per_plot = min(len(A_learned), 10)
    for A_list, name in zip(
        [A_known, A_incr, A_learned], ["known", "incremental", "learned"]
    ):
        n_plots = ceil(len(A_list) / n_per_plot)
        for i in range(n_plots):
            fig, ax = plot_matrices(
                A_list,
                n_per_plot,
                start_idx=i * n_per_plot,
                colorbar=False,
                title=name,
                nticks=3,
            )
            if fname != "":
                savefig(fig, fname + f"_{name}{i}.png")
    plt.show()
    plt.close()


def get_tightness_study(
    lifter, n_shuffles, n_random_noise, noise, verbose=True, use_known=True, fname=""
):
    def branch_and_bound(left, right, costs):
        """perform branch and abound on number of constraints."""

        if verbose:
            print("window:", left, right)

        for idx in [left, right]:
            if costs[idx] == 0:
                dual_Xhat, info = solve_sdp_cvxpy(Q, A_b_list[: idx + 1], verbose=False)
                costs[idx] = info["cost"]

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
            assert costs[right] != 0
            return right
        # emergency stop
        if all([(v is None) or (v > 0) for v in costs.values()]):
            return None

        if costs[half] == 0:
            dual_Xhat, info = solve_sdp_cvxpy(Q, A_b_list[: half + 1], verbose=False)
            costs[half] = info["cost"]

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
            dual_Xhat, info = solve_sdp_cvxpy(Q, A_b_list[: idx + 1], verbose=False)
            dual_cost = info["cost"]
            costs[idx] = dual_cost if dual_cost else 0.0

    data = []
    for noise_seed in range(n_random_noise):
        print(f"noise {noise_seed+1}/{n_random_noise}")
        np.random.seed(noise_seed)

        lifter.generate_random_setup()
        if use_known:
            A_known = lifter.get_A_known()
        else:
            A_known = []
        A_all = lifter.get_A_learned(A_known=A_known, eps=EPS_LEARNED, normalize=False)
        n_learned = len(A_all) - len(A_known)
        Q, y = lifter.get_Q(noise=noise)

        A_b_list_all = lifter.get_A_b_list(A_all)
        # increase how many constraints we add to the problem
        qcqp_that, qcqp_cost = find_local_minimum(lifter, y=y)
        if qcqp_cost is None:
            print("Warning: could not solve local.")
            continue
        elif qcqp_cost < 1e-7:
            print("Warning: too low qcqp cost, numerical issues.")
            continue

        shuffle_idx = np.arange(len(A_all))
        for shuffle_seed in range(-1, n_shuffles):
            print(f"    shuffle {shuffle_seed+1}/{n_shuffles}")

            lamda = None
            if shuffle_seed == -1:
                from solvers.sparse import solve_lambda

                xhat = lifter.get_x(qcqp_that)
                H, lamda = solve_lambda(Q, A_b_list_all, xhat)
                if lamda is None:
                    print("Warning: problem doesn't have feasible solution!")
                    continue
                shuffle_idx = np.argsort(np.abs(lamda[1:]))[::-1]
                lamda = np.abs(lamda[1:][shuffle_idx])
            elif shuffle_seed == 0:
                shuffle_idx = np.arange(len(A_all))
            elif shuffle_seed > 0:
                np.random.seed(shuffle_seed)
                np.random.shuffle(shuffle_idx)

            min_number = 1
            max_number = len(A_all) + 1
            A_b_list = [A_b_list_all[0]] + [A_b_list_all[s + 1] for s in shuffle_idx]
            for method in METHODS:  # , "all"]:
                costs = {i + 1: 0 for i in range(0, len(A_all) + 1)}
                print(f"      solve with {method}")
                if method == "bnb":
                    current_num = branch_and_bound(min_number, max_number, costs)
                elif method == "all":
                    try_all(min_number, max_number, costs)
                    # find minimum required constraints for strong duality
                    current_num = np.where(
                        [
                            (qcqp_cost - c) / qcqp_cost > TOL_REL_GAP
                            for c in costs.values()
                        ]
                    )[0]
                    current_num = current_num[0] + 1 if len(current_num) else None
                dual_cost1 = costs[current_num] if current_num else 0.0
                A_b_list_here = A_b_list[: current_num + 1] if current_num else A_b_list

                # solve again to get rank etc.
                dual_Xhat, info = solve_sdp_cvxpy(Q, A_b_list_here, verbose=False)
                dual_cost = info["cost"]

                rel_error = abs(dual_cost - dual_cost1) / dual_cost
                if rel_error > 1e-3:
                    print(
                        f"Warning, mismatch between dual errors: {dual_cost, dual_cost1}. Skipping..."
                    )
                    continue
                rank = np.linalg.matrix_rank(dual_Xhat)
                gap = qcqp_cost - dual_cost
                results_dict = dict(
                    shuffle_seed=shuffle_seed,
                    random_noise_seed=noise_seed,
                    dual_cost=dual_cost,
                    qcqp_cost=qcqp_cost,
                    method=method,
                    gap=gap,
                    rank=rank,
                    costs=costs,
                    num_constraints=current_num,
                    lamda=lamda,
                )
                data.append(results_dict)

        if fname != "":
            df = pd.DataFrame(data)
            pd.to_pickle(df, fname)
            print("saved intermediate as", fname)
    df = pd.DataFrame(data)
    return df


def stereo_2d_study(fname_root):
    n_landmarks = 3
    noise = 1e-2
    n_shuffles = 3
    n_random_noise = 5

    levels = ["urT"]  # ["urT", "no"]
    for level in levels:
        lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level)

        fname = fname_root + f"_{lifter}_learned.pkl"
        df = get_tightness_study(
            lifter,
            n_shuffles=n_shuffles,
            n_random_noise=n_random_noise,
            noise=noise,
            fname=fname,
            use_known=False,
        )


def stereo_1d_study(fname_root):
    n_landmarks = 3
    noise = 1e-3
    n_shuffles = 3  # includes -1 and 0.
    n_random_noise = 5

    lifter = Stereo1DLifter(n_landmarks=n_landmarks)

    fname = fname_root + f"_{lifter}"
    compare_constraints(lifter, fname)

    fname = fname_root + f"_{lifter}.pkl"
    df = get_tightness_study(
        lifter,
        n_shuffles=n_shuffles,
        n_random_noise=n_random_noise,
        noise=noise,
        fname=fname,
    )
    pd.to_pickle(df, fname)
    print("saved final as", fname)


if __name__ == "__main__":
    import warnings
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # fname_root = str(root / "_results/constraints")
    # stereo_1d_study(fname_root=fname_root)

    fname_root = str(root / "_results/experiments")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stereo_2d_study(fname_root=fname_root)
