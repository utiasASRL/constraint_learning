import numpy as np

from lifters.plotting_tools import make_dirs_safe
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from solvers.common import find_local_minimum, solve_dual, solve_sdp_cvxpy
from utils import get_fname

METHOD = "qrp"
NOISE_DICT = {0: 1e-1}  # dict(zip(range(5), np.logspace(-3, 1, 5)))
USE_MATRICES = "all"  # "first"

SOLVER = "MOSEK"


def run_noise_study(
    lifter,
    A_list,
    noise=1e-3,
    n_shuffles=0,
    n_seeds=1,
    fname="",
    verbose=False,
    solver=SOLVER,
):
    from copy import deepcopy

    import pandas as pd
    from progressbar import ProgressBar

    data = []

    seeds = range(n_seeds)
    for j, seed in enumerate(seeds):
        print(f"seed {j+1}/{len(seeds)}")
        # generate random measurements
        np.random.seed(seed)
        Q, y = lifter.get_Q(noise=noise)

        # find global optimum
        xhat, local_cost = find_local_minimum(lifter, y, delta=noise, verbose=verbose)
        if xhat is None:
            print("Warning: local didn't solve!!")
            continue

        # try to make the SDP only as hard as it has to be. Given the primal cost
        # we know roughly what accuracy we need to see if we are tight.
        # we clip that to [1e-9, 1e-4] to make sure we're reasonable.
        tol = min(max(1e-9, local_cost / 10), 1e-4)
        # tol = 1e-5

        A_shuffle = deepcopy(A_list)
        for shuffle in range(n_shuffles + 1):
            if shuffle > 0:
                # print(f"using shuffle {shuffle}")
                np.random.seed(shuffle)
                np.random.shuffle(A_shuffle)
            else:
                # print("using original")
                pass

            print(f"  shuffle {shuffle}/{n_shuffles}")
            p = ProgressBar(max_value=len(A_shuffle))

            if len(A_shuffle) > 200:
                if USE_MATRICES == "last":
                    indices = list(
                        range(1, len(A_shuffle))[-200:-50:10]
                    )  # ca. 20 datapoints
                    indices += list(range(1, len(A_shuffle) + 1)[-50:])  # 50 datapoints
                elif USE_MATRICES == "uniform":
                    indices = np.linspace(1, len(A_shuffle), 20).astype(int)
                elif USE_MATRICES == "first":
                    indices = np.linspace(1, min(len(A_shuffle), 400), 41).astype(int)
                elif USE_MATRICES == "all":
                    indices = range(1, len(A_shuffle) + 1)
            else:
                indices = range(1, len(A_shuffle) + 1)

            for i_count, i in enumerate(indices):
                # print(f"adding {i}/{len(A_shuffle)}")
                # solve dual
                A_b_list = lifter.get_A_b_list(A_shuffle[:i])
                H, info = solve_sdp_cvxpy(Q, A_b_list, tol=1e-10, solver=solver)
                dual_cost = info["cost"]
                status = ""
                # dual_cost, H, status = solve_dual(
                #    Q, A_shuffle[:i], tol=tol, verbose=verbose
                # )

                if H is None:
                    eigs = None
                elif type(H) == np.ndarray:
                    eigs = np.linalg.eigvalsh(H) if H is not None else None
                else:
                    try:
                        import scipy.sparse.linalg as spl

                        eigs = spl.eigs(H, which="SA", k=5)
                    except Exception as e:
                        print(e)
                        eigs = np.linalg.eigvalsh(H.toarray())
                data.append(
                    {
                        "n": i,
                        "seed": j,
                        "shuffle": shuffle,
                        "dual cost": dual_cost,
                        "local cost": local_cost,
                        "eigs": eigs,
                        "status": status,
                    }
                )
                p.update(i)

                if i_count % 5 == 0:
                    df = pd.DataFrame(data)
                    if fname != "":
                        make_dirs_safe(fname)
                        df.to_pickle(fname)
                        print(f"saved intermediate as {fname}")
    df = pd.DataFrame(data)
    if fname != "":
        make_dirs_safe(fname)
        df.to_pickle(fname)
        print(f"saved final as {fname}")
    return df


def run_poly_study():
    import itertools

    eps = 1e-4
    method = "qr"
    noise = 0.0

    from lifters.poly_lifters import Poly4Lifter, Poly6Lifter

    lifters = [Poly4Lifter(), Poly6Lifter()]
    for lifter in lifters:
        A_list = lifter.get_A_learned(eps=eps, method=method, factor=3)
        params = dict(
            lifter=lifter,
            A_list=A_list,
            noise=noise,
            n_seeds=1,
            n_shuffles=0,
            fname="",
        )
        run_noise_study(**params, verbose=False)


def run_stereo_study():
    import itertools

    noises = NOISE_DICT.keys()
    level_list = ["no", "urT"]
    d_list = [1, 2, 3]

    # fixed
    n_landmarks = 3
    eps = 1e-4
    n_seeds = 1
    n_shuffles = 0
    method = "qrp"
    for d, level in itertools.product(d_list, level_list):
        if d == 1:
            lifter = Stereo1DLifter(n_landmarks=n_landmarks)
            if level == "urT":
                print("skipping Lasserre for 1D")
                continue
        elif d == 2:
            lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level)
        elif d == 3:
            # if level == "urT":
            #    print("skipping Lasserre for 3D cause too slow")
            #    continue
            lifter = Stereo3DLifter(n_landmarks=n_landmarks, level=level)
        else:
            raise ValueError(d)

        A_list = lifter.get_A_learned(eps=eps, method=method, factor=3)
        for name in noises:
            noise = NOISE_DICT[name]
            fname = get_fname(f"study_stereo{d}d_{name}noise_{level}level")
            params = dict(
                lifter=lifter,
                A_list=A_list,
                noise=noise,
                n_seeds=n_seeds,
                n_shuffles=n_shuffles,
                fname=fname,
            )
            run_noise_study(**params, verbose=False)


def run_range_study():
    import itertools

    from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
    from lifters.range_only_slam2 import RangeOnlySLAM2Lifter

    lifter_types = {"slam1": RangeOnlySLAM1Lifter, "slam2": RangeOnlySLAM2Lifter}

    noises = NOISE_DICT.keys()
    d_list = [2, 3]

    # fixed
    n_landmarks = 5
    n_positions = 5
    eps = 1e-4
    n_seeds = 1
    n_shuffles = 0

    method = "qr"

    for d, (lifter_type, Lifter) in itertools.product(d_list, lifter_types.items()):
        lifter = Lifter(n_landmarks=n_landmarks, n_positions=n_positions, d=d)

        A_list = lifter.get_A_learned(eps=eps, method=method, factor=2)

        for name in noises:
            noise = NOISE_DICT[name]
            fname = get_fname(f"study_range{lifter_type}_{d}d_{name}noise_sparse")
            params = dict(
                lifter=lifter,
                A_list=A_list,
                noise=noise,
                n_seeds=n_seeds,
                n_shuffles=n_shuffles,
                fname=fname,
            )
            run_noise_study(**params, verbose=False)


if __name__ == "__main__":
    # run_poly_study()
    run_stereo_study()
    # run_range_study()
