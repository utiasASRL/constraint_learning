import numpy as np

from lifters.plotting_tools import make_dirs_safe
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.stereo_lifter import StereoLifter
from utils import get_fname

METHOD = "qr"
NOISE_DICT = dict(zip(range(5), np.logspace(-3, 1, 5)))


def run_noise_study(
    lifter, A_list, noise=1e-3, n_shuffles=0, n_seeds=1, fname="", verbose=False
):
    from copy import deepcopy

    import pandas as pd
    from progressbar import ProgressBar

    from solvers.common import find_local_minimum, solve_dual

    data = []

    seeds = range(n_seeds)
    for j, seed in enumerate(seeds):
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

            p = ProgressBar(max_value=len(A_shuffle))

            if len(A_shuffle) > 200:
                indices = list(
                    range(1, len(A_shuffle))[-200:-50:10]
                )  # ca. 20 datapoints
                indices += list(range(1, len(A_shuffle) + 1)[-50:])  # 50 datapoints
            else:
                indices = range(1, len(A_shuffle) + 1)[-200:]

            for i in indices:
                # print(f"adding {i}/{len(A_shuffle)}")
                # solve dual
                dual_cost, H, status = solve_dual(
                    Q, A_shuffle[:i], tol=tol, verbose=verbose
                )
                # print(f"status: {status}, dual_cost: {dual_cost}")

                eigs = np.linalg.eigvalsh(H) if H is not None else None
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

                if i % 10 == 0:
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


if __name__ == "__main__":
    import itertools

    eps = 1e-4
    method = "qr"
    noise = 0.0

    from lifters.poly_lifters import Poly4Lifter, Poly6Lifter

    lifters = [Poly4Lifter(), Poly6Lifter()]
    for lifter in lifters:
        Y = lifter.generate_Y(factor=3)
        basis, S = lifter.get_basis(Y, eps=eps, method=method)
        A_list = lifter.generate_matrices(basis)

        params = dict(
            lifter=lifter,
            A_list=A_list,
            noise=noise,
            n_seeds=1,
            n_shuffles=0,
            fname="",
        )
        run_noise_study(**params, verbose=False)


elif False:  # __name__ == "__main__":
    import itertools

    noises = NOISE_DICT.keys()
    level_list = ["no", "urT"]
    d_list = [1, 2, 3]

    # fixed
    n_landmarks = 3
    eps = 1e-4
    n_seeds = 1
    n_shuffles = 0
    method = "qr"
    for d, level in itertools.product(d_list, level_list):
        if d == 1:
            lifter = Stereo1DLifter(n_landmarks=n_landmarks)
            if level == "urT":
                print("skipping Lasserre for 1D")
                continue
        elif d == 2:
            lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level)
        elif d == 3:
            if level == "urT":
                print("skipping Lasserre for 3D cause too slow")
                continue
            lifter = Stereo3DLifter(n_landmarks=n_landmarks, level=level)
        else:
            raise ValueError(d)

        Y = lifter.generate_Y(factor=3)
        basis, S = lifter.get_basis(Y, eps=eps, method=method)
        A_list = lifter.generate_matrices(basis)

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
