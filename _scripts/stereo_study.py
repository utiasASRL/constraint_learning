from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.plotting_tools import *

from utils import get_fname

METHOD = "qr"


def run_dimension_study(level, eps, fname, plot=False):
    import itertools
    import pandas as pd
    from progressbar import ProgressBar

    if fname != "":
        print(f"saving to {fname}")

    d_list = [2, 3]
    K_list = range(1, 6)

    data = []
    n = len(d_list) * len(K_list)
    p = ProgressBar(max_value=n)
    i = 0
    for d, K in itertools.product(d_list, K_list):
        p.update(i)
        i += 1

        if d == 2:
            lifter = Stereo2DLifter(n_landmarks=K, level=level)
        else:
            lifter = Stereo3DLifter(n_landmarks=K, level=level)

        Y = lifter.generate_Y(factor=5)
        basis, S = lifter.get_basis(Y, eps=eps, method=METHOD)
        A_list = lifter.generate_matrices(basis)

        if plot:
            fig, ax = plot_singular_values(S, eps=eps)
            ax.set_title(f"d={d}, K={K}")
            # plot_matrices(A_list, colorbar=False)
            plt.show()

        A_known = lifter.get_A_known()

        known = len(A_known)
        found = len(A_list)
        data.append(
            dict(
                d=d,
                landmarks=K,
                variables=lifter.N,
                substitutions=lifter.M,
                known=known,
                missing=found - known,
                found=found,
            )
        )

    df = pd.DataFrame(data)
    if fname != "":
        make_dirs_safe(fname)
        df.to_pickle(fname)
        print(f"saved as {fname}")


if __name__ == "__main__":
    fname = get_fname("stereo_study")
    params = dict(fname=fname, level=0, eps=1e-7)
    run_dimension_study(**params)

    fname = get_fname("stereo_study_lasserre")
    params = dict(fname=fname, level=3, eps=1e-4)
    run_dimension_study(**params)
