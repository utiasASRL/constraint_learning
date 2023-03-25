from lifters.stereo_lifter import StereoLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.plotting_tools import *

from utils import get_fname

METHOD = "qr"


def save_matrix_results():
    eps = 1e-4

    d_list = [2, 3]
    K = 3
    for d in d_list:
        for level in StereoLifter.LEVELS:
            if d == 2:
                lifter = Stereo2DLifter(n_landmarks=K, level=level)
            elif d == 3:
                lifter = Stereo3DLifter(n_landmarks=K, level=level)
            else:
                raise ValueError(d)

            Y = lifter.generate_Y(factor=5)
            basis, S = lifter.get_basis(Y, eps=eps, method=METHOD)

            dirname = get_dirname()
            fname = f"{dirname}/svd_{lifter}.png"
            fig, ax = plot_singular_values(S, eps=eps)
            ax.set_title(f"level {level}")
            savefig(fig, fname)

            A_list = lifter.generate_matrices(basis)
            Q, y = lifter.get_Q(noise=0.1)
            if len(A_list) > 0:
                partial_plot_and_save(lifter, Q, A_list, save=True)


if __name__ == "__main__":
    save_matrix_results()
