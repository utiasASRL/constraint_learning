from lifters.plotting_tools import *
from lifters.poly_lifters import Poly4Lifter, Poly6Lifter
from lifters.range_only_slam_lifters import RangeOnlyLifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

METHOD = "qr"

K = 3
lifters = [
    Stereo1DLifter(n_landmarks=K),
    RangeOnlyLifter(n_landmarks=K, d=2),
    RangeOnlyLifter(n_landmarks=K, d=3),
    Poly4Lifter(),
    Poly6Lifter(),
]
lifters += [
    Stereo2DLifter(n_landmarks=K, level=level) for level in Stereo2DLifter.LEVELS
]
lifters += [
    Stereo3DLifter(n_landmarks=K, level=level) for level in Stereo2DLifter.LEVELS
]


def save_matrix_results():
    eps = 1e-4
    for lifter in lifters:
        Y = lifter.generate_Y(factor=5)
        basis, S = lifter.get_basis(Y, eps=eps, method=METHOD)

        dirname = get_dirname()
        fname = f"{dirname}/svd_{lifter}.png"
        fig, ax = plot_singular_values(S, eps=eps)
        ax.set_title(f"{lifter}")
        savefig(fig, fname)

        A_list = lifter.generate_matrices(basis)
        Q, y = lifter.get_Q(noise=0.1)
        if len(A_list) > 0:
            partial_plot_and_save(lifter, Q, A_list, save=True)


if __name__ == "__main__":
    save_matrix_results()
