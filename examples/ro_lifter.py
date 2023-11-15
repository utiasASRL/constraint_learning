import numpy as np
import matplotlib.pylab as plt

from lifters.range_only_lifters import RangeOnlyLocLifter as Lifter
from utils.plotting_tools import savefig


def get_problem(t_init, centered=False, level="no"):
    # Create lifter
    np.random.seed(0)
    n_landmarks = 4
    d = 2
    lifter = Lifter(
        n_positions=t_init.shape[0], n_landmarks=n_landmarks, d=d, level=level
    )
    lifter.landmarks = np.c_[
        np.ones(n_landmarks) + np.random.rand(4) * 0.5, np.arange(n_landmarks)
    ]
    lifter.theta = np.hstack(
        [np.full(n_positions, 0.3)[:, None], np.linspace(1, 3, n_positions)[:, None]]
    ).flatten()

    Q, y = lifter.get_Q()

    from auto_template.learner import Learner

    learner = Learner(lifter=lifter, variable_list=[["h", "x_0", "z_0"]])
    learner.run()
    Constraints = learner.get_A_b_list()

    t, *_ = lifter.local_solver(t_init=t_init, y=y)

    if centered:
        D = lifter.get_D(t)
        Q = D.T @ Q @ D
        x_hat = np.zeros(lifter.get_dim_x())
        x_hat[0] = 1.0
    else:
        x_hat = lifter.get_x(theta=t)
    return dict(Q=Q, Constraints=Constraints, x_cand=x_hat), lifter


def plot_problem(prob, lifter, fname=""):
    Q = prob["Q"]

    theta_hat = prob["x_cand"][1 : 1 + lifter.theta.size].reshape(
        lifter.n_positions, -1
    )
    theta_gt = lifter.theta.reshape(lifter.n_positions, -1)
    n_plots = min(3, lifter.n_positions)
    xx, yy = np.meshgrid(np.arange(-1, 4, 0.1), np.arange(-1, 4, 0.1))
    fig, ax = plt.subplots(1, n_plots, squeeze=False)
    for n in range(n_plots):
        zz = np.empty(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(yy.shape[1]):
                theta = theta_gt.copy()
                theta[n, :] = xx[i, j], yy[i, j]
                x = lifter.get_x(theta=theta)
                zz[i, j] = x.T @ Q @ x

        ax[0, n].pcolormesh(xx, yy, np.log10(zz))
        ax[0, n].scatter(
            lifter.landmarks[:, 0], lifter.landmarks[:, 1], color="white", marker="o"
        )
        ax[0, n].scatter(theta_gt[n, 0], theta_gt[n, 1], color="white", marker="*")
        ax[0, n].scatter(theta_hat[n, 0], theta_hat[n, 1], color="green", marker="*")
    plt.show()
    if fname != "":
        savefig(fig, fname)


if __name__ == "__main__":
    from problem_utils import save_test_problem

    # number = 10
    # level = "no"

    n_positions = 1
    number = 11
    level = "quad"

    # n_positions = 3
    # number = 12
    # level = "quad"

    # n_positions = 10
    # number = 13
    # level = "quad"

    for centered in [False, True]:
        appendix = "c" if centered else ""

        prob, lifter = get_problem(
            t_init=np.zeros((n_positions, 2)), centered=centered, level=level
        )
        fname = f"certifiable-tools/_test/test_prob_{number}G{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, lifter, fname=fname.replace(".pkl", ".png"))

        prob, lifter = get_problem(
            t_init=np.hstack(
                [np.full(n_positions, 2)[:, None], np.zeros((n_positions, 1))]
            ),
            centered=centered,
            level=level,
        )
        fname = f"certifiable-tools/_test/test_prob_{number}L{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, lifter, fname=fname.replace(".pkl", ".png"))
