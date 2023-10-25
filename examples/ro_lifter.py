import numpy as np
import matplotlib.pylab as plt

from lifters.range_only_lifters import RangeOnlyLocLifter as Lifter
from utils.plotting_tools import savefig


def get_problem(t_init=[0, 0], centered=False, level="no"):
    # Create lifter
    np.random.seed(0)
    n_landmarks = 4
    d = 2
    lifter = Lifter(n_positions=1, n_landmarks=n_landmarks, d=d, level=level)
    lifter.landmarks = np.c_[
        np.ones(n_landmarks) + np.random.rand(4) * 0.5, np.arange(n_landmarks)
    ]
    lifter.theta = np.array([0.3, 1.0])

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

    xx, yy = np.meshgrid(np.arange(-1, 4, 0.1), np.arange(-1, 4, 0.1))
    zz = np.empty(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(yy.shape[1]):
            x = lifter.get_x(theta=np.array([xx[i, j], yy[i, j]]))
            zz[i, j] = x.T @ Q @ x

    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, np.log10(zz))
    ax.scatter(
        lifter.landmarks[:, 0], lifter.landmarks[:, 1], color="white", marker="o"
    )
    ax.scatter(lifter.theta[0], lifter.theta[1], color="white", marker="*")
    ax.scatter(prob["x_cand"][1], prob["x_cand"][2], color="green", marker="*")
    plt.show()
    if fname != "":
        savefig(fig, fname)


if __name__ == "__main__":
    from problem_utils import save_test_problem

    number = 10
    level = "no"

    # number = 11
    # level = "quad"

    for centered in [False, True]:
        appendix = "c" if centered else ""

        prob, lifter = get_problem(
            t_init=np.array([0, 0]), centered=centered, level=level
        )
        fname = f"certifiable-tools/_test/test_prob_{number}G{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, lifter, fname=fname.replace(".pkl", ".png"))

        prob, lifter = get_problem(
            t_init=np.array([2, 0]), centered=centered, level=level
        )
        fname = f"certifiable-tools/_test/test_prob_{number}L{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, lifter, fname=fname.replace(".pkl", ".png"))
