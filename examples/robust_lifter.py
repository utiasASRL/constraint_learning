import numpy as np
import matplotlib.pylab as plt

from lifters.wahba_lifter import WahbaLifter as Lifter
from utils.plotting_tools import savefig


def get_problem(robust=True):
    if robust:
        level = "xwT"
    else:
        level = "no"
    # Create lifter
    np.random.seed(0)
    n_landmarks = 4
    d = 3
    if robust:
        n_outliers = 1
    else:
        n_outliers = 0

    lifter = Lifter(
        d=d,
        n_landmarks=n_landmarks + n_outliers,
        robust=robust,
        n_outliers=n_outliers,
        level=level,
        variable_list=None,
    )
    Q, y = lifter.get_Q()

    from auto_template.learner import Learner

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
    learner.run()
    Constraints = learner.get_A_b_list()

    t0 = lifter.get_vec_around_gt(delta=0)  # initialize at gt
    t, *_ = lifter.local_solver(t0, y=y)
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

    for robust, number in zip([False, True], [14, 15]):
        prob, lifter = get_problem(robust=robust)
        fname = f"certifiable-tools/_examples/test_prob_{number}G.pkl"
        save_test_problem(**prob, fname=fname)
        # plot_problem(prob, lifter, fname=fname.replace(".pkl", ".png"))
