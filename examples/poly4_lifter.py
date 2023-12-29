import numpy as np

import matplotlib.pylab as plt

from lifters.poly_lifters import Poly4Lifter as Lifter
from utils.plotting_tools import savefig


def plot_problem(problem, t_lims=[-2, 5], fname=""):
    lifter = Lifter()
    Q = problem["Q"]

    ts = np.arange(*t_lims, 0.1)
    ys = [lifter.get_x(t).T @ Q @ lifter.get_x(t) for t in ts]

    fig, ax = plt.subplots()
    ax.plot(ts, ys)
    t = problem["x_cand"][1]
    ax.scatter([t], problem["x_cand"].T @ Q @ problem["x_cand"])
    ax.set_yscale("symlog")
    if fname != "":
        savefig(fig, fname)


def get_problem(centered=False, t_init=-1):
    lifter = Lifter()
    Q, __ = lifter.get_Q()
    A_list = lifter.get_A_known()
    Constraints = lifter.get_A_b_list(A_list)

    t_hat, info, cost = lifter.local_solver(t_init)

    if centered:
        D = lifter.get_D(t_hat)
        Q = D.T @ Q @ D

        x_hat = np.zeros(Q.shape[0])
        x_hat[0] = 1.0
    else:
        x_hat = lifter.get_x(t_hat)
        for A_i in A_list[1:]:
            assert abs(x_hat.T @ A_i @ x_hat) <= 1e-10

    return dict(Q=Q, Constraints=Constraints, x_cand=x_hat)


if __name__ == "__main__":
    from problem_utils import save_test_problem

    number = 16
    for centered in [False, True]:
        appendix = "c" if centered else ""
        prob = get_problem(t_init=-5, centered=centered)
        fname = f"certifiable-tools/_examples/test_prob_{number}G{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, t_lims=[-4, 4], fname=fname.replace(".pkl", ".png"))

        prob = get_problem(t_init=2, centered=centered)
        fname = f"certifiable-tools/_examples/test_prob_{number}L{appendix}.pkl"
        save_test_problem(**prob, fname=fname)
        plot_problem(prob, t_lims=[-4, 4], fname=fname.replace(".pkl", ".png"))