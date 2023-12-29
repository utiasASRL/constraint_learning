import numpy as np
import scipy.sparse as sp

import matplotlib.pylab as plt


def plot_problem(problem, t_lims=[-1, 1]):
    Q = problem["C"]

    ts = np.linspace(*t_lims, 20)
    ys = [(t ** np.arange(4)).T @ Q @ t ** np.arange(4) for t in ts]

    fig, ax = plt.subplots()
    ax.plot(ts, ys)
    t = problem["x_cand"][1]
    ax.scatter([t], t ** np.arange(4) @ Q @ t ** np.arange(4))
    plt.show()


def get_problem():
    # Sixth order polynomial that requires redundant constraints to solve
    Q = np.array(
        [
            [5.0000, 1.3167, -1.4481, 0],
            [1.3167, -1.4481, 0, 0.2685],
            [-1.4481, 0, 0.2685, -0.0667],
            [0, 0.2685, -0.0667, 0.0389],
        ]
    )
    Constraints = []
    A = sp.csc_array((4, 4))  # w^2 = 1
    A[0, 0] = 1
    Constraints += [(A, 1.0)]
    A = sp.csc_array((4, 4))  # x^2 = x*x
    A[2, 0] = 1 / 2
    A[0, 2] = 1 / 2
    A[1, 1] = -1
    Constraints += [(A, 0.0)]
    A = sp.csc_array((4, 4))  # x^3 = x^2*x
    A[3, 0] = 1
    A[0, 3] = 1
    A[1, 2] = -1
    A[2, 1] = -1
    Constraints += [(A, 0.0)]
    A = sp.csc_array((4, 4))  # x^3*x = x^2*x^2
    A[3, 1] = 1 / 2
    A[1, 3] = 1 / 2
    A[2, 2] = -1
    Constraints += [(A, 0.0)]

    # Candidate solution
    x_cand = np.array([[1.0000, -1.4871, 2.2115, -3.2888]]).T

    # Dual optimal
    mults = -np.array([[-3.1937], [2.5759], [-0.0562], [0.8318]])

    return dict(Constraints=Constraints, Q=Q, x_cand=x_cand, opt_mults=mults)


if __name__ == "__main__":
    fname = "certifiable-tools/_test/test_prob_9.pkl"
    from problem_utils import save_test_problem

    prob = get_problem()
    save_test_problem(**prob, fname=fname)

    plot_problem(prob, t_lims=[-3, 3])
