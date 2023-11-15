from auto_template.learner import Learner
from lifters.poly_lifters import Poly6Lifter
from utils.plotting_tools import plot_matrix, import_plt, savefig

import numpy as np

plt = import_plt()


def plot_poly_examples():
    lifter = Poly6Lifter()

    learner = Learner(lifter, variable_list=lifter.VARIABLE_LIST, use_known=False)
    learner.run(plot=True)

    savefig(plt.gcf(), "_plots/toy_example_svd.png")

    A_learned = learner.get_A_b_list()
    A_list = [lifter.get_A0()] + lifter.get_A_known()

    basis = np.empty((lifter.get_dim_X(), len(A_list)))
    fig1, axs1 = plt.subplots(1, len(A_list))
    for i, A_i in enumerate(A_list):
        plot_matrix(A_i, ax=axs1[i], colorbar=False)
        if i == 0:
            axs1[i].set_title(f"$A_{i}$")
        elif i < len(A_list) - 1:
            axs1[i].set_title(f"$A_{i}$ (known)")
        else:
            axs1[-1].set_title(f"$B_{0}$ (known)")
        basis[:, i] = lifter.get_vec(A_i)
    fig1b, *_ = plot_matrix(basis, colorbar=True, discrete=True)

    basis = np.empty((lifter.get_dim_X(), len(A_list)))
    fig2, axs2 = plt.subplots(1, len(A_learned))
    for i, (A_i, b) in enumerate(A_learned):
        plot_matrix(A_i.toarray(), ax=axs2[i], colorbar=False)
        if i == 0:
            axs2[i].set_title(f"$A_{i}$")
        else:
            axs2[i].set_title(f"$A_{i}$ (learned)")
        basis[:, i] = lifter.get_vec(A_i)
    fig2b, *_ = plot_matrix(basis, colorbar=True, discrete=True)

    savefig(fig1, "_plots/toy_example_1.png")
    savefig(fig2, "_plots/toy_example_2.png")
    savefig(fig1b, "_plots/toy_example_1b.png")
    savefig(fig2b, "_plots/toy_example_2b.png")

    plt.show()
    print("done")

def plot_all_examples():



if __name__ == "__main__":
    # plot_poly_examples()
    plot_all_examples()
