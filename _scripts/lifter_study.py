import pickle
import itertools

import numpy as np
import matplotlib.pylab as plt

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

def range_only_tightness():
    """
    Find the set of minimal constraints required for tightness, using the one-shot approach.
    """
    n_landmarks = 10
    d = 3
    seed = 0
    level_names = {
        "no": "$z_n$",
        "quad": "$\\boldsymbol{y}_n$",
    }

    for level in ["no", "quad"]:
        n_positions = 2 if level == "quad" else 4
        variable_list = [["l"] + [f"x_{i}" for i in range(n_positions)] + [f"z_{i}" for i in range(n_positions)]]
        np.random.seed(seed)
        lifter = RangeOnlyLocLifter(n_positions=n_positions, n_landmarks=n_landmarks, d=d, level=level, W=None, variable_list=variable_list)
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, apply_patterns=False)

        fname_root = f"_results/{lifter}_seed{seed}"
        learner.run(verbose=True, use_known=False, plot=True)

        fig = plt.gcf()
        ax = plt.gca()
        ax.legend(loc="lower left")
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_svd.png")


        patterns_poly = learner.generate_patterns_poly(factor_out_parameters=True)
        df = learner.get_sorted_df(patterns_poly=patterns_poly)
        title = f"substitution level: {level_names[level]}"
        fig, ax = learner.save_sorted_patterns(df, title=title, drop_zero=True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(5*w/h, 5)
        savefig(fig, fname_root + "_patterns.png")

        fig, ax = learner.save_matrices_poly(A_matrices=learner.A_matrices[:7])
        w, h = fig.get_size_inches()
        fig.set_size_inches(10*w/h, 10)
        savefig(fig, fname_root + "_matrices.png")
        
        fig_cost, ax_cost = plt.subplots()
        ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k")
        fig_cost.set_size_inches(5, 5)
        # investigate tightness before and after reordering

        fig_eigs1, ax_eigs1 = plt.subplots()
        fig_eigs1.set_size_inches(5, 5)
        learner.generate_minimal_subset(reorder=False, ax_cost=ax_cost, ax_eigs=ax_eigs1)

        fig_eigs2, ax_eigs2 = plt.subplots()
        fig_eigs2.set_size_inches(5, 5)
        A_matrices_sorted = learner.generate_minimal_subset(reorder=True, ax_cost=ax_cost, ax_eigs=ax_eigs2)

        fig, ax = learner.save_matrices_poly(A_matrices=A_matrices_sorted)
        w, h = fig.get_size_inches()
        fig.set_size_inches(10*w/h, 10)
        savefig(fig, fname_root + "_matrices_sorted.png")

        ax_cost.legend(["QCQP cost", "dual cost, original ordering", "dual cost, new ordering"], loc="lower right")
        ax_eigs1.legend(loc="upper right", title="number of added\n constraints")
        ax_eigs1.set_title("original order")
        ax_eigs2.legend(loc="upper right", title="number of added\n constraints")
        ax_eigs2.set_title("sorted by dual values")

        savefig(fig_cost, fname_root + "_tightness-cost.png")
        savefig(fig_eigs1, fname_root + "_tightness-eigs-original.png")
        savefig(fig_eigs2, fname_root + "_tightness-eigs-sorted.png")

    plt.show()
    print("done")

def range_only_scalability():
    """
    Find the set of minimal constraints required for tightness, using the one-shot approach.
    """
    n_landmarks = 10
    n_positions_list = [3, 4, 5] 
    #n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    #level = "no" # for range-only
    level = "no" # for range-only
    d = 3

    for level in ["quad", "no"]:
        n_seeds = 10
        variable_list = None
        df_data = []
        for seed, n_positions in itertools.product(range(n_seeds), n_positions_list):
            print(f"===== {n_positions} ====")
            np.random.seed(seed)
            lifter = RangeOnlyLocLifter(n_positions=n_positions, n_landmarks=n_landmarks, d=d, level=level, W=None, variable_list=variable_list)
            fname_root = f"_results/{lifter}"
            learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

            times = learner.run(verbose=True, use_known=False, plot=False)
            for t_dict in times:
                t_dict["N"] = n_positions
                df_data.append(t_dict)

        import pandas as pd
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator

        df = pd.DataFrame(df_data)
        df.drop(inplace=True, columns=["variables"])
        df_plot = df.melt(id_vars=["N"], value_vars=df.columns, value_name="time [s]", var_name="operation")
        fig, ax = plt.subplots()
        sns.lineplot(df_plot, x="N", y="time [s]", hue="operation", ax=ax)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_scalability.png")

def stereo_tightness():
    import itertools
    #seeds = range(10) 
    seeds = [0]
    d = 2

    n_landmarks = 10
    level = "urT" # for stereo
    parameter_levels = ["ppT"] #["no", "p", "ppT"]
    for seed, param_level in itertools.product(seeds, parameter_levels):
        print(f"============= seed {seed} parameters {param_level} ================")
        np.random.seed(seed)
        if d == 2:
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks, level=level, param_level=param_level
            )
        elif d == 3:
            lifter = Stereo3DLifter(
                n_landmarks=n_landmarks, level=level, param_level=param_level
            )
        learner = Learner(lifter=lifter, variable_list=lifter.VARIABLE_LIST)

        fname_root = f"_results/{lifter}_seed{seed}"
        learner.run(verbose=True, use_known=False)

        fig_cost, ax_cost = plt.subplots()
        ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k")
        fig_eigs, ax_eigs = plt.subplots()
        A_matrices_sorted = learner.generate_minimal_subset(reorder=True, ax_cost=ax_cost, ax_eigs=ax_eigs)
        savefig(fig_cost, fname_root + "_tightness-cost.png")
        savefig(fig_eigs, fname_root + "_tightness-ranks.png")


        from poly_matrix import PolyMatrix
        A_matrices_patterns = []
        for key, bvec in learner.b_tuples:
            i, mat_vars = key
            var_dict = lifter.get_var_dict(mat_vars)
            ai = lifter.get_reduced_a(bvec, var_subset=var_dict)
            Ai = lifter.get_mat(ai)
            A_poly = PolyMatrix.init_from_sparse(Ai, var_dict)
            A_matrices_patterns.append((mat_vars, A_poly))
        learner.save_matrices_poly(A_matrices=A_matrices_patterns, fname_root=fname_root)

        learner.save_matrices_poly(A_matrices=A_matrices_sorted, fname_root=fname_root)

        patterns_poly = learner.generate_patterns_poly(factor_out_parameters=True)
        df = learner.get_sorted_df(patterns_poly=patterns_poly)
        title = f"{param_level} parameters, {level} level, seed {seed}"
        learner.save_sorted_patterns(df, fname_root=fname_root, title=title)

        learner.save_tightness(fname_root=fname_root, title=title)

        #learner.save_patterns(
        #    fname_root=fname_root, title=title, with_parameters=False
        #)

    plt.show()

if __name__ == "__main__":
    range_only_tightness()
    #range_only_scalability()