import pickle
import itertools

import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use('Agg')
plt.ioff()

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

def plot_scalability(df):
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    df.drop(inplace=True, columns=["variables"])
    df_plot = df.melt(id_vars=["N"], value_vars=df.columns, value_name="time [s]", var_name="operation")
    fig, ax = plt.subplots()
    sns.lineplot(df_plot, x="N", y="time [s]", hue="operation", ax=ax)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax 

def tightness_study(learner: Learner, fname_root="", plot=True, tightness="rank"):
    """ investigate tightness before and after reordering """
    if plot:
        fig_cost, ax_cost = plt.subplots()
        ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k")
        fig_cost.set_size_inches(5, 5)

        fig_eigs1, ax_eigs1 = plt.subplots()
        fig_eigs1.set_size_inches(5, 5)

        fig_eigs2, ax_eigs2 = plt.subplots()
        fig_eigs2.set_size_inches(5, 5)
    else:
        ax_cost = ax_eigs1 = ax_eigs2 = None
    idx_subset_original = learner.generate_minimal_subset(reorder=False, ax_cost=ax_cost, ax_eigs=ax_eigs1, tightness=tightness)
    idx_subset_reorder = learner.generate_minimal_subset(reorder=True, ax_cost=ax_cost, ax_eigs=ax_eigs2, tightness=tightness)

    if plot:
        ax_cost.legend(["QCQP cost", "dual cost, original ordering", "dual cost, new ordering"], loc="lower right")
        ax_eigs1.legend(loc="upper right", title="number of added\n constraints")
        ax_eigs1.set_title("original order")
        ax_eigs2.legend(loc="upper right", title="number of added\n constraints")
        ax_eigs2.set_title("sorted by dual values")

    if plot and fname_root != "":
        savefig(fig_cost, fname_root + "_tightness-cost.png")
        savefig(fig_eigs1, fname_root + "_tightness-eigs-original.png")
        savefig(fig_eigs2, fname_root + "_tightness-eigs-sorted.png")
    return idx_subset_original, idx_subset_reorder

def run_oneshot_experiment(learner, fname_root, plots, tightness="rank"):
    learner.run(verbose=True, use_known=False, plot=True)

    if "svd" in plots:
        fig = plt.gcf()
        ax = plt.gca()
        ax.legend(loc="lower left")
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_svd.png")

    idx_subset_original, idx_subset_reorder = tightness_study(learner, plot="tightness" in plots, fname_root=fname_root, tightness=tightness)

    if "matrices" in plots:
        A_matrices = [learner.A_matrices[i] for i in idx_subset_original]
        fig, ax = learner.save_matrices_poly(A_matrices=A_matrices[:5])
        w, h = fig.get_size_inches()
        fig.set_size_inches(5*w/h, 5)
        savefig(fig, fname_root + "_matrices.png")

        fig, ax = learner.save_matrices_sparsity(learner.A_matrices)
        savefig(fig, fname_root + "_matrices-sparsity.png")

    if "patterns" in plots:
        patterns_poly = learner.generate_patterns_poly(factor_out_parameters=True)
        add_columns = {
            "required": idx_subset_original,
            "required (reordered)": idx_subset_reorder,
        }
        df = learner.get_sorted_df(patterns_poly=patterns_poly, add_columns=add_columns)
        #df.sort_values(by="required", axis=0, inplace=True)
        title = f"substitution level: {learner.lifter.LEVEL_NAMES[learner.lifter.level]}"
        fig, ax = learner.save_sorted_patterns(df, title=title, drop_zero=True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(5*w/h, 5)
        savefig(fig, fname_root + "_patterns.png")

def range_only_tightness():
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    n_landmarks = 10
    d = 3
    seed = 0
    plots = [] #["svd", "matrices"]

    for level in ["no", "quad"]:
        n_positions = 2 if level == "quad" else 4
        variable_list = [["l"] + [f"x_{i}" for i in range(n_positions)] + [f"z_{i}" for i in range(n_positions)]]
        np.random.seed(seed)
        lifter = RangeOnlyLocLifter(n_positions=n_positions, n_landmarks=n_landmarks, d=d, level=level, W=None, variable_list=variable_list)
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, apply_patterns=False)
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(learner, fname_root, plots)

def range_only_scalability():
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
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
        df = pd.DataFrame(df_data)

        fig, ax = plot_scalability(df)
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_scalability.png")

def stereo_tightness():
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    n_landmarks = 4
    d = 2
    seed = 0
    #plots = ["tightness", "svd", "matrices", "patterns"]
    plots = ["matrices", "patterns"]
    levels = ["no", "urT"]

    #parameter_levels = ["ppT"] #["no", "p", "ppT"]
    param_level = "no"
    for level in levels: 
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        if d == 2:
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks, level=level, param_level=param_level, variable_list=variable_list
            )
        elif d == 3:
            lifter = Stereo3DLifter(
                n_landmarks=n_landmarks, level=level, param_level=param_level, variable_list=variable_list
            )

        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, apply_patterns=False)
        fname_root = f"_results/{lifter}_seed{seed}"

        run_oneshot_experiment(learner, fname_root, plots, tightness="cost")


def stereo_scalability():
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
    """
    n_landmarks_list = [4, 5, 6] 
    #n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    level = "urT" 
    param_level = "ppT"

    d = 2
    n_seeds = 3
    variable_list = None
    df_data = []
    for seed, n_landmarks in itertools.product(range(n_seeds), n_landmarks_list):
        print(f"===== {n_landmarks} ====")
        np.random.seed(seed)
        if d == 2:
            lifter = Stereo2DLifter(n_landmarks=n_landmarks, level=level, param_level=param_level, variable_list=variable_list)
        elif d == 3:
            lifter = Stereo3DLifter(n_landmarks=n_landmarks, level=level, param_level=param_level, variable_list=variable_list)
        fname_root = f"_results/{lifter}"
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

        times = learner.run(verbose=True, use_known=False, plot=False)
        for t_dict in times:
            t_dict["N"] = n_landmarks
            df_data.append(t_dict)

    import pandas as pd
    df = pd.DataFrame(df_data)

    fig, ax = plot_scalability(df)
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability.png")

if __name__ == "__main__":
    import warnings

    #with warnings.catch_warnings():
    #    warnings.simplefilter("error")
    #stereo_tightness()
    #range_only_tightness()
    #range_only_scalability()
    stereo_scalability()