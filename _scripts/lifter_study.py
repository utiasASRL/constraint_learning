import itertools
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# import matplotlib
# matplotlib.use('Agg')
plt.ioff()

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

n_landmarks_list = [5, 10, 15]
n_seeds = 1

plot_dict = {
    "t ": {"value_name": "time [s]", "var_name": "operation", "start": "t "},
    "n ": {"value_name": "number", "var_name": "number of elements", "start": "n "},
}


def plot_scalability(df, log=True, start="t "):
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    dict_ = plot_dict[start]

    df_plot = df.melt(
        id_vars=["N"],
        value_vars=[v for v in df.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=dict_["var_name"],
    )
    fig, ax = plt.subplots()
    if len(df_plot.N.unique()) == 1:
        sns.lineplot(df_plot, x=dict_["var_name"], y=dict_["value_name"], ax=ax)
        if log:
            ax.set_yscale("log")
    else:
        sns.lineplot(
            df_plot, x="N", y=dict_["value_name"], hue=dict_["var_name"], ax=ax
        )
        if log:
            ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def save_table(df, tex_name):
    df_tab = df.set_index("N", drop=True)
    df_tab.drop(
        columns=[c for c in df_tab.columns if c.startswith("t ")] + ["variables"],
        inplace=True,
    )
    df_tab.style.to_latex(tex_name)
    print(f"saved table as {tex_name}")


def tightness_study(learner: Learner, fname_root="", plot=True, tightness="rank"):
    """investigate tightness before and after reordering"""
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
    idx_subset_original = learner.generate_minimal_subset(
        reorder=False, ax_cost=ax_cost, ax_eigs=ax_eigs1, tightness=tightness
    )

    # fig, ax = plt.subplots()
    idx_subset_reorder = learner.generate_minimal_subset(
        reorder=True,
        ax_cost=ax_cost,
        ax_eigs=ax_eigs2,
        # ax_lambda=ax,
        tightness=tightness,
    )
    # ax.axvline(len(idx_subset_reorder))

    if plot:
        ax_cost.legend(
            ["QCQP cost", "dual cost, original ordering", "dual cost, new ordering"],
            loc="lower right",
        )
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

    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, plot="tightness" in plots, fname_root=fname_root, tightness=tightness
    )

    if "matrices" in plots:
        A_matrices = [learner.constraints[i].A_poly() for i in idx_subset_original]
        fig, ax = learner.save_matrices_poly(A_matrices=A_matrices[:5])
        w, h = fig.get_size_inches()
        fig.set_size_inches(5 * w / h, 5)
        savefig(fig, fname_root + "_matrices.png")

        fig, ax = learner.save_matrices_sparsity(
            [c.A_poly() for c in learner.constraints]
        )
        savefig(fig, fname_root + "_matrices-sparsity.png")

    if "templates" in plots:
        templates_poly = learner.generate_templates_poly(factor_out_parameters=True)
        add_columns = {
            "required": idx_subset_original,
            "required (reordered)": idx_subset_reorder,
        }
        df = learner.get_sorted_df(
            templates_poly=templates_poly, add_columns=add_columns
        )
        # df.sort_values(by="required", axis=0, inplace=True)
        title = (
            f"substitution level: {learner.lifter.LEVEL_NAMES[learner.lifter.level]}"
        )
        fig, ax = learner.save_sorted_templates(df, title=title, drop_zero=True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(5 * w / h, 5)
        savefig(fig, fname_root + "_templates.png")


def range_only_tightness():
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    n_landmarks = 10
    d = 3
    seed = 0
    plots = []  # ["svd", "matrices"]

    for level in ["no", "quad"]:
        n_positions = 2 if level == "quad" else 4
        variable_list = [
            ["l"]
            + [f"x_{i}" for i in range(n_positions)]
            + [f"z_{i}" for i in range(n_positions)]
        ]
        np.random.seed(seed)
        lifter = RangeOnlyLocLifter(
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            W=None,
            variable_list=variable_list,
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(learner, fname_root, plots)


def range_only_scalability():
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
    """
    n_landmarks = 10
    n_positions_list = [3, 4, 5]
    # n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    # level = "no" # for range-only
    level = "no"  # for range-only
    d = 3

    for level in ["quad", "no"]:
        n_seeds = 10
        variable_list = None
        df_data = []
        for seed, n_positions in itertools.product(range(n_seeds), n_positions_list):
            print(f"===== {n_positions} ====")
            np.random.seed(seed)
            lifter = RangeOnlyLocLifter(
                n_positions=n_positions,
                n_landmarks=n_landmarks,
                d=d,
                level=level,
                W=None,
                variable_list=variable_list,
            )
            fname_root = f"_results/{lifter}"
            learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

            times = learner.run(verbose=True, use_known=False, plot=False)
            for t_dict in times:
                t_dict["N"] = n_positions
                df_data.append(t_dict)

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
    # plots = ["tightness", "svd", "matrices", "templates"]
    plots = ["matrices", "templates"]
    levels = ["no", "urT"]

    # parameter_levels = ["ppT"] #["no", "p", "ppT"]
    param_level = "no"
    for level in levels:
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        if d == 2:
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )
        elif d == 3:
            lifter = Stereo3DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )

        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"

        run_oneshot_experiment(learner, fname_root, plots, tightness="cost")


def stereo_scalability_new():
    import time

    d = 2
    level = "urT"
    param_level = "ppT"

    n_landmarks = 3

    variable_list = None  # use the default one.
    # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
    np.random.seed(0)
    if d == 2:
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
    elif d == 3:
        lifter = Stereo3DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

    # find which of the constraints are actually necessary
    t1 = time.time()
    data = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
    data_dict = data[-1]
    data_dict["t total learn templates"] = time.time() - t1

    t1 = time.time()
    # just use cost tightness because rank tightness was not achieved even in toy example
    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, plot=False, tightness="cost"
    )
    data_dict["n req1"] = len(idx_subset_original)
    data_dict["n req2"] = len(idx_subset_reorder)
    data_dict["t determine required"] = time.time() - t1

    new_order = idx_subset_reorder
    name = "reordered"

    # new_order = idx_subset_original
    # name = "original"

    df_data = []
    for seed, n_landmarks in itertools.product(range(n_seeds), n_landmarks_list):
        print(f"================== N={n_landmarks},seed={seed} ======================")
        data_dict["N"] = n_landmarks
        np.random.seed(seed)

        # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        variable_list = None
        new_lifter = Stereo2DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
        new_learner = Learner(lifter=new_lifter, variable_list=lifter.variable_list)

        # apply the templates to all new landmarks
        t1 = time.time()
        # (make sure the dimensions of the constraints are correct)
        [learner.constraints[i].scale_to_new_lifter(new_lifter) for i in new_order]
        new_learner.constraints = [learner.constraints[i] for i in new_order]
        new_learner.a_current_ = learner.get_a_row_list(new_learner.constraints)
        # apply the templates
        n_new, n_total = new_learner.apply_templates(reapply_all=True)
        data_dict["n total"] = n_total
        data_dict["t apply templates"] = time.time() - t1

        # TODO(FD) below should not be necessary
        new_learner.clean_constraints([], remove_imprecise=True)

        # determine tightness
        print(f"=========== tightness test: {name} ===============")
        t1 = time.time()
        print(new_learner.is_tight(verbose=True, tightness="cost"))
        data_dict["t final tightness"] = time.time() - t1
        # times = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
        df_data.append(deepcopy(data_dict))

    fname_root = f"_results/{new_lifter}"

    df = pd.DataFrame(df_data)
    fig, ax = plot_scalability(df, log=True)
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability_new.png")

    fig, ax = plot_scalability(df, log=True, start="n ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability_new_n.png")

    tex_name = fname_root + "_scalability_new.tex"
    save_table(df, tex_name)


def stereo_scalability():
    """
    Deteremine how the range-only problem sclaes with nubmer of positions.
    """
    # n_positions_list = np.logspace(0.1, 2, 10).astype(int)
    n_landmarks_list = [3, 4, 5]

    level = "urT"
    param_level = "no"
    # param_level = "ppT"

    d = 2
    # variable_list = None
    df_data = []
    for seed, n_landmarks in itertools.product(range(n_seeds), n_landmarks_list):
        print(f"================== N={n_landmarks},seed={seed} ======================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        if d == 2:
            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )
        elif d == 3:
            lifter = Stereo3DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )
        fname_root = f"_results/{lifter}"
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )

        # just use cost tightness because rank tightness was not achieved even in toy example
        times = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
        t_dict = times[0]

        idx_subset_original, idx_subset_reorder = tightness_study(
            learner, plot=False, tightness="cost"
        )
        t_dict["n req1"] = len(idx_subset_original)
        t_dict["n req2"] = len(idx_subset_reorder)
        t_dict["N"] = n_landmarks
        df_data.append(t_dict)

    df = pd.DataFrame(df_data)

    fig, ax = plot_scalability(df, log=True)
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability.png")

    fig, ax = plot_scalability(df, log=True, start="n ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + "_scalability_n.png")

    tex_name = fname_root + "_scalability.tex"
    save_table(df, tex_name)


if __name__ == "__main__":
    import warnings

    # with warnings.catch_warnings():
    #    warnings.simplefilter("error")
    # stereo_tightness()
    # range_only_tightness()
    # range_only_scalability()

    # import cProfile
    # cProfile.run('stereo_scalability()')

    stereo_scalability()
    stereo_scalability_new()

    # with open("temp.pkl", "rb") as f:
    #    learner = pickle.load(f)
    # learner.apply_templates()