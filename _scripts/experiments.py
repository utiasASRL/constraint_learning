import time
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from utils.plotting_tools import savefig

plot_dict = {
    "t ": {"value_name": "time [s]", "var_name": "operation", "start": "t "},
    "n ": {"value_name": "number", "var_name": "number of elements", "start": "n "},
    "N ": {"value_name": "number", "var_name": "number of elements", "start": "N "},
}
rename_dict = {
    "t learn templates": "learn templates",
    "t determine required": "determine subset",
    "t apply templates": "apply templates",
    "t check tightness": "solve SDP",
}


def plot_scalability_new(df, log=True, start="t "):
    import seaborn as sns

    dict_ = plot_dict[start]

    df_plot = df.melt(
        id_vars=["variables"],
        value_vars=[v for v in df.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=dict_["var_name"],
    )
    fig, ax = plt.subplots()
    df_plot.variables = df_plot.variables.astype("str")
    if len(df_plot.variables.unique()) == 1:
        for i, row in df_plot.iterrows():
            ax.scatter([0], row["time [s]"], label=row.operation)
        ax.set_xticks([0], [row.variables])
        ax.set_xlabel("variables")
        ax.set_ylabel("time [s]")
    else:
        sns.lineplot(
            df_plot, x="variables", y=dict_["value_name"], hue=dict_["var_name"], ax=ax
        )
    if log:
        ax.set_yscale("log")
    ax.legend(loc="upper left")
    return fig, ax


def plot_scalability(df, log=True, start="t ", ymin=None, ymax=None):
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    dict_ = plot_dict[start]

    df_plot = df.dropna(axis=1, inplace=False, how="all")
    df_plot = df_plot.melt(
        id_vars=["N"],
        value_vars=[v for v in df_plot.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=dict_["var_name"],
    )
    df_plot.replace(rename_dict, inplace=True)
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
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(df_plot.N.unique())
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right")
    return fig, ax


def save_table(df, tex_name):
    df_tab = df.set_index("N", drop=True)
    df_tab.drop(
        columns=[
            c
            for c in df_tab.columns
            if c.startswith("t ") or c.startswith("variables") or c.startswith("order")
        ],
        inplace=True,
    )
    df_tab.style.to_latex(tex_name)
    print(f"saved table as {tex_name}")


def save_tightness_order(learner: Learner, fname_root=""):
    from matplotlib.ticker import MaxNLocator

    if (learner.df_tight is None) or (len(learner.df_tight) == 0):
        print(f"no tightness data for {learner.lifter}")
        return
    df_current = learner.df_tight[learner.df_tight.lifter == str(learner.lifter)]

    fig_cost, ax_cost = plt.subplots()
    ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k", label=".QCQP cost")
    for reorder, df in df_current.groupby("reorder"):
        label = (
            "dual cost, sorted by dual values"
            if reorder
            else "dual cost, original ordering"
        )
        ax_cost.semilogy(df["n"], df["dual cost"], label=label)

        fig_eigs, ax_eigs = plt.subplots()
        fig_eigs.set_size_inches(5, 5)

        cmap = plt.get_cmap("viridis", len(df))

        cost_tight = np.where(df.cost_tight.values == True)[0]
        cost_idx = cost_tight[0] if len(cost_tight) else None
        rank_tight = np.where(df.rank_tight.values == True)[0]
        rank_idx = rank_tight[0] if len(rank_tight) else None

        for i in range(len(df)):
            n = df.iloc[i].n
            eig = df.iloc[i].eigs
            label = None
            color = cmap(i)
            if i == len(df) // 2:
                label = "..."
            if i == 0:
                label = f"{n}"
            if i == len(df) - 1:
                label = f"{n}"
            if i == cost_idx:
                label = f"{n}: cost-tight"
                color = "red"
            if i == rank_idx:
                label = f"{n}: rank-tight"
                color = "black"
            ax_eigs.semilogy(eig, color=color, label=label)

        # make sure these two are in foreground
        if cost_idx is not None:
            ax_eigs.semilogy(df.iloc[cost_idx].eigs, color="red")
        if rank_idx is not None:
            ax_eigs.semilogy(df.iloc[rank_idx].eigs, color="black")
        ax_eigs.set_xlabel("index")
        ax_eigs.set_ylabel("eigenvalue")
        ax_eigs.grid(True)

        if reorder:
            name = "sorted"
            ax_eigs.set_title("sorted by dual values")
        else:
            name = "original"
            ax_eigs.set_title("original order")
        ax_eigs.legend(loc="upper right", title="number of added\n constraints")
        if fname_root != "":
            savefig(fig_eigs, fname_root + f"_tightness-eigs-{name}.pdf")

    ax_cost.legend(
        loc="lower right",
    )
    ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cost.set_xlabel("number of added constraints")
    ax_cost.set_ylabel("cost")
    fig_cost.set_size_inches(5, 5)

    ax_cost.grid(True)

    if fname_root != "":
        savefig(fig_cost, fname_root + "_tightness-cost.pdf")
    return


def tightness_study(learner: Learner, tightness="rank", original=False):
    """investigate tightness before and after reordering"""
    print("reordering...")
    idx_subset_reorder = learner.generate_minimal_subset(
        reorder=True, tightness=tightness
    )
    if not original:
        return None, idx_subset_reorder
    print("original ordering...")
    idx_subset_original = learner.generate_minimal_subset(
        reorder=False, tightness=tightness, start=250
    )
    return idx_subset_original, idx_subset_reorder


def run_scalability_plot(learner: Learner):
    fname_root = f"_results/scalability_{learner.lifter}"
    data = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
    df = pd.DataFrame(data)
    fig, ax = plot_scalability_new(df, start="t ")
    savefig(fig, fname_root + f"_small.pdf")

    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, tightness="cost", original=False
    )
    templates_poly = learner.generate_templates_poly(factor_out_parameters=False)
    add_columns = {
        "required (reordered)": idx_subset_reorder,
    }
    df = learner.get_sorted_df(templates_poly=templates_poly, add_columns=add_columns)
    title = (
        ""  # f"substitution level: {learner.lifter.LEVEL_NAMES[learner.lifter.level]}"
    )
    fig, ax = learner.save_sorted_templates(
        df, title=title, drop_zero=True, simplify=True
    )
    w, h = fig.get_size_inches()
    fig.set_size_inches(10, 10 * h / w)
    savefig(fig, fname_root + f"_templates.pdf")


def run_scalability_new(
    learner: Learner,
    param_list: list,
    n_seeds: int = 1,
    vmin=None,
    vmax=None,
    recompute=True,
    tightness="cost",
):
    import pickle

    fname = f"{learner.lifter}.pkl"
    fname_root = f"_results/scalability_{learner.lifter}"

    try:
        assert not recompute, "forcing to recompute"
        with open(fname, "rb") as f:
            learner = pickle.load(f)
            data_dict = pickle.load(f)
    except (AssertionError, FileNotFoundError) as e:
        print(e)
        # find which of the constraints are actually necessary
        orig_dict = {}
        t1 = time.time()
        data = learner.run(
            verbose=True, use_known=False, plot=False, tightness=tightness
        )
        orig_dict["t learn templates"] = time.time() - t1

        df = pd.DataFrame(data)
        fig, ax = plot_scalability_new(df, start="t ")
        savefig(fig, fname_root + f"_small.pdf")

        with open(fname, "wb") as f:
            pickle.dump(learner, f)
            pickle.dump(orig_dict, f)
        print("wrote intermediate as", fname)

    fname = fname_root + "_df_all.pkl"
    try:
        assert not recompute, "forcing to recompute"
        df = pd.read_pickle(fname)
    except (AssertionError, FileNotFoundError) as e:
        print(e)
        order_dict = {}

        t1 = time.time()
        idx_subset_original, idx_subset_reorder = tightness_study(
            learner,
            tightness=tightness,
            original=True,
        )
        orig_dict["t determine required"] = time.time() - t1
        orig_dict["n templates"] = len(learner.constraints)
        orig_dict["n sufficient (sorted)"] = (
            len(idx_subset_reorder) if idx_subset_original is not None else None
        )
        orig_dict["n sufficient (original)"] = (
            len(idx_subset_original) if idx_subset_original is not None else None
        )
        if idx_subset_reorder is not None:
            order_dict["sorted"] = idx_subset_reorder
        # if idx_subset_original is not None:
        #    order_dict["original"] = idx_subset_original
        order_dict["all"] = range(len(learner.constraints))

        max_seeds = n_seeds + 5
        df_data = []
        for name, new_order in order_dict.items():
            data_dict = deepcopy(orig_dict)
            for n_params in param_list:
                n_successful_seeds = 0
                for seed in range(max_seeds):
                    print(
                        f"================== N={n_params},seed={seed} ======================"
                    )
                    data_dict["N"] = n_params
                    data_dict["order"] = name

                    np.random.seed(seed)
                    # TODO(FD): replace below with copy constructor
                    if isinstance(learner.lifter, Stereo2DLifter):
                        new_lifter = Stereo2DLifter(
                            n_landmarks=n_params,
                            level=learner.lifter.level,
                            param_level=learner.lifter.param_level,
                            variable_list=None,
                        )
                    elif isinstance(learner.lifter, Stereo3DLifter):
                        new_lifter = Stereo3DLifter(
                            n_landmarks=n_params,
                            level=learner.lifter.level,
                            param_level=learner.lifter.param_level,
                            variable_list=None,
                        )
                    elif isinstance(learner.lifter, RangeOnlyLocLifter):
                        new_lifter = RangeOnlyLocLifter(
                            n_positions=n_params,
                            n_landmarks=learner.lifter.n_landmarks,
                            level=learner.lifter.level,
                            d=learner.lifter.d,
                            variable_list=None,
                        )
                    else:
                        raise ValueError(learner.lifter)

                    # doesn't matter because we don't use the usual pipeline.
                    # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
                    new_learner = Learner(
                        lifter=new_lifter, variable_list=new_lifter.variable_list
                    )

                    success = new_learner.find_local_solution()
                    if not success:
                        continue
                    n_successful_seeds += 1

                    # apply the templates to all new landmarks
                    t1 = time.time()
                    # (make sure the dimensions of the constraints are correct)

                    new_learner.templates = [
                        learner.constraints[i].scale_to_new_lifter(new_lifter)
                        for i in new_order
                    ]
                    # apply the templates
                    n_new, n_total = new_learner.apply_templates(reapply_all=True)
                    data_dict[f"n total ({name})"] = n_total
                    data_dict["t apply templates"] = time.time() - t1

                    # TODO(FD) below should not be necessary
                    new_learner.constraints = new_learner.clean_constraints(
                        new_learner.constraints, [], remove_imprecise=True
                    )

                    # determine tightness
                    print(f"=========== tightness test: {name} ===============")
                    t1 = time.time()
                    new_learner.is_tight(verbose=True, tightness=tightness)
                    data_dict["t check tightness"] = time.time() - t1
                    # times = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
                    df_data.append(deepcopy(data_dict))

                    if n_successful_seeds >= n_seeds:
                        break
            df = pd.DataFrame(df_data)
            df.to_pickle(fname)

    df = df.apply(pd.to_numeric, errors="ignore")
    vmax = df[[c for c in df.columns if c.startswith("t ")]].max().max() * 1.1
    for name, df_plot in df.groupby("order"):
        fig, ax = plot_scalability(df_plot, log=True, ymin=vmin, ymax=vmax)
        ax.set_xlabel("number of landmarks")
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + f"_{name}.pdf")

        tex_name = fname_root + f"_{name}_n.tex"
        save_table(df_plot, tex_name)

    fig, ax = plot_scalability(df, log=True, start="n ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + f"_n.pdf")


def run_oneshot_experiment(
    learner: Learner, fname_root, plots, tightness="rank", add_original=True
):
    learner.run(verbose=True, use_known=True, plot=True, tightness=tightness)

    if "svd" in plots:
        fig = plt.gcf()
        ax = plt.gca()
        ax.legend(loc="lower left")
        fig.set_size_inches(3, 3)
        savefig(fig, fname_root + "_svd.pdf")

    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, tightness=tightness, original=add_original
    )
    if "tightness" in plots:
        save_tightness_order(learner, fname_root)

    if "matrices" in plots:
        A_matrices = [
            c.A_poly_ for c in learner.constraints if "x:0" in c.A_poly_.adjacency_i
        ]
        save_individual = False
        if "matrix" in plots:
            save_individual = True
        fig, ax = learner.save_matrices_poly(
            A_matrices=A_matrices,
            n_matrices=5,
            save_individual=save_individual,
            fname_root=fname_root,
        )
        w, h = fig.get_size_inches()
        fig.set_size_inches(5 * w / h, 5)
        savefig(fig, fname_root + "_matrices.pdf")

        if idx_subset_reorder is not None:
            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_reorder]
            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_reorder]
            fig, ax = learner.save_matrices_sparsity(A_matrices)
            savefig(fig, fname_root + "_matrices-sparsity-reorder.pdf")

        if idx_subset_original is not None:
            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_original]
            fig, ax = learner.save_matrices_sparsity(A_matrices)
            savefig(fig, fname_root + "_matrices-sparsity-original.pdf")

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
        fig, ax = learner.save_sorted_templates(
            df, title=title, drop_zero=True, simplify=True
        )
        w, h = fig.get_size_inches()
        fig.set_size_inches(5, 5 * h / w)
        savefig(fig, fname_root + "_templates.pdf")
