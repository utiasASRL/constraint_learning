import itertools
import time
from copy import deepcopy

import pandas as pd
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

def save_tightness_order(learner: Learner, fname_root=""):
    from matplotlib.ticker import MaxNLocator

    if learner.df_tight is None:
        print(f"no tightness data for {learner.lifter}")
        return
    df_current = learner.df_tight[learner.df_tight.lifter == str(learner.lifter)]

    fig_cost, ax_cost = plt.subplots()
    ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k", label="QCQP cost")
    for reorder, df in df_current.groupby("reorder"):
        label = "dual cost, sorted by dual values" if reorder else "dual cost, original ordering"
        ax_cost.semilogy(range(len(df)), df["dual cost"], label=label)

        fig_eigs, ax_eigs = plt.subplots()
        fig_eigs.set_size_inches(5, 5)

        cmap = plt.get_cmap("viridis", len(df))

        cost_tight = np.where(df.cost_tight.values == True)[0]
        cost_idx = cost_tight[0] if len(cost_tight) else None
        rank_tight = np.where(df.rank_tight.values == True)[0]
        rank_idx = rank_tight[0] if len(rank_tight) else None

        for i in range(len(df)):
            eig = df.iloc[i].eigs
            label = None
            color = cmap(i)
            if i == len(df) // 2:
                label = "..."
            if i == 0:
                label = f"{i+1}"
            if i == len(df) - 1:
                label = f"{i+1}"
            if i == cost_idx:
                label = f"{i+1}: cost-tight"
                color = "red"
            if i == rank_idx:
                label = f"{i+1}: rank-tight"
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
            savefig(fig_eigs, fname_root + f"_tightness-eigs-{name}.png")

    ax_cost.legend(
        loc="lower right",
    )
    ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cost.set_xlabel("number of added constraints")
    ax_cost.set_ylabel("cost")
    fig_cost.set_size_inches(5, 5)


    ax_cost.grid(True)

    if fname_root != "":
        savefig(fig_cost, fname_root + "_tightness-cost.png")
    return


def tightness_study(learner: Learner, tightness="rank", original=False):
    """investigate tightness before and after reordering"""
    print("reordering...")
    idx_subset_reorder = learner.generate_minimal_subset(reorder=True, tightness=tightness)
    if not original:
        return None, idx_subset_reorder
    print("original ordering...")
    idx_subset_original = learner.generate_minimal_subset(reorder=False, tightness=tightness)
    return idx_subset_original, idx_subset_reorder


def run_scalability_new(learner, param_list, n_seeds=1):
    # find which of the constraints are actually necessary
    t1 = time.time()
    data = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
    data_dict = data[-1]
    data_dict["t total learn templates"] = time.time() - t1

    t1 = time.time()
    # just use cost tightness because rank tightness was not achieved even in toy example
    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, tightness="cost"
    )
    data_dict["n req1"] = len(idx_subset_original) if idx_subset_original is not None else None
    data_dict["n req2"] = len(idx_subset_reorder) if idx_subset_original is not None else None
    data_dict["t determine required"] = time.time() - t1

    new_order = idx_subset_reorder
    name = "reordered"

    # new_order = idx_subset_original
    # name = "original"

    df_data = []
    for seed, n_params in itertools.product(range(n_seeds), param_list):
        print(f"================== N={n_params},seed={seed} ======================")

        data_dict["N"] = n_params 
        np.random.seed(seed)

        # TODO(FD): replace below with copy constructor
        if isinstance(learner.lifter, Stereo2DLifter):
            new_lifter = Stereo2DLifter(n_landmarks=n_params, level=learner.lifter.level, param_level=learner.lifter.param_level, variable_list=None)
        elif isinstance(learner.lifter, Stereo3DLifter):
            new_lifter = Stereo3DLifter(n_landmarks=n_params, level=learner.lifter.level, param_level=learner.lifter.param_level, variable_list=None)
        elif isinstance(learner.lifter, RangeOnlyLocLifter):
            new_lifter = RangeOnlyLocLifter(n_landmarks=n_params, level=learner.lifter.level, param_level=learner.lifter.param_level, d=learner.lifter.d, variable_list=None)
        else:
            raise ValueError(learner.lifter)
            
        # doesn't matter because we don't use the usual pipeline.
        # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        new_learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)

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


def run_oneshot_experiment(learner:Learner, fname_root, plots, tightness="rank", add_original=True):
    learner.run(verbose=True, use_known=False, plot=True, tightness=tightness)

    if "svd" in plots:
        fig = plt.gcf()
        ax = plt.gca()
        ax.legend(loc="lower left")
        fig.set_size_inches(5, 5)
        savefig(fig, fname_root + "_svd.png")

    idx_subset_original, idx_subset_reorder = tightness_study(learner, tightness=tightness, original=add_original)
    if "tightness" in plots:
        save_tightness_order(learner, fname_root)

    if "matrices" in plots:
        if idx_subset_reorder is not None:
            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_reorder]
            fig, ax = learner.save_matrices_poly(A_matrices=A_matrices[:5])
            w, h = fig.get_size_inches()
            fig.set_size_inches(5 * w / h, 5)
            savefig(fig, fname_root + "_matrices.png")

            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_reorder]
            fig, ax = learner.save_matrices_sparsity(A_matrices)
            savefig(fig, fname_root + "_matrices-sparsity-reorder.png")

        if idx_subset_original is not None:
            A_matrices = [learner.constraints[i].A_poly_ for i in idx_subset_original]
            fig, ax = learner.save_matrices_sparsity(A_matrices)
            savefig(fig, fname_root + "_matrices-sparsity-original.png")

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
        fig, ax = learner.save_sorted_templates(df, title=title, drop_zero=True, simplify=True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(5 * w / h, 5)
        savefig(fig, fname_root + "_templates.png")
