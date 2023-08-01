import time
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
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

def create_newinstance(lifter, n_params):
    # TODO(FD): replace below with copy constructor
    if type(lifter) == Stereo2DLifter:
        new_lifter = Stereo2DLifter(
            n_landmarks=n_params,
            level=lifter.level,
            param_level=lifter.param_level,
            variable_list=None,
        )
    elif type(lifter) ==  Stereo3DLifter:
        new_lifter = Stereo3DLifter(
            n_landmarks=n_params,
            level=lifter.level,
            param_level=lifter.param_level,
            variable_list=None,
        )
    elif type(lifter) ==  RangeOnlyLocLifter:
        new_lifter = RangeOnlyLocLifter(
            n_positions=n_params,
            n_landmarks=lifter.n_landmarks,
            level=lifter.level,
            d=lifter.d,
            variable_list=None,
        )
    elif type(lifter) == MonoLifter:
        new_lifter = MonoLifter(
            n_landmarks=n_params,
            robust=lifter.robust,
            level=lifter.level,
            d=lifter.d,
            variable_list=None,
        )
    elif type(lifter) ==  WahbaLifter:
        new_lifter = WahbaLifter(
            n_landmarks=n_params,
            robust=lifter.robust,
            level=lifter.level,
            d=lifter.d,
            variable_list=None,
        )
    else:
        raise ValueError(lifter)
    return new_lifter


def plot_scalability_new(df, log=True, start="t "):
    import seaborn as sns

    dict_ = plot_dict[start]

    var_name=dict_["var_name"]
    df_plot = df.melt(
        id_vars=["variables"],
        value_vars=[v for v in df.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=var_name,
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
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig, ax


def plot_scalability(df, log=True, start="t ", ymin=None, ymax=None):
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    dict_ = plot_dict[start]
    var_name=dict_["var_name"]

    df_plot = df.dropna(axis=1, inplace=False, how="all")
    df_plot = df_plot.melt(
        id_vars=["N", "type"],
        value_vars=[v for v in df_plot.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=var_name,
    )
    df_plot.replace(rename_dict, inplace=True)
    var_name_list = df_plot[var_name].unique()
    fig, axs = plt.subplots(1, len(var_name_list), sharex=True, sharey=True)
    axs = {op:ax for op, ax in zip(var_name_list, axs)}
    for var_name, df_sub in df_plot.groupby(var_name):
        last = var_name == var_name_list[-1]
        sns.lineplot(
            df_sub, x="N", y=dict_["value_name"], style="type", ax=axs[var_name], legend=last
        )
        if log:
            axs[var_name].set_yscale("log")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[var_name].set_xticks(df_plot.N.unique())
        axs[var_name].set_title(var_name)
        #axs[operation].legend(loc="lower right")
    #axs[var_name].legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
    return fig, axs


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
        ax_cost.semilogy(df.index, df["dual cost"], label=label)

        fig_eigs, ax_eigs = plt.subplots()
        fig_eigs.set_size_inches(5, 5)

        cmap = plt.get_cmap("viridis", len(df))

        cost_tight = np.where(df.cost_tight.values == True)[0]
        cost_idx = cost_tight[0] if len(cost_tight) else None
        rank_tight = np.where(df.rank_tight.values == True)[0]
        rank_idx = rank_tight[0] if len(rank_tight) else None

        for i in range(len(df)):
            n = df.iloc[i].name
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


def tightness_study(learner: Learner, tightness="rank", original=False, use_last=None, use_bisection=False):
    """investigate tightness before and after reordering"""
    print("reordering...")
    idx_subset_reorder = learner.generate_minimal_subset(
        reorder=True, tightness=tightness, use_last=use_last, use_bisection=use_bisection
    )
    if not original:
        return None, idx_subset_reorder
    print("original ordering...")
    idx_subset_original = learner.generate_minimal_subset(
        reorder=False, tightness=tightness, use_last=use_last, use_bisection=use_bisection
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
    recompute=False,
    tightness="cost",
    use_last=None,
    use_bisection=False

):
    import pickle

    fname = f"_results/{learner.lifter}.pkl"
    fname_root = f"_results/scalability_{learner.lifter}"

    try:
        assert not recompute, "forcing to recompute"
        with open(fname, "rb") as f:
            learner = pickle.load(f)
            orig_dict = pickle.load(f)
        print(f"--------- read {fname} \n")
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

    fname = fname_root + "_order_dict.pkl"
    try:
        assert not recompute, "forcing to recompute"
        with open(fname, "rb") as f:
            order_dict = pickle.load(f)
        print(f"--------- read {fname} \n")
    except (AssertionError, FileNotFoundError, EOFError) as e:
        print(e)
        order_dict = {}

        t1 = time.time()
        idx_subset_original, idx_subset_reorder = tightness_study(
            learner,
            tightness=tightness,
            original=True,
            use_last=use_last,
            use_bisection=use_bisection
        )
        orig_dict["t determine required"] = time.time() - t1
        if idx_subset_reorder is not None:
            order_dict["sorted"] = idx_subset_reorder
        if idx_subset_original is not None:
            order_dict["original"] = idx_subset_original
        #order_dict["all"] = range(len(learner.constraints))
        order_dict["basic"] = None
        with open(fname, "wb") as f:
            pickle.dump(order_dict, f)

    fname = fname_root + "_df_all.pkl"
    try:
        assert not recompute, "forcing to recompute"
        df = pd.read_pickle(fname)
        assert set(param_list).issubset(df.N.unique())
        print(f"--------- read {fname} \n")
    except (AssertionError, FileNotFoundError):
        max_seeds = n_seeds + 5
        df_data = []
        for name, new_order in order_dict.items():
            data_dict = {}
            for n_params in param_list:
                n_successful_seeds = 0
                for seed in range(max_seeds):
                    print(
                        f"================== apply ({name}) to N={n_params},seed={seed} ======================"
                    )
                    data_dict["N"] = n_params
                    data_dict["seed"] = seed
                    data_dict["type"] = name

                    new_lifter = create_newinstance(learner.lifter, n_params)
                    # doesn't matter because we don't use the usual pipeline.
                    # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
                    np.random.seed(seed)
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
                    if new_order is not None:
                        new_learner.templates = [
                            learner.constraints[i].scale_to_new_lifter(new_lifter)
                            for i in new_order
                        ]
                    else:
                        new_learner.templates = learner.templates
                    # apply the templates
                    data_dict[f"n templates"] = len(new_learner.templates)
                    n_new, n_total = new_learner.apply_templates(reapply_all=True)
                    data_dict[f"n constraints"] = n_total
                    data_dict[f"t create constraints"] = time.time() - t1

                    # TODO(FD) below should not be necessary
                    new_learner.constraints = new_learner.clean_constraints(
                        new_learner.constraints, [], remove_imprecise=True
                    )
                    # determine tightness
                    if isinstance(new_lifter, Stereo3DLifter) and (n_params > 25) and (name == "basic"):
                        print(f"skipping tightness test of stereo3D with {n_params} because it leeds to memory error")
                        continue
                    print(f"=========== tightness test: {name} ===============")
                    t1 = time.time()
                    new_learner.is_tight(verbose=True, tightness=tightness)
                    data_dict[f"t solve SDP"] = time.time() - t1
                    # times = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
                    df_data.append(deepcopy(data_dict))

                    if n_successful_seeds >= n_seeds:
                        break
            df = pd.DataFrame(df_data)
            df.to_pickle(fname)

    fname = f"{fname_root}_df_oneshot.pkl"
    try:
        assert recompute is False
        df_oneshot = pd.read_pickle(fname)
        print(f"--------- read {fname} \n")
    except (FileNotFoundError, AssertionError):
        max_seeds = n_seeds + 5
        df_data = []
        for n_params in param_list:
            if isinstance(learner.lifter, Stereo2DLifter) and n_params > 25:
                print(f"skipping N={n_params} for stereo2D because this will cause out of memory error.")
                continue
            if isinstance(learner.lifter, Stereo3DLifter) and n_params > 10:
                print(f"skipping N={n_params} for stereo3D because this will cause out of memory error.")
                continue
            n_successful_seeds = 0
            for seed in range(max_seeds):
                print(
                    f"================== oneshot N={n_params},seed={seed} ======================"
                )
                data_dict = {
                    "N": n_params,
                    "seed": seed,
                    "type": "from scratch"
                }
                new_lifter = create_newinstance(learner.lifter, n_params)

                ############ standard one-shot ############
                new_lifter.param_level = "no"
                variable_list = new_lifter.get_all_variables()
                new_learner = Learner(
                    lifter=new_lifter, variable_list=variable_list, apply_templates=False
                )

                success = new_learner.find_local_solution()
                if not success:
                    continue
                n_successful_seeds += 1

                new_dict = new_learner.run(tightness=tightness, verbose=True)[-1]
                data_dict["t create constraints"] = new_dict["t learn templates"] 
                data_dict["t solve SDP"] = new_dict["t check tightness"]
                data_dict["n templates"] = new_dict["n templates"]
                data_dict["n constraints"] = new_dict["n templates"]

                ############ incremental one-shot ############
                #new_lifter.param_level = "ppT"
                #new_learner = Learner(
                #    lifter=new_lifter, variable_list=new_lifter.VARIABLE_LIST, apply_templates=True
                #)
                #t1 = time.time()
                #new_learner.run(tightness=tightness, verbose=True)
                #data_dict["t learn from scratch (incremental)"] = t1 - time.time()
                df_data.append(deepcopy(data_dict))
                if n_successful_seeds >= n_seeds:
                    break

            df_oneshot = pd.DataFrame(df_data)
            df_oneshot.to_pickle(fname)

    df = pd.concat([df, df_oneshot], axis=0)
    df = df.apply(pd.to_numeric, errors="ignore")

    fig, ax = plot_scalability(df, log=True, start="t ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + f"_t.pdf")
    fig, ax = plot_scalability(df, log=True, start="n ")
    fig.set_size_inches(5, 5)
    savefig(fig, fname_root + f"_n.pdf")

    tex_name = fname_root + f"_n.tex"
    save_table(df, tex_name)

    
def run_oneshot_experiment(
    learner: Learner, fname_root, plots, tightness="rank", add_original=True, use_last=None, use_bisection=False
):
    learner.run(verbose=True, use_known=True, plot=True, tightness=tightness)

    if "svd" in plots:
        fig = plt.gcf()
        ax = plt.gca()
        ax.legend(loc="lower left")
        fig.set_size_inches(3, 3)
        savefig(fig, fname_root + "_svd.pdf")

    idx_subset_original, idx_subset_reorder = tightness_study(
        learner, tightness=tightness, original=add_original, use_last=use_last, use_bisection=use_bisection
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
            "required (sorted)": idx_subset_reorder,
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
