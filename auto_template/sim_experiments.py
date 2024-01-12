import time
from copy import deepcopy
import pickle

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from auto_template.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.robust_pose_lifter import RobustPoseLifter
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
ylabels = {
    "t solve SDP": "solve SDP",
    "t create constraints": "create constraints",
    "zoom": "",
}

RESULTS_FOLDER = "_results"
EARLY_STOP = False


def create_newinstance(lifter, n_params, n_outliers=None):
    # TODO(FD): replace below with copy constructor
    if type(lifter) == Stereo2DLifter:
        new_lifter = Stereo2DLifter(
            n_landmarks=n_params,
            level=lifter.level,
            param_level=lifter.param_level,
            variable_list=None,
        )
    elif type(lifter) == Stereo3DLifter:
        new_lifter = Stereo3DLifter(
            n_landmarks=n_params,
            level=lifter.level,
            param_level=lifter.param_level,
            variable_list=None,
        )
    elif type(lifter) == RangeOnlyLocLifter:
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
            n_outliers=lifter.n_outliers if not n_outliers else n_outliers,
        )
    elif type(lifter) == WahbaLifter:
        new_lifter = WahbaLifter(
            n_landmarks=n_params,
            robust=lifter.robust,
            level=lifter.level,
            d=lifter.d,
            variable_list=None,
            n_outliers=lifter.n_outliers if not n_outliers else n_outliers,
        )
    else:
        raise ValueError(lifter)
    return new_lifter


def plot_scalability_new(df, log=True, start="t "):
    import seaborn as sns

    dict_ = plot_dict[start]

    var_name = dict_["var_name"]
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


def plot_scalability(df, log=True, start="t ", legend_idx=0):
    import seaborn as sns

    dict_ = plot_dict[start]
    var_name = dict_["var_name"]

    df_plot = df[df.type != "original"]

    df_plot = df_plot.dropna(axis=1, inplace=False, how="all")
    df_plot = df_plot.replace(
        {
            "sorted": "\\textsc{AutoTemplate} (suff.)",
            "basic": "\\textsc{AutoTemplate} (all)",
            "from scratch": "\\textsc{AutoTight}",
        }
    )
    df_plot = df_plot.melt(
        id_vars=["N", "type"],
        value_vars=[v for v in df_plot.columns if v.startswith(start)],
        value_name=dict_["value_name"],
        var_name=var_name,
    )
    df_plot.replace(rename_dict, inplace=True)
    var_name_list = list(df_plot[var_name].unique())
    fig, axs = plt.subplots(1, len(var_name_list), sharex=True, sharey=True)

    def plot_here(ax, df_sub, add_legend):
        remove = []
        for type_, df_per_type in df_sub.groupby("type"):
            values = df_per_type[dict_["value_name"]]
            if (~values.isna()).sum() <= 1:
                ax.scatter(df_per_type.N, values, marker="o", label=type_, color="k")
                remove.append(type_)
            if add_legend:
                ax.legend()

        df_sub = df_sub[~df_sub.type.isin(remove)]
        values = df_sub[dict_["value_name"]]
        sns.lineplot(
            df_sub,
            x="N",
            y=dict_["value_name"],
            style="type",
            ax=ax,
            legend=add_legend,
            color="k",
        )

    for i, key in enumerate(var_name_list):
        df_sub = df_plot[df_plot[var_name] == key]
        plot_here(axs[i], df_sub, i == legend_idx)

    for i, ax in enumerate(axs):
        key = var_name_list[i]
        title = ylabels[key]
        ax.set_title(title, visible=True)
        ax.set_yscale("log")
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


def save_tightness_order(
    learner: Learner, fname_root="", use_bisection=False, figsize=4
):
    from matplotlib.ticker import MaxNLocator

    if (learner.df_tight is None) or (len(learner.df_tight) == 0):
        print(f"no tightness data for {learner.lifter}")
        return
    df_current = learner.df_tight[learner.df_tight.lifter == str(learner.lifter)]

    fig_cost, ax_cost = plt.subplots()
    ax_cost.axhline(learner.solver_vars["qcqp_cost"], color="k", label="$q^\\star$")
    for reorder, df in df_current.groupby("reorder"):
        label = "$d^\\star$ (sort.)" if reorder else "$d^\\star$ (orig.)"
        if use_bisection:
            ax_cost.semilogy(df.index, df["dual cost"], label=label, ls="", marker="o")
        else:
            ax_cost.semilogy(df.index, df["dual cost"], label=label, ls="-")

        fig_eigs, ax_eigs = plt.subplots()
        fig_eigs.set_size_inches(figsize, figsize)

        cost_idx = df[df.cost_tight == True].index.min()
        rank_idx = df[df.rank_tight == True].index.min()

        df.sort_index(inplace=True)

        df_valid = df[~df["dual cost"].isna()]
        n_min = df_valid.iloc[0].name
        n_max = df_valid.iloc[-1].name
        n_mid = df_valid.iloc[len(df_valid) // 2].name
        ls = ["--", "-.", ":"]
        for n, ls in zip([n_min, n_mid, n_max], ls):
            eig = df.loc[n].eigs
            if not np.any(np.isfinite(eig)):
                continue
            if n in [cost_idx, rank_idx]:
                continue
            label = f"{n}"
            color = "gray"
            ax_eigs.semilogy(eig, ls=ls, label=label, color=color)

        if cost_idx == rank_idx:
            n = cost_idx
            eig = df.loc[n].eigs
            label = f"{n} (C+R)"
            color = "red"
            ax_eigs.semilogy(eig, ls="-", label=label, color=color)
        else:
            for n in [cost_idx, rank_idx]:
                if not np.isfinite(n):
                    continue
                elif n == cost_idx:
                    label = f"{n} (C)"
                    color = "red"
                elif n == rank_idx:
                    label = f"{n} (R)"
                    color = "black"
                eig = df.loc[n].eigs
                ax_eigs.semilogy(eig, ls="-", label=label, color=color)
        ax_eigs.set_xlabel("index")
        ax_eigs.set_ylabel("eigenvalue")
        ax_eigs.grid(True)

        handles, labels = ax_eigs.get_legend_handles_labels()
        order = np.argsort([int(l.split(" ")[0]) for l in labels])
        ax_eigs.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

        if reorder:
            name = "sorted"
            # ax_eigs.set_title("sorted by dual values")
        else:
            name = "original"
            # ax_eigs.set_title("original order")
        if fname_root != "":
            savefig(fig_eigs, fname_root + f"_tightness-eigs-{name}.pdf")

    if use_bisection:
        if any(df["dual cost"] < 0):
            ax_cost.set_yscale("symlog")
        ax_cost.legend()
    else:
        ax_cost.legend(
            loc="lower right",
        )
    ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cost.set_xlabel("number of added constraints")
    ax_cost.set_ylabel("cost")
    fig_cost.set_size_inches(figsize, figsize)

    ax_cost.grid(True)
    ax_cost.legend()

    if fname_root != "":
        savefig(fig_cost, fname_root + "_tightness-cost.pdf")
    return


def tightness_study(learner: Learner, use_bisection=True):
    """investigate tightness before and after reordering"""
    print("reordering...")
    idx_subset_reorder = learner.generate_minimal_subset(
        reorder=True,
        tightness=learner.lifter.TIGHTNESS,
        use_bisection=use_bisection,
    )
    print("original ordering...")
    idx_subset_original = learner.generate_minimal_subset(
        reorder=False,
        tightness=learner.lifter.TIGHTNESS,
        use_bisection=use_bisection,
    )
    return idx_subset_original, idx_subset_reorder


def run_scalability_plot(learner: Learner, recompute=False, fname_root=""):
    fname = fname_root + "_plot.pkl"
    try:
        assert not recompute, "forcing to recompute"
        with open(fname, "rb") as f:
            learner = pickle.load(f)
            df = pickle.load(f)
        print(f"--------- read {fname} \n")
    except (AssertionError, FileNotFoundError, AttributeError) as e:
        print(e)
        # find which of the constraints are actually necessary
        data, success = learner.run(verbose=False, plot=False)
        df = pd.DataFrame(data)
        with open(fname, "wb") as f:
            pickle.dump(learner, f)
            pickle.dump(df, f)
        print("wrote intermediate as", fname)

    fig, ax = plot_scalability_new(df, start="t ")
    savefig(fig, fname_root + f"_small.pdf")

    idx_subset_reorder = learner.generate_minimal_subset(
        reorder=True, tightness=learner.lifter.TIGHTNESS, use_bisection=True
    )
    templates_poly = learner.generate_templates_poly(factor_out_parameters=False)
    add_columns = {
        "required (sorted)": idx_subset_reorder,
    }
    df = learner.get_sorted_df(templates_poly=templates_poly, add_columns=add_columns)
    title = ""

    fig, ax = learner.save_sorted_templates(
        df, title=title, drop_zero=True, simplify=True
    )
    w, h = fig.get_size_inches()
    fig.set_size_inches(10, 10 * h / w)
    savefig(fig, fname_root + f"_templates.pdf")


def run_scalability_new(
    learner: Learner,
    param_list: list,
    results_folder: str,
    n_seeds: int = 1,
    recompute: bool = False,
):
    fname_root = f"{results_folder}/scalability_{learner.lifter}"
    fname_all = fname_root + "_complete.pkl"
    # try:
    #    assert recompute is False
    #    df = pd.read_pickle(fname_all)
    #    print("read", fname_all)
    #    return df
    # except (AssertionError, FileNotFoundError) as e:
    #    print(e)

    fname = f"{results_folder}/{learner.lifter}.pkl"
    try:
        assert not recompute, "forcing to recompute"
        with open(fname, "rb") as f:
            learner = pickle.load(f)
            orig_dict = pickle.load(f)
        print(f"--------- read {fname} \n")
    except (AssertionError, FileNotFoundError, AttributeError) as e:
        print(e)
        # find which of the constraints are actually necessary
        orig_dict = {}
        t1 = time.time()
        data, success = learner.run(verbose=False, plot=False)
        if not success:
            raise RuntimeError(f"{learner}: did not achieve tightness.")
        orig_dict["t learn templates"] = time.time() - t1

        with open(fname, "wb") as f:
            pickle.dump(learner, f)
            pickle.dump(orig_dict, f)
        print("wrote intermediate as", fname)

    fname = fname_root + "_order_dict.pkl"
    try:
        assert not recompute, "forcing to recompute"
        try:
            with open(fname, "rb") as f:
                order_dict = pickle.load(f)
                learner = pickle.load(f)
        except EOFError:
            learner = None
        print(f"--------- read {fname} \n")

    except (AssertionError, FileNotFoundError, AttributeError) as e:
        print(e)
        t1 = time.time()
        idx_subset_original, idx_subset_reorder = tightness_study(
            learner, use_bisection=learner.lifter.TIGHTNESS == "cost"
        )

        order_dict = {}
        orig_dict["t determine required"] = time.time() - t1
        if idx_subset_reorder is not None:
            order_dict["sorted"] = idx_subset_reorder
        if idx_subset_original is not None:
            order_dict["original"] = idx_subset_original
        # order_dict["all"] = range(len(learner.constraints))
        order_dict["basic"] = None
        with open(fname, "wb") as f:
            pickle.dump(order_dict, f)
            pickle.dump(learner, f)

    # if learner is not None:
    #    save_tightness_order(
    #        learner,
    #        fname_root + "_new",
    #        use_bisection=learner.lifter.TIGHTNESS == "cost",
    #    )

    if EARLY_STOP:
        return None

    fname = fname_root + "_df_all.pkl"
    try:
        assert False
        assert not recompute, "forcing to recompute"
        df = pd.read_pickle(fname)
        # assert set(param_list).issubset(df.N.unique())
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

                    np.random.seed(seed)
                    new_lifter = create_newinstance(learner.lifter, n_params)
                    # doesn't matter because we don't use the usual pipeline.
                    # variable_list = [["h", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
                    new_learner = Learner(
                        lifter=new_lifter,
                        variable_list=new_lifter.variable_list,
                        n_inits=1,
                    )
                    success = new_learner.find_local_solution()
                    if not success:
                        continue
                    n_successful_seeds += 1

                    # extract the templates from constraints
                    t1 = time.time()
                    if new_order is not None:
                        new_learner.templates = learner.get_sufficient_templates(
                            new_order, new_lifter
                        )
                    else:
                        new_learner.templates = learner.templates

                    # apply the templates, to generate constraints
                    new_learner.templates_known = new_learner.get_known_templates()
                    new_learner.apply_templates()
                    data_dict["t create constraints"] = time.time() - t1
                    data_dict["n templates"] = len(new_learner.templates)
                    data_dict["n constraints"] = len(new_learner.constraints)

                    # determine tightness
                    if (
                        isinstance(new_lifter, Stereo3DLifter)
                        and (n_params > 25)
                        and (name == "basic")
                    ):
                        print(
                            f"skipping tightness test of stereo3D with {n_params} because it leeds to memory error"
                        )
                    else:
                        print(f"=========== tightness test: {name} ===============")
                        t1 = time.time()
                        new_learner.is_tight(verbose=False)
                        data_dict[f"t solve SDP"] = time.time() - t1
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
        n_seeds = 1
        max_seeds = n_seeds + 5
        df_data = []
        for n_params in param_list:
            # if isinstance(learner.lifter, RobustPoseLifter) and learner.lifter.robust:
            #    print("cannot do one-shot with robust pose lifters, too expensive!")
            #    df_oneshot = None
            #    break

            if isinstance(learner.lifter, Stereo2DLifter) and n_params >= 20:
                print(f"skipping N={n_params} for stereo2D because so slow.")
                continue
            if isinstance(learner.lifter, Stereo3DLifter) and n_params > 10:
                print(
                    f"skipping N={n_params} for stereo3D because this will cause out of memory error."
                )
                continue
            if isinstance(learner.lifter, RobustPoseLifter) and n_params > 11:
                print(
                    f"skipping N={n_params} for {learner.lifter} because this will cause out of memory error."
                )
                continue

            if (
                isinstance(learner.lifter, RangeOnlyLocLifter)
                and (learner.lifter.level == "quad")
                and (n_params > 15)
            ):
                print(f"skipping N={n_params} for {learner.lifter} because so slow.")
                continue

            n_successful_seeds = 0
            for seed in range(max_seeds):
                print(
                    f"================== oneshot N={n_params},seed={seed} ======================"
                )
                data_dict = {"N": n_params, "seed": seed, "type": "from scratch"}
                new_lifter = create_newinstance(learner.lifter, n_params)

                ############ standard one-shot ############
                new_lifter.param_level = "no"
                variable_list = new_lifter.get_all_variables()
                new_learner = Learner(
                    lifter=new_lifter,
                    variable_list=variable_list,
                    apply_templates=False,
                    n_inits=1,
                )

                success = new_learner.find_local_solution()
                if not success:
                    continue
                n_successful_seeds += 1

                dict_list, success = new_learner.run(verbose=True)
                new_dict = dict_list[-1]
                if not success:
                    raise RuntimeError(
                        f"{new_learner.lifter}: did not achieve tightness."
                    )
                data_dict["t create constraints"] = new_dict["t learn templates"]
                data_dict["t solve SDP"] = new_dict["t check tightness"]
                data_dict["n templates"] = new_dict["n templates"]
                data_dict["n constraints"] = new_dict["n templates"]

                df_data.append(deepcopy(data_dict))
                if n_successful_seeds >= n_seeds:
                    break

            df_oneshot = pd.DataFrame(df_data)
            df_oneshot.to_pickle(fname)
            print("saved oneshot as", fname)

    if df_oneshot is not None:
        df = pd.concat([df, df_oneshot], axis=0)
    df = df.apply(pd.to_numeric, errors="ignore")

    df.to_pickle(fname_all)
    print("wrote df as", fname_all)

    return df


def run_oneshot_experiment(
    learner: Learner,
    fname_root,
    plots,
):
    learner.run(verbose=True, plot=True)

    if "svd" in plots:
        fig = plt.gcf()
        ax = plt.gca()
        l = ax.get_legend()
        if l is not None:
            l.remove()
        fig.set_size_inches(3, 3)
        savefig(fig, fname_root + "_svd.pdf")

    idx_subset_original, idx_subset_reorder = tightness_study(
        learner,
        use_bisection=learner.lifter.TIGHTNESS == "cost",
    )
    if "tightness" in plots:
        save_tightness_order(
            learner,
            fname_root,
            figsize=4,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
        )

    if "matrices" in plots:
        A_matrices = []
        from poly_matrix import PolyMatrix

        for c in learner.constraints:
            if c.A_poly_ is None:
                c.A_poly_, __ = PolyMatrix.init_from_sparse(
                    c.A_sparse_, learner.lifter.var_dict, unfold=True
                )
            # if "x:0" in c.A_poly_.adjacency_i:
            A_matrices.append(c.A_poly_)

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

        if idx_subset_reorder is not None and len(idx_subset_reorder):
            constraints = learner.templates_known + learner.constraints
            A_matrices = [constraints[i - 1].A_poly_ for i in idx_subset_reorder[1:]]
            fig, ax = learner.save_matrices_sparsity(A_matrices)
            savefig(fig, fname_root + "_matrices-sparsity-reorder.pdf")

        if idx_subset_original is not None and len(idx_subset_original):
            constraints = learner.templates_known + learner.constraints
            A_matrices = [constraints[i - 1].A_poly_ for i in idx_subset_original[1:]]
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
        # title = (
        #    f"substitution level: {learner.lifter.LEVEL_NAMES[learner.lifter.level]}"
        # )
        title = ""
        fig, ax = learner.save_sorted_templates(
            df, title=title, drop_zero=True, simplify=True
        )
        w, h = fig.get_size_inches()
        fig.set_size_inches(5, 5 * h / w)
        savefig(fig, fname_root + "_templates.pdf")

        if "templates-full" in plots:
            fig, ax = learner.save_sorted_templates(
                df, title=title, drop_zero=True, simplify=False
            )
            w, h = fig.get_size_inches()
            fig.set_size_inches(5, 5 * h / w)
            savefig(fig, fname_root + "_templates_full.pdf")
