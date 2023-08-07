import itertools
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import matplotlib

matplotlib.use("TkAgg")
plt.ion()
# matplotlib.use('Agg') # non-interactive
# plt.ioff()

from experiments import run_scalability_new, run_oneshot_experiment
from experiments import plot_scalability, save_table
from utils.plotting_tools import savefig

from lifters.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from utils.plotting_tools import savefig


def stereo_tightness(d=2, n_landmarks=None):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    if n_landmarks is None:
        n_landmarks = d + 1
    seed = 0

    # parameter_levels = ["ppT"] #["no", "p", "ppT"]
    levels = ["no", "urT"]
    param_level = "no"
    for level in levels:
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        if d == 2:
            # plots = ["tightness"]#, "matrix"]
            # tightness = "rank"

            plots = ["matrices", "templates", "svd"]
            tightness = "cost"

            lifter = Stereo2DLifter(
                n_landmarks=n_landmarks,
                level=level,
                param_level=param_level,
                variable_list=variable_list,
            )
        elif d == 3:
            plots = ["tightness"]
            tightness = "cost"

            # plots = ["matrices", "templates", "svd"]
            # tightness = "cost"

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

        run_oneshot_experiment(
            learner, fname_root, plots, tightness=tightness, add_original=True
        )


def stereo_scalability_new(d=2):
    if d == 2:
        n_landmarks_list = [5, 10, 15, 20, 25, 30]
        n_seeds = 10
    elif d == 3:
        n_landmarks_list = [10, 15, 20, 25, 30]
        n_seeds = 10 
    level = "urT"
    param_level = "ppT"

    n_landmarks = d + 1

    # variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]] runs out of memory for d=3
    variable_list = None  # use the default one for the first step.
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
    #run_scalability_plot(learner)
    df = run_scalability_new(
        learner, param_list=n_landmarks_list, n_seeds=n_seeds, recompute=False, use_bisection=True, use_known=False, add_original=False
    )

    fname_root = f"_results/scalability_{learner.lifter}"

    fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=0)

    #[ax.set_ylim(10, 1000) for ax in axs.values()]
    fig.set_size_inches(5, 3)
    axs["t create constraints"].legend(loc="lower right")
    savefig(fig, fname_root + f"_t.pdf")
    #fig, ax = plot_scalability(df, log=True, start="n ")
    #axs["t solve SDP"].legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
    #fig.set_size_inches(5, 5)
    #savefig(fig, fname_root + f"_n.pdf")
    #tex_name = fname_root + f"_n.tex"
    #save_table(df, tex_name)


def stereo_3d_study():
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    seed = 0
    n_landmark_list = [4, 5, 6]  # , 7, 8, 9, 10]
    noise_list = np.logspace(0, 2, 3)

    level = "urT"
    param_level = "no"

    tightness = "cost"
    data = []
    for noise, n_landmarks in itertools.product(noise_list, n_landmark_list):
        print(
            f"============= noise {noise} / n_landmarks {n_landmarks} ================"
        )
        np.random.seed(seed)

        variable_list = [["l", "x"] + [f"z_{i}" for i in range(n_landmarks)]]

        lifter = Stereo3DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
        learner = Learner(
            lifter=lifter,
            variable_list=lifter.variable_list,
            apply_templates=False,
            noise=noise,
        )
        data_here = learner.run(
            verbose=True, use_known=False, plot=False, tightness=tightness
        )[0]
        dcost = learner.dual_costs[-1]
        qcost = learner.solver_vars["qcqp_cost"]
        data_here.update(
            {"value": dcost, "cost": "dual", "noise": noise, "n_landmarks": n_landmarks}
        )
        data.append(deepcopy(data_here))
        data_here.update(
            {"value": qcost, "cost": "qcqp", "noise": noise, "n_landmarks": n_landmarks}
        )
        data.append(deepcopy(data_here))
        ratio = abs(qcost - dcost) / qcost if dcost is not None else None
        data_here.update(
            {
                "value": ratio,
                "cost": "ratio",
                "noise": noise,
                "n_landmarks": n_landmarks,
            }
        )
        data.append(deepcopy(data_here))

    import seaborn as sns

    fname = f"_results/stereo3d_df.pkl"
    df = pd.DataFrame(data)
    df.to_pickle(fname)

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    markers = {"qcqp": "o", "dual": "x"}
    colors = {n: f"C{i}" for i, n in enumerate(noise_list)}
    for [cost, noise], df_ in df.groupby(["cost", "noise"]):
        if cost == "ratio":
            continue
        label = f"{noise:.1f}" if cost == "qcqp" else None
        ax.scatter(
            df_.n_landmarks,
            df_.value,
            color=colors[noise],
            marker=markers[cost],
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(df_.n_landmarks.unique())
    ax.set_xlabel("number of landmarks")
    ax.set_ylabel("dual (x) vs. QCQP (o) cost")
    ax.legend(bbox_to_anchor=[1.0, 1.0], loc="upper left", title="noise")
    fname = f"_results/stereo3d_seed{seed}_study_cost.pdf"
    savefig(fig, fname)

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    colors = {n: f"C{i}" for i, n in enumerate(n_landmark_list)}
    for n, df_ in df.groupby("n_landmarks"):
        df_plot = df_[df_.cost == "ratio"]
        label = f"{n:.0f}"
        ax.loglog(df_plot.noise, df_plot.value, label=label)
    # ax.legend(bbox_to_anchor=[1.0, 1.0], loc="upper left", title="number of \n landmarks")
    ax.legend(loc="lower right", title="number of \n landmarks")
    ax.set_xlabel("noise")
    ax.set_ylabel("relative duality gap")
    fname = f"_results/stereo3d_seed{seed}_study_absratio.pdf"
    savefig(fig, fname)

    pt = df.pivot_table(index=["n_landmarks", "noise"], columns="cost", values="value")
    pt["ratio2"] = (pt["qcqp"] - pt["dual"]) / pt["qcqp"]
    pt["ratio3"] = pt["qcqp"] - pt["dual"]
    for ratio in ["ratio", "ratio2", "ratio3"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        sns.scatterplot(
            pt,
            ax=ax,
            x="noise",
            hue="n_landmarks",
            color=colors,
            y=ratio,
            palette="tab10",
        )
        ax.legend(title="number of landmarks")
        ax.set_xscale("log")
        ax.set_yscale("symlog")
        ax.set_ylim(-0.1, None)
        ax.axhline(0, color="k")
        ax.set_xticks(pt.index.get_level_values("noise").unique())
        ax.grid()
        fname = f"_results/stereo3d_seed{seed}_study_{ratio}.pdf"
        savefig(fig, fname)


if __name__ == "__main__":
    # import warnings
    # with warnings.catch_warnings():
    #    warnings.simplefilter("error")

    # stereo_scalability(d=2)
    #stereo_scalability(d=3)

    #stereo_scalability_new(d=2)
    stereo_scalability_new(d=3)

    #stereo_tightness(d=2)
    #stereo_tightness(d=3)

    #stereo_3d_study()