from copy import deepcopy
import itertools

import numpy as np
import pandas as pd

from utils.plotting_tools import import_plt, savefig

plt = import_plt()


def noise_study(learner):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    seed = 0
    n_landmark_list = [4, 5, 6]  # , 7, 8, 9, 10]
    noise_list = np.logspace(0, 2, 3)
    tightness = "cost"

    data = []
    for noise, n_landmarks in itertools.product(noise_list, n_landmark_list):
        print(
            f"============= noise {noise} / n_landmarks {n_landmarks} ================"
        )
        np.random.seed(seed)

        data_here, success = learner.run(
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
