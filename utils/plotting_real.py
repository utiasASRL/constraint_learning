from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import spatialmath as sm
from cert_tools.linalg_tools import project_so3

from auto_template.real_experiments import DEGENERATE_DICT, Experiment, load_experiment
from utils.geometry import get_T
from utils.plotting_tools import add_scalebar, plot_frame, savefig

DATASET_ROOT = str(Path(__file__).parent.parent)

PLOT_LIMITS = {
    "eight_s3": {
        "xlim": [-3, 0],
        "ylim": [-3, 0],
        "zlim": [1, 4],
    },
    "loop-2d_s4": {
        "xlim": [-2, 2],
        "ylim": [-2, 2],
        "zlim": [0, 4],
    },
    "starrynight": {
        "xlim": [1, 4],
        "ylim": [1, 4],
        "zlim": [0, 3],
    },
    "zigzag_s3": {
        "xlim": [-4, 1],
        "ylim": [-2, 3],
        "zlim": [0, 5],
    },
}


def plot_local_vs_global(df, fname_root="", cost_thresh=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    style = {
        "tp": {"color": "C0", "marker": "o"},
        "tn": {"color": "C1", "marker": "x"},
        "fn": {"color": "C2", "marker": "s"},
    }
    for i, row in df.iterrows():
        # for ro, we don't have a certificate (it's always true because rank-1)
        type_ = None
        if row.get("global solution cert", True):  # solution is certified
            if cost_thresh and row.qcqp_cost < cost_thresh:  # it is global minimum
                type_ = "tp"
            elif cost_thresh and row.qcqp_cost >= cost_thresh:  # it is local minimum
                raise ValueError("false positive detected")
            else:
                type_ = "tp"
        else:  # non-certified
            if cost_thresh and row.qcqp_cost < cost_thresh:  # it is a global minimum
                type_ = "fn"
            elif cost_thresh and row.qcqp_cst >= cost_thresh:
                type_ = "tn"
            else:
                type_ = "tn"

        ax.scatter(row["max res"], row.qcqp_cost, **style[type_])
        for key in row.index:
            if key.startswith("local solution") and not ("cert" in key):
                idx = int(key.split("local solution ")[-1])
                cert = row.get(f"local solution {idx} cert", False)
                cost = row[f"local cost {idx}"]
                if not np.isnan(cert) and cert:  # certified
                    if cost_thresh and cost < cost_thresh:
                        type_ = "tp"
                    elif cost_thresh and cost > cost_thresh:
                        raise ValueError("false positive detected")
                    else:
                        type_ = "tp"
                else:
                    if cost_thresh and cost < cost_thresh:
                        type_ = "fn"
                    elif cost_thresh and cost > cost_thresh:
                        type_ = "tn"
                    else:
                        type_ = "tn"
                ax.scatter(row["max res"], cost, **style[type_])

    for key, style_dict in style.items():
        ax.scatter([], [], **style_dict, label=key)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("maximum residual")
    ax.set_ylabel("cost")
    ax.grid()
    if fname_root != "":
        savefig(fig, fname_root + "_local_vs_global.pdf")
    return fig, ax


hue_order = ["loop-2d_s4", "eight_s3", "zigzag_s3", "starrynight"]


def plot_results(df, ylabel="RDG", fname_root="", thresh=None, datasets=hue_order):
    label_names = {"max res": "maximum residual"}
    kwargs = {"edgecolor": "none"}
    for x in ["max res"]:  # ["total error", "cond Hess", "max res", "q"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        sns.scatterplot(
            data=df,
            x=x,
            y=ylabel,
            ax=ax,
            hue="dataset",
            hue_order=datasets,
            style="dataset",
            style_order=datasets,
            **kwargs,
        )
        # ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
        if thresh is not None:
            ax.axhline(thresh, color="k", ls=":", label="tightness threshold")
        ax.legend(framealpha=1.0)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(label_names.get(x, x))
        ax.grid()
        if fname_root != "":
            savefig(fig, f"{fname_root}_{x.replace(' ', '_')}_{ylabel}.pdf")

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    sns.boxplot(
        data=df,
        x="dataset",
        y="success rate",
        ax=ax,
    )
    # if fname_root != "":
    #    savefig(fig, f"{fname_root}_successrate.pdf")


def plot_ground_truth(df_all, fname_root="", rangeonly=False):
    for dataset, df in df_all.groupby("dataset"):
        if rangeonly:
            exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)
        else:
            exp = load_experiment(dataset)
        try:
            landmarks = exp.all_landmarks[["x", "y", "z"]].values
        except IndexError:
            landmarks = exp.all_landmarks
        fig, ax = plt.subplots()
        ax.scatter(*landmarks[:, :2].T, color="k", marker="+")
        for __, row in df.iterrows():
            if "gt theta" not in row:
                continue

            if dataset == "starrynight":
                plot_frame(
                    ax,
                    theta=row["gt theta"],
                    color="blue",
                    marker="o",
                    scale=0.1,
                )
            else:
                plot_frame(
                    ax,
                    theta=row["gt theta"],
                    color="blue",
                    marker="o",
                    scale=1.0,
                )

        fig.set_size_inches(5, 5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        add_scalebar(ax, size=1, size_vertical=0.03, loc="lower right", fontsize=40)
        ax.axis("off")
        savefig(fig, f"{fname_root}_{dataset}_poses.pdf")


def plot_positions(df_all, fname_root=""):
    for (n_landmarks, dataset), df in df_all.groupby(
        ["n landmarks", "dataset"], sort=False
    ):
        exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)
        for chosen_idx, df_sub in df.groupby("chosen idx"):
            if chosen_idx >= 0:
                landmark_ids = DEGENERATE_DICT[chosen_idx]
                landmarks = exp.all_landmarks.loc[
                    exp.all_landmarks.id.isin(landmark_ids), ["x", "y", "z"]
                ].values
            else:
                landmarks = exp.all_landmarks[["x", "y", "z"]].values

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                landmarks[:, 2],
                marker="x",
                color="k",
            )

            theta = np.vstack(df_sub["gt theta"].values)
            ax.plot(theta[:, 0], theta[:, 1], theta[:, 2], color="k")

            theta = np.vstack(df_sub["global theta"].values)
            ax.scatter(theta[:, 0], theta[:, 1], theta[:, 2], color="green")

            for i in range(10):
                label = f"local solution {i}"
                if label in df.columns:
                    values = [row for row in df[label] if np.ndim(row) > 0]
                    if len(values):
                        theta = np.vstack(values)
                        ax.scatter(theta[:, 0], theta[:, 1], theta[:, 2], color="red")

            fig.set_size_inches(10, 10)

            default = [np.min(landmarks), np.max(landmarks)]
            ax.set_xlim(*PLOT_LIMITS.get(dataset, {}).get("xlim", default))
            ax.set_ylim(*PLOT_LIMITS.get(dataset, {}).get("ylim", default))
            ax.set_zlim(*PLOT_LIMITS.get(dataset, {}).get("zlim", default))
            origin = np.eye(4)
            centroid = [ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]
            origin[:3, 3] = centroid
            pose = sm.SE3(origin)
            pose.plot(
                ax=ax,
                color="k",
                length=1.0,
                axislabel=True,
                axislabels=["x", "y", "z"],
                origincolor="k",
            )
            ax.view_init(elev=20.0, azim=-45)
            ax.axis("off")
            savefig(fig, f"{fname_root}_{dataset}_{n_landmarks}.pdf")
    # ce7a69

    return


def plot_success_rate(df_all, fname_root):
    import seaborn as sns

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    sns.lineplot(df_all, x="n landmarks", y="success rate", ax=ax, hue="dataset")
    savefig(fig, f"{fname_root}_success.pdf")
    return


def plot_poses(df_all, fname_root=""):
    for dataset, df in df_all.groupby("dataset"):
        exp = load_experiment(dataset)
        try:
            landmarks = exp.all_landmarks[["x", "y", "z"]].values
        except IndexError:
            landmarks = exp.all_landmarks
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        scale = abs(np.min(landmarks) - np.max(landmarks))
        default_kwargs = dict(
            axislabel=False,
            originsize=scale,  # default is 20
            style="line",
            ax=ax,
        )

        ax.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            landmarks[:, 2],
            marker="x",
            color="k",
        )
        coordinates = []
        for i, row in df.iterrows():
            if ("global theta" in row) and np.ndim(row["global theta"]):
                # plot global
                T_c0 = get_T(theta=row["global theta"], d=3)
                pose = sm.SE3(T_c0)
                pose = pose.inv()  # T_0c
                pose.plot(
                    color=["r", "g", "b"],
                    origincolor="green",
                    length=scale * 0.02,
                    **default_kwargs,
                )

            T_c0 = get_T(theta=row["gt theta"], d=3)  # T_0c
            pose = sm.SE3(T_c0)
            pose = pose.inv()
            ax.scatter(*pose.t, color="k", s=1)
            coordinates.append(pose.t[None, :])

            # plot local
            local_solution_labels = row.filter(
                regex="local solution [0-9]",
            )
            for l in local_solution_labels.index:
                if np.ndim(row[l]) > 0:
                    T_c0 = get_T(theta=row[l], d=3)
                    # project to orthogonal matrices.
                    T_c0 = project_so3(T_c0)
                    pose = sm.SE3(T_c0)
                    pose = pose.inv()
                    pose.plot(
                        color=["#ce7a69", "#84c464", "#6482c4"],  # pastel colors
                        origincolor="red",
                        length=scale * 0.01,
                        **default_kwargs,
                    )

        coordinates = np.vstack(coordinates)
        # plot ground truth
        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color="k",
            ls="-",
            alpha=0.5,
        )
        fig.set_size_inches(10, 10)
        ax.view_init(elev=20.0, azim=-45)
        default = [np.min(landmarks), np.max(landmarks)]
        ax.set_xlim(*PLOT_LIMITS.get(dataset, {}).get("xlim", default))
        ax.set_ylim(*PLOT_LIMITS.get(dataset, {}).get("ylim", default))
        ax.set_zlim(*PLOT_LIMITS.get(dataset, {}).get("zlim", default))
        origin = np.eye(4)
        # centroid = np.min(landmarks, axis=0)
        # centroid = np.mean(landmarks, axis=0)
        centroid = [ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]
        origin[:3, 3] = centroid
        pose = sm.SE3(origin)
        pose.plot(
            ax=ax,
            color="k",
            length=1.0,
            axislabel=True,
            axislabels=["x", "y", "z"],
            origincolor="k",
        )

        ax.axis("off")
        savefig(fig, f"{fname_root}_{dataset}_all.pdf")
    return
