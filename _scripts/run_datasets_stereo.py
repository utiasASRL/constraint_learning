from pathlib import Path

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from cert_tools.linalg_tools import project_so3

try:
    matplotlib.use("TkAgg")  # non-interactive
except ImportError:
    pass

import spatialmath as sm

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP
from auto_template.real_experiments import (
    Experiment,
    create_rmse_table,
    plot_local_vs_global,
    plot_results,
    run_experiments,
)
from utils.geometry import get_T
from utils.plotting_tools import savefig

DATASET_ROOT = str(Path(__file__).parent.parent)
MAX_N_LANDMARKS = 8
MIN_N_LANDMARKS = 4

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True

RESULTS_DIR = "_results_v4"


def load_experiment(dataset):
    if dataset == "starrynight":
        exp = Experiment(
            dataset_root=DATASET_ROOT, dataset="starrynight", data_type="stereo"
        )
    else:
        data_type = "apriltag_cal_individual"
        exp = Experiment(
            dataset_root=DATASET_ROOT, dataset=dataset, data_type=data_type
        )
    return exp


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
        ax.set_xlim(np.min(landmarks), np.max(landmarks))
        ax.set_ylim(np.min(landmarks), np.max(landmarks))
        ax.set_zlim(np.min(landmarks), np.max(landmarks))
        lims = {
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
        default = [np.min(landmarks), np.max(landmarks)]
        ax.set_xlim(*lims.get(dataset, {}).get("xlim", default))
        ax.set_ylim(*lims.get(dataset, {}).get("ylim", default))
        ax.set_zlim(*lims.get(dataset, {}).get("zlim", default))
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


def run_all(recompute=RECOMPUTE, n_successful=10, stride=1, results_dir=RESULTS_DIR):
    df_list = []

    fname_root = f"{results_dir}/stereo"
    # don't change order! (because of plotting)
    datasets = ["loop-2d_s4", "eight_s3", "zigzag_s3", "starrynight"]
    for dataset in datasets:
        if USE_GT:
            fname = f"{fname_root}_{dataset}_{n_successful}_gt.pkl"
        else:
            fname = f"{fname_root}_{dataset}_{n_successful}.pkl"
        try:
            assert recompute is False
            df_all = pd.read_pickle(fname)
            print(f"read {fname}")
        except (AssertionError, FileNotFoundError):
            exp = load_experiment(dataset)
            exp.set_params(
                min_n_landmarks=MIN_N_LANDMARKS,
                max_n_landmarks=MAX_N_LANDMARKS,
                use_gt=USE_GT,
                sim_noise=SIM_NOISE,
                level="urT",
            )
            df_all = run_experiments(
                exp,
                out_name=fname,
                n_successful=n_successful,
                stride=stride,
                results_dir=results_dir,
            )

        if df_all is not None:
            df_all["dataset"] = dataset
            df_list.append(df_all)

    fname_root = f"{results_dir}/stereo"
    df = pd.concat(df_list)
    fname = fname_root + f"_dataset_errors_n{n_successful}.pkl"
    df.to_pickle(fname)
    print(f"result df as {fname}")

    constraint_type = "sorted"
    df = df[df.type == constraint_type]
    df["RDG"] = df["RDG"].abs()

    plot_poses(df, fname_root=fname_root)

    # cost below is found empirically
    plot_local_vs_global(df, fname_root=fname_root, cost_thresh=1e3)

    plot_results(
        df,
        ylabel="RDG",
        fname_root=fname_root,
        thresh=TOL_REL_GAP,
        datasets=datasets,
    )
    plot_results(
        df,
        ylabel="SVR",
        fname_root=fname_root,
        thresh=TOL_RANK_ONE,
        datasets=datasets,
    )
    create_rmse_table(df, fname_root=fname_root)
    plt.show()


if __name__ == "__main__":
    # run many and plot distributions
    run_all(n_successful=100, stride=1, recompute=False, results_dir=RESULTS_DIR)
