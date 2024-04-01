import itertools
from pathlib import Path

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import spatialmath as sm

from auto_template.real_experiments import DEGENERATE_DICT

try:
    matplotlib.use("TkAgg")  # non-interactive
except ImportError:
    pass

import pandas as pd

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP
from auto_template.real_experiments import (
    PLOT_LIMITS,
    Experiment,
    create_rmse_table,
    plot_local_vs_global,
    plot_results,
    run_experiments,
)
from utils.plotting_tools import savefig

DATASET_ROOT = str(Path(__file__).parent.parent)

MIN_N_LANDMARKS = 4
N_LANDMARKS_LIST = [4, 5, 6]

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True

RESULTS_DIR = "_results_v4"


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


def run_all(
    recompute=RECOMPUTE, n_successful=100, plot_poses=False, results_dir=RESULTS_DIR
):
    level = "quad"

    df_list = []

    datasets = ["loop-2d_s4", "eight_s3", "zigzag_s3"]

    fname_root = f"{results_dir}/ro"
    for dataset in datasets:
        exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)
        for max_n_landmarks in N_LANDMARKS_LIST:
            if USE_GT:
                fname = f"{fname_root}_{dataset}_{level}_{n_successful}_{max_n_landmarks}_gt.pkl"
            else:
                fname = f"{fname_root}_{dataset}_{level}_{n_successful}_{max_n_landmarks}.pkl"

            try:
                assert recompute is False
                df_all = pd.read_pickle(fname)
                print(f"read {fname}")
            except (AssertionError, FileNotFoundError):
                exp.set_params(
                    min_n_landmarks=MIN_N_LANDMARKS,
                    max_n_landmarks=max_n_landmarks,
                    use_gt=USE_GT,
                    sim_noise=SIM_NOISE,
                    level=level,
                )
                df_all = run_experiments(
                    exp,
                    out_name=fname,
                    n_successful=n_successful,
                    stride=20,
                    plot_poses=plot_poses,
                    results_dir=results_dir,
                    start_idx=10,
                )
                # df_all.to_pickle(fname)
            if df_all is not None:
                df_all["dataset"] = dataset
                df_list.append(df_all)

    fname_root = f"{results_dir}/ro_{level}"
    df = pd.concat(df_list)
    fname = fname_root + f"_dataset_errors_n{n_successful}.pkl"
    df.to_pickle(fname)
    print(f"saved result df as {fname}")

    constraint_type = "sorted"
    df = df[df.type == constraint_type]
    df["RDG"] = df["RDG"].abs()
    df["success rate"] = df["n global"] / (
        df["n global"] + df["n fail"] + df["n local"]
    )
    create_rmse_table(df, fname_root=fname_root, add_n_landmarks=False)
    plot_success_rate(df, fname_root=fname_root)

    df_here = df[df.dataset == "loop-2d_s4"]
    create_rmse_table(df_here, fname_root=fname_root, add_n_landmarks=True)

    plot_positions(df, fname_root=fname_root)

    plot_local_vs_global(df, fname_root=fname_root)

    plot_results(
        df,
        ylabel="SVR",
        fname_root=fname_root,
        thresh=TOL_RANK_ONE,
        datasets=datasets,
    )
    plot_results(
        df,
        ylabel="RDG",
        fname_root=fname_root,
        thresh=TOL_REL_GAP,
        datasets=datasets,
    )

    plt.show()


if __name__ == "__main__":
    # run many and plot distributions
    run_all(n_successful=100, plot_poses=False, recompute=False)
