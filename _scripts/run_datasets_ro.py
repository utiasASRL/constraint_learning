from pathlib import Path

import matplotlib
import matplotlib.pylab as plt

try:
    matplotlib.use("TkAgg")  # non-interactive
except Exception as e:
    pass

import pandas as pd

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP
from auto_template.real_experiments import (
    Experiment,
    plot_local_vs_global,
    plot_results,
    run_experiments,
)

DATASET_ROOT = str(Path(__file__).parent.parent)

MAX_N_LANDMARKS = 4
MIN_N_LANDMARKS = 4

USE_GT = False
SIM_NOISE = 1.0

RECOMPUTE = True

RESULTS_DIR = "_results_server"


def run_all(recompute=RECOMPUTE, n_successful=100, plot_poses=False):
    level = "quad"

    datasets = ["loop-2d_s4", "eight_s3", "zigzag_s3"]
    df_list = []
    for dataset in datasets:
        exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)

        if USE_GT:
            fname = f"{RESULTS_DIR}/ro_{dataset}_{level}_{n_successful}_gt.pkl"
        else:
            fname = f"{RESULTS_DIR}/ro_{dataset}_{level}_{n_successful}.pkl"

        try:
            assert recompute is False
            df_all = pd.read_pickle(fname)
            print(f"read {fname}")
        except (AssertionError, FileNotFoundError):
            df_all = run_experiments(
                exp,
                min_n_landmarks=MIN_N_LANDMARKS,
                max_n_landmarks=MAX_N_LANDMARKS,
                use_gt=USE_GT,
                sim_noise=SIM_NOISE,
                out_name=fname,
                n_successful=n_successful,
                level=level,
                stride=20,
                plot_poses=plot_poses,
            )
            # df_all.to_pickle(fname)
        if df_all is not None:
            df_all["dataset"] = dataset
            df_list.append(df_all)

    if n_successful > 10:
        df = pd.concat(df_list)
        constraint_type = "sorted"
        df = df[df.type == constraint_type]
        df["RDG"] = df["RDG"].abs()
        fname_root = f"{RESULTS_DIR}/ro_{level}"

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
        print("done")


if __name__ == "__main__":
    run_all()
