import itertools
from pathlib import Path

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import spatialmath as sm

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP
from auto_template.real_experiments import (
    DEGENERATE_DICT,
    Experiment,
    create_rmse_table,
    run_experiments,
)
from utils.plotting_real import (
    plot_ground_truth,
    plot_local_vs_global,
    plot_positions,
    plot_results,
    plot_success_rate,
)
from utils.plotting_tools import add_scalebar, plot_frame, savefig

# matplotlib.use("Agg")  # non-interactive


DATASET_ROOT = str(Path(__file__).parent.parent)

MIN_N_LANDMARKS = 4
N_LANDMARKS_LIST = [4, 5, 6]

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True

RESULTS_DIR = "_results_v4"


def run_all(recompute=RECOMPUTE, n_successful=100, results_dir=RESULTS_DIR):
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

    plot_ground_truth(df, fname_root=fname_root, rangeonly=True)

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
    run_all(n_successful=100, recompute=False)
