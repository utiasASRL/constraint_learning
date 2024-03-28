from pathlib import Path

import matplotlib
import matplotlib.pylab as plt
import pandas as pd

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP

try:
    matplotlib.use("TkAgg")  # non-interactive
except:
    pass

from auto_template.real_experiments import (
    Experiment,
    plot_local_vs_global,
    plot_results,
    run_experiments,
)

DATASET_ROOT = str(Path(__file__).parent.parent)
MAX_N_LANDMARKS = 8
MIN_N_LANDMARKS = 4

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True

RESULTS_DIR = "_results_server"


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


def run_all(recompute=RECOMPUTE, n_successful=10):
    df_list = []

    # don't change order! (because of plotting)
    datasets = ["loop-2d_s4", "eight_s3", "zigzag_s3", "starrynight"]
    datasets = ["starrynight"]
    for dataset in datasets:
        if USE_GT:
            fname = f"{RESULTS_DIR}/stereo_{dataset}_{n_successful}_gt.pkl"
        else:
            fname = f"{RESULTS_DIR}/stereo_{dataset}_{n_successful}.pkl"
        try:
            assert recompute is False
            df_all = pd.read_pickle(fname)
            print(f"read {fname}")
        except (AssertionError, FileNotFoundError):
            exp = load_experiment(dataset)
            df_all = run_experiments(
                exp,
                min_n_landmarks=MIN_N_LANDMARKS,
                max_n_landmarks=MAX_N_LANDMARKS,
                use_gt=USE_GT,
                sim_noise=SIM_NOISE,
                out_name=fname,
                n_successful=n_successful,
                level="urT",
                stride=1,
                results_dir=RESULTS_DIR,
            )

        if df_all is not None:
            df_all["dataset"] = dataset
            df_list.append(df_all)

    df = pd.concat(df_list)
    constraint_type = "sorted"
    df = df[df.type == constraint_type]
    df["RDG"] = df["RDG"].abs()
    fname_root = f"{RESULTS_DIR}/stereo"

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
    plt.show()
    print("done")


if __name__ == "__main__":
    run_all(n_successful=20)
