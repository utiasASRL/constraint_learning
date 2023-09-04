from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # non-interactive

from utils.real_experiments import Experiment
from utils.real_experiments import run_all, plot_results, plot_local_vs_global

DATASET_ROOT = str(Path(__file__).parent.parent)
MAX_N_LANDMARKS = 8  # 10
MIN_N_LANDMARKS = 4  # 8

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True


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


if __name__ == "__main__":
    df_list = []
    n_successful = 100

    datasets = ["loop-2d_s4", "eight_s3", "zigzag_s3", "starrynight"]
    for dataset in datasets:
        if USE_GT:
            fname = f"_results/stereo_{dataset}_{n_successful}_gt.pkl"
        else:
            fname = f"_results/stereo_{dataset}_{n_successful}.pkl"
        try:
            assert RECOMPUTE is False
            df_all = pd.read_pickle(fname)
        except (AssertionError, FileNotFoundError):
            exp = load_experiment(dataset)
            df_all = run_all(
                exp,
                min_n_landmarks=MIN_N_LANDMARKS,
                max_n_landmarks=MAX_N_LANDMARKS,
                use_gt=USE_GT,
                sim_noise=SIM_NOISE,
                out_name=fname,
                n_successful=n_successful,
                level="urT",
                stride=1,
            )

        if df_all is not None:
            df_all["dataset"] = dataset
            df_list.append(df_all)

    df = pd.concat(df_list)
    constraint_type = "sorted"
    df = df[df.type == constraint_type]
    df["RDG"] = df["RDG"].abs()
    fname_root = "_results/stereo"

    plot_local_vs_global(df, fname_root=fname_root)
    plot_results(df, ylabel="RDG", fname_root=fname_root)
    plt.show()
    print("done")
