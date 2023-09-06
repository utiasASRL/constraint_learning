from pathlib import Path

import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")  # non-interactive

import pandas as pd

from utils.real_experiments import (
    Experiment,
    run_all,
    plot_results,
    plot_local_vs_global,
)

DATASET_ROOT = str(Path(__file__).parent.parent)

MAX_N_LANDMARKS = 4
MIN_N_LANDMARKS = 4

USE_GT = False
SIM_NOISE = 1.0

RECOMPUTE = True

RESULTS_DIR = "_results"


if __name__ == "__main__":
    datasets = ["zigzag_s3", "loop-2d_s4", "eight_s3"]
    n_successful = 10  # was 100
    level = "quad"

    df_list = []
    for dataset in datasets:
        exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)

        if USE_GT:
            fname = f"{RESULTS_DIR}/ro_{dataset}_{level}_{n_successful}_gt.pkl"
        else:
            fname = f"{RESULTS_DIR}/ro_{dataset}_{level}_{n_successful}.pkl"

        try:
            assert RECOMPUTE is False
            df_all = pd.read_pickle(fname)
        except (AssertionError, FileNotFoundError):
            df_all = run_all(
                exp,
                min_n_landmarks=MIN_N_LANDMARKS,
                max_n_landmarks=MAX_N_LANDMARKS,
                use_gt=USE_GT,
                sim_noise=SIM_NOISE,
                out_name=fname,
                n_successful=n_successful,
                level=level,
                stride=20,
            )
            # df_all.to_pickle(fname)
        if df_all is not None:
            df_all["dataset"] = dataset
            df_list.append(df_all)

    df = pd.concat(df_list)

    fname_root = f"{RESULTS_DIR}/ro_{level}"
    plot_local_vs_global(df, fname_root=fname_root)

    from lifters.learner import TOL_RANK_ONE, TOL_REL_GAP

    plot_results(df, ylabel="SVR", fname_root=fname_root, thresh=TOL_RANK_ONE)
    plot_results(df, ylabel="RDG", fname_root=fname_root, thresh=TOL_REL_GAP)
    plt.show()
    print("done")
