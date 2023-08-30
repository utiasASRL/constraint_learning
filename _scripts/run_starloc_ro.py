from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd

from utils.real_experiments import Experiment, run_all, plot_results

DATASET_ROOT = str(Path(__file__).parent.parent)

MAX_N_LANDMARKS = None
MIN_N_LANDMARKS = None

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = False


if __name__ == "__main__":
    datasets = ["zigzag_s3", "loop-2d_s4", "eight_s3"]
    # level = "quad"
    level = "no"
    n_successful = 100

    df_list = []
    for dataset in datasets:
        exp = Experiment(DATASET_ROOT, dataset, data_type="uwb", from_id=1)

        if USE_GT:
            fname = f"_results/{dataset}_output_{level}_{n_successful}.pkl"
        else:
            fname = f"_results/{dataset}_output_{level}_{n_successful}_gt.pkl"

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
            )
        df_all["dataset"] = dataset
        df_list.append(df_all)

    df = pd.concat(df_list)

    fname_root = f"_results/ro_{level}"
    plot_results(df, ylabel="SVR", fname_root=fname_root)
    plt.show()
    print("done")
