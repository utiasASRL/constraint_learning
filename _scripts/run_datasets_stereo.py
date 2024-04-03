import matplotlib
import matplotlib.pylab as plt
import pandas as pd

try:
    matplotlib.use("TkAgg")  # non-interactive
except ImportError:
    pass

from auto_template.learner import TOL_RANK_ONE, TOL_REL_GAP
from auto_template.real_experiments import (
    create_rmse_table,
    load_experiment,
    run_experiments,
)
from utils.plotting_real import (
    plot_ground_truth,
    plot_local_vs_global,
    plot_poses,
    plot_results,
    plot_success_rate,
)

MAX_N_LANDMARKS = 8
MIN_N_LANDMARKS = 4

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = True

RESULTS_DIR = "_results_v4"


def run_all(recompute=RECOMPUTE, n_successful=10, results_dir=RESULTS_DIR):
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
    df["success rate"] = df["n global"] / (
        df["n global"] + df["n fail"] + df["n local"]
    )

    plot_poses(df, fname_root=fname_root)

    plot_success_rate(df, fname_root)

    plot_ground_truth(df, fname_root=fname_root)

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
    run_all(n_successful=100, recompute=False)
