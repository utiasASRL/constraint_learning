import matplotlib
import matplotlib.pylab as plt

from _scripts.run_autotemplate import run_all as run_autotemplate
from _scripts.run_datasets_ro import run_all as run_datasets_ro
from _scripts.run_datasets_stereo import run_all as run_datasets_stereo
from _scripts.run_other_study import run_all as run_other_study
from _scripts.run_range_only_study import run_all as run_range_only_study
from _scripts.run_stereo_study import run_all as run_stereo_study
from auto_template.real_experiments import create_rmse_table

try:
    matplotlib.use("TkAgg")
    # matplotlib.use("Agg") # no plotting
except Exception as e:
    pass

RESULTS_DIR = "_results_server_v3/"
# RESULTS_DIR = "_results_v4/"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Run experiments. Use --recompute to regenerate all results. Otherwise plots exisitng data and recomputes missing."
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        default=False,
        help="regenerate results",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=RESULTS_DIR,
        help="results directory",
    )
    parser.add_argument(
        "-n",
        "--n_seeds",
        default=10,
        help="number of random seeds",
    )
    args = parser.parse_args()
    recompute = args.overwrite
    results_dir = args.directory
    n_seeds = int(args.n_seeds)
    autotight = True
    autotemplate = True
    n_successful = 100
    print("n seeds:", n_seeds)

    print("------- Generate results table and templates-------")
    run_autotemplate(recompute=recompute, results_dir=results_dir)
    plt.close("all")

    print("------- Generate RO results -------")
    run_range_only_study(
        n_seeds=n_seeds,
        recompute=recompute,
        autotight=autotight,
        autotemplate=autotemplate,
        results_dir=results_dir,
    )
    plt.close("all")

    print("------- Generate other results -------")
    run_other_study(
        n_seeds=n_seeds,
        recompute=recompute,
        autotight=autotight,
        autotemplate=autotemplate,
        results_dir=results_dir,
    )
    plt.close("all")

    print("------- Generate stereo results -------")
    run_stereo_study(
        n_seeds=n_seeds,
        recompute=recompute,
        autotight=autotight,
        autotemplate=autotemplate,
        results_dir=results_dir,
    )
    plt.close("all")

    print("------- Generate dataset results -------")
    run_datasets_ro(
        recompute=recompute, n_successful=n_successful, results_dir=results_dir
    )
    plt.close("all")

    run_datasets_stereo(
        recompute=recompute, n_successful=n_successful, results_dir=results_dir
    )
    plt.close("all")
