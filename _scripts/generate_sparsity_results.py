from _scripts.run_clique_study import run_hierarchy_study
from _scripts.run_examples import run_example
from _scripts.run_tightness_study import (
    plot_accuracy_study_all,
    plot_success_study_all,
    run_tightness_study,
)
from _scripts.run_time_study import run_time_study

DEBUG = False
RESULTS_DIR = "_results_sparsity_server"

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
        default=10 if not DEBUG else 3,
        help="number of random seeds",
    )

    args = parser.parse_args()

    n_seeds = args.n_seeds
    results_dir = args.directory
    overwrite = args.overwrite

    print("========== running exampels ===========")
    # run_example(results_dir, "exampleRO")
    # run_example(results_dir, "exampleMW")

    print("========== running tightness study ===========")
    run_tightness_study(
        results_dir,
        overwrite=overwrite,
        n_seeds=n_seeds,
        appendix="noisetest" if DEBUG else "noise",
    )
    print("========== plotting accuracy study ===========")
    plot_success_study_all(results_dir, appendix="noisetest" if DEBUG else "noise")
    plot_accuracy_study_all(results_dir, appendix="noisetest" if DEBUG else "noise")

    print("========== running timing study ===========")
    run_time_study(
        results_dir,
        overwrite=overwrite,
        n_seeds=n_seeds,
        appendix="timetest" if DEBUG else "time",
    )
    # print("========== running hierarchy study ===========")
    # run_hierarchy_study(
    #     results_dir,
    #     overwrite=overwrite,
    #     n_seeds=n_seeds,
    #     appendix="hierarchytest" if DEBUG else "hierarchy",
    # )
