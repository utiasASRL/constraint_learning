import matplotlib

from _scripts.run_clique_study import run_hierarchy_study
from _scripts.run_examples import run_example
from _scripts.run_tightness_study import (
    plot_accuracy_study_all,
    plot_success_study_all,
    run_tightness_study,
)
from _scripts.run_time_study import run_time_study

DEBUG = True
OVERWRITE = True
HEADLESS = True
RESULTS_DIR = "_results_sparsity"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Run experiments. Use --recompute to regenerate all results. Otherwise plots exisitng data and recomputes missing."
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="regenerate results",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=RESULTS_DIR,
        help="results directory",
    )
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        default=DEBUG,
        help="run in debug mode",
    )
    parser.add_argument(
        "-n",
        "--no_windows",
        action="store_true",
        default=HEADLESS,
        help="do not open figure windows",
    )

    args = parser.parse_args()
    if args.no_windows:
        matplotlib.use("Agg")

    debug = args.debug
    results_dir = args.directory
    overwrite = args.overwrite

    print("========== running exampels ===========")
    run_example(results_dir, "exampleRO")
    run_example(results_dir, "exampleMW")

    print("========== running tightness study ===========")
    run_tightness_study(results_dir, overwrite=overwrite, debug=debug)
    print("========== plotting accuracy study ===========")
    plot_success_study_all(results_dir, debug=debug)
    plot_accuracy_study_all(results_dir, debug=debug)

    print("========== running timing study ===========")
    run_time_study(
        results_dir,
        overwrite=overwrite,
        debug=debug,
    )
    print("done")
