from _scripts.run_clique_study import run_hierarchy_study
from _scripts.run_examples import run_example
from _scripts.run_tightness_study import (
    plot_accuracy_study_all,
    plot_success_study_all,
    run_tightness_study,
)
from _scripts.run_time_study import run_time_study

DEBUG = False

if __name__ == "__main__":
    n_seeds = 3 if DEBUG else 10
    results_dir = "_results_sparsity_server"
    overwrite = False

    print("========== running exampels ===========")
    # run_example(results_dir, "exampleRO")
    # run_example(results_dir, "exampleMW")

    print("========== running tightness study ===========")
    # run_tightness_study(
    #     results_dir,
    #     overwrite=overwrite,
    #     n_seeds=n_seeds,
    #     appendix="noisetest" if DEBUG else "noise",
    # )
    print("========== plotting accuracy study ===========")
    # plot_success_study_all(results_dir, appendix="noisetest" if DEBUG else "noise")
    plot_accuracy_study_all(results_dir, appendix="noisetest" if DEBUG else "noise")

    # print("========== running timing study ===========")
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
