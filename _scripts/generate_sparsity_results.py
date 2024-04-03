from _scripts.run_clique_study import run_hierarchy_study
from _scripts.run_examples import run_example
from _scripts.run_tightness_study import run_tightness_study
from _scripts.run_time_study import run_time_study

if __name__ == "__main__":
    n_seeds = 1
    results_dir = "_results_new"
    overwrite = True

    print("========== running exampels ===========")
    run_example(results_dir, "exampleRO")
    run_example(results_dir, "exampleMW")

    print("========== running tightness study ===========")
    run_tightness_study(
        results_dir, overwrite=overwrite, n_seeds=n_seeds, appendix="noisetest"
    )
    print("========== running timing study ===========")
    run_time_study(
        results_dir, overwrite=overwrite, n_seeds=n_seeds, appendix="timetest"
    )
    print("========== running hierarhcy study ===========")
    run_hierarchy_study(
        results_dir, overwrite=overwrite, n_seeds=n_seeds, appendix="hierarchytest"
    )