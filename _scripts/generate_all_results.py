from _scripts.run_all_study import run_all as run_all_study
from _scripts.run_stereo_study import run_all as run_stereo_study
from _scripts.run_range_only_study import run_all as run_range_only_study
from _scripts.run_other_study import run_all as run_other_study
from _scripts.run_datasets_stereo import run_all as run_datasets_stereo
from _scripts.run_datasets_ro import run_all as run_datasets_ro


import matplotlib

matplotlib.use("Agg")

if __name__ == "__main__":
    import sys

    run_datasets_ro(recompute=False, n_successful=100)
    run_datasets_stereo(recompute=False, n_successful=100)

    sys.exit()

    run_datasets_ro(recompute=True, n_successful=10)
    run_datasets_stereo(recompute=True, n_successful=10)

    n_seeds = 10
    recompute = False
    tightness = False
    scalability = True

    run_all_study(recompute=recompute)
    run_other_study(
        n_seeds=n_seeds,
        recompute=recompute,
        tightness=tightness,
        scalability=scalability,
    )
    run_range_only_study(
        n_seeds=n_seeds,
        recompute=recompute,
        tightness=tightness,
        scalability=scalability,
    )
    run_stereo_study(
        n_seeds=n_seeds,
        recompute=recompute,
        tightness=tightness,
        scalability=scalability,
    )
