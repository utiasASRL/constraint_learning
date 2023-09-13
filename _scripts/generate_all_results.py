from _scripts.run_all_study import run_all as run_all_study
from _scripts.run_stereo_study import run_all as run_stereo_study
from _scripts.run_range_only_study import run_all as run_range_only_study
from _scripts.run_other_study import run_all as run_other_study
from _scripts.run_datasets_stereo import run_all as run_datasets_stereo
from _scripts.run_datasets_ro import run_all as run_datasets_ro

import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    pass

if __name__ == "__main__":
    n_seeds = 1 # was 10
    recompute = False
    tightness = True
    scalability = True

    run_all_study(recompute=recompute)
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
    run_other_study(
        n_seeds=n_seeds,
        recompute=recompute,
        tightness=tightness,
        scalability=scalability,
    )



    n_successful = 3 # was 100
    run_datasets_stereo(recompute=False, n_successful=n_successful)
    run_datasets_ro(recompute=False, n_successful=n_successful)

    n_successful = 3 # was 10
    run_datasets_ro(recompute=True, n_successful=n_successful)
    run_datasets_stereo(recompute=True, n_successful=n_successful)
