from _scripts.run_all_study import run_all as run_all_study
from _scripts.run_stereo_study import run_all as run_stereo_study
from _scripts.run_range_only_study import run_all as run_range_only_study
from _scripts.run_other_study import run_all as run_other_study

if __name__ == "__main__":
    n_seeds = 1
    recompute = False
    run_stereo_study(n_seeds=n_seeds, recompute=recompute, tightness=True, scalability=False)
    run_range_only_study(n_seeds=n_seeds, recompute=recompute, tightness=True, scalability=False)
    run_other_study(n_seeds=1, recompute=recompute, tightness=True, scalability=False)
    run_all_study(recompute=True)