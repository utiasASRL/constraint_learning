import numpy as np

from lifters.learner import Learner
from lifters.range_only_lifters import RangeOnlyLocLifter
from experiments import plot_scalability, save_table
from utils.plotting_tools import savefig

from _scripts.stereo_study import run_oneshot_experiment, run_scalability_new

RECOMPUTE = True

n_positions = 3
n_landmarks = 10
d = 3

def range_only_tightness():
    """
    Find the set of minimal constraints required for tightness for range-only problem.
    """
    seed = 0
    #plots = [] 
    plots = ["svd", "matrices", "matrix", "tightness", "templates"]  # ["svd", "matrices"]

    for level in ["no", "quad"]:
        np.random.seed(seed)
        lifter = RangeOnlyLocLifter(
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            d=d,
            level=level,
            variable_list="all",
        )
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"_results/{lifter}_seed{seed}"
        run_oneshot_experiment(learner, fname_root, plots, tightness="rank", add_original=True, use_known=False)

def range_only_scalability_new():
    n_positions_list = [10, 15, 20, 25, 30]
    n_seeds = 10

    for level in ["no", "quad"]:
        variable_list = None  # use the default one for the first step.
        np.random.seed(0)
        lifter = RangeOnlyLocLifter(
            d=d,
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            level=level,
            variable_list=variable_list,
        )
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
        df = run_scalability_new(learner, param_list=n_positions_list, n_seeds=n_seeds, tightness="rank", recompute=RECOMPUTE, add_original=False)


        df = df[df.type != 'original']
        fname_root = f"_results/scalability_{learner.lifter}"

        fig, axs = plot_scalability(df, log=True, start="t ", legend_idx=1)

        #[ax.set_ylim(10, 1000) for ax in axs.values()]

        fig.set_size_inches(8, 3)
        axs["t solve SDP"].legend(loc="upper right") #, bbox_to_anchor=[1.0, 1.0])
        savefig(fig, fname_root + f"_t.pdf")

        tex_name = fname_root + f"_n.tex"
        save_table(df, tex_name)

if __name__ == "__main__":
    range_only_tightness()
    range_only_scalability_new()