import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from auto_template.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.stereo_lifter import StereoLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from solvers.chordal import investigate_sparsity, get_aggregate_sparsity
from utils.plotting_tools import plot_aggregate_sparsity, savefig

from cert_tools.sparse_solvers import solve_oneshot

COMPUTE_MINIMAL = True

NOISE_SEED = 5

USE_FUSION = True
USE_PRIMAL = True
USE_KNOWN = False

VERBOSE = True


def plot_sparsities(learner: Learner):
    A_b_known = [(constraint.A_sparse_, 0) for constraint in learner.templates_known]
    A_b_all = learner.get_A_b_list()
    A_b_suff = None
    if COMPUTE_MINIMAL:
        indices = learner.generate_minimal_subset(
            reorder=True,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )
        if indices is None:
            print(f"{lifter}: did not find valid lamdas.")
        else:
            A_b_suff = [A_b_all[i] for i in indices]

    for A_b, name in zip([A_b_known, A_b_all, A_b_suff], ["known", "learned", "suff"]):
        if A_b is None:
            continue

        title = f"{learner.lifter}_{name}"
        mask = get_aggregate_sparsity(learner.solver_vars["Q"], A_b)
        I, J = mask.nonzero()
        df = pd.DataFrame()
        df.loc[:, "indices_i"] = I
        df.loc[:, "indices_j"] = J
        df.attrs["mask_shape"] = mask.shape
        fname = f"_results/{title}_mask.pkl"
        df.to_pickle(fname)
        print("wrote", fname)
        fig, ax = investigate_sparsity(mask)
        ax.set_title(title)
        savefig(fig, f"_plots/{title}_cliques.png", verbose=True)
        fig, ax = plot_aggregate_sparsity(mask)
        ax.set_title(title)
        savefig(fig, f"_plots/{title}_mask.png", verbose=True)


def solve_by_cliques(lifter, overlaps=[0]):
    # from _scripts.run_by_cliques_bkp import run_by_clique
    from _scripts.generate_cliques import create_clique_list

    np.random.seed(NOISE_SEED)
    Q, y = lifter.get_Q()
    theta_gt = lifter.theta
    theta_est, info, cost = lifter.local_solver(theta_gt, lifter.y_)
    # xtheta_gt = lifter.get_x()
    # theta_est, info, cost = lifter.local_solver(xtheta_gt, lifter.y_)
    x = lifter.get_x(theta_est)
    assert (x.T @ Q @ x - cost) / cost < 1e-7
    for overlap in overlaps:
        print("creating cliques...", end="")
        clique_list = create_clique_list(
            lifter, overlap_mode=overlap, use_known=USE_KNOWN
        )
        print("solving...", end="")
        X_list, info = solve_oneshot(
            clique_list, use_primal=USE_PRIMAL, use_fusion=USE_FUSION, verbose=VERBOSE
        )
        print(f"results for overlap {overlap} : q={cost:.4e}, p={info['cost']:.4e}")


def solve_in_one(lifter):
    name = ""  # recompute
    # name = "known"
    # name = "learned"
    # name = "suff"
    try:
        title = f"{lifter}_{name}"
        df = pd.read_pickle(f"_results/{title}_mask.pkl")
        mask = sp.csr_array(
            (np.ones(len(df.indices_i)), [df.indices_i, df.indices_j]),
            shape=df.attrs["mask_shape"],
        )
        plt.matshow(mask.toarray())
        investigate_sparsity(mask)
        print("done")

    except FileNotFoundError:
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        _, success = learner.run(verbose=False, plot=False)
        if not success:
            raise RuntimeError(
                f"{lifter}: did not achieve {learner.lifter.TIGHTNESS} tightness."
            )
        plot_sparsities(learner)


if __name__ == "__main__":
    np.random.seed(0)
    # TODO(FD): Some running points
    # - currently Stereo3D works for seed=0 but fails for others (didn't test extensively)
    # - Mono robust doesn't become tight, but Wahba robust does. This is weird as they usually behave the same!
    lifters = [
        # Stereo2DLifter(n_landmarks=5, param_level="ppT", level="urT"),
        Stereo3DLifter(n_landmarks=4, param_level="ppT", level="urT"),
        # RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="no"),
        # RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="quad"),
        # WahbaLifter(n_landmarks=10, d=2, robust=True, level="xwT", n_outliers=1),
        # WahbaLifter(n_landmarks=10, d=3, robust=True, level="xwT", n_outliers=1),
        # MonoLifter(n_landmarks=6, d=3, robust=True, level="xwT", n_outliers=1),
        # WahbaLifter(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0),
        # MonoLifter(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0),
    ]
    # overlaps = [0, 1, 2]
    overlaps = [0, 1, 2]
    for lifter in lifters:
        print(f"=============={lifter}===============")
        # solve_in_one(lifter)
        solve_by_cliques(lifter, overlaps=overlaps)
    print("done")
