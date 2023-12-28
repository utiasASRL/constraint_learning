import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from auto_template.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from solvers.chordal import investigate_sparsity, get_aggregate_sparsity
from utils.plotting_tools import plot_aggregate_sparsity, savefig

COMPUTE_MINIMAL = False


def plot_sparsities(learner: Learner):
    A_b_known = [(constraint.A_sparse_, 0) for constraint in learner.templates_known]
    A_b_all = learner.get_A_b_list()
    if COMPUTE_MINIMAL:
        indices = learner.generate_minimal_subset(
            reorder=True,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )
        if indices is None:
            print(f"{lifter}: did not find valid lamdas tightness.")
        A_b_suff = [A_b_all[i] for i in indices]
    else:
        A_b_suff = None

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
        print("wrote")
        fig, ax = investigate_sparsity(mask)
        ax.set_title(title)
        savefig(fig, f"_plots/{title}_cliques.png", verbose=True)
        fig, ax = plot_aggregate_sparsity(mask)
        ax.set_title(title)
        savefig(fig, f"_plots/{title}_mask.png", verbose=True)


def run_by_clique(lifter: Stereo2DLifter):
    from cert_tools.sparse_solvers import solve_oneshot
    from cert_tools.base_clique import BaseClique

    clique_vars = [("h", "x", f"z_{i}") for i in range(lifter.n_landmarks)]
    clique_list = []
    Q = lifter.get_Q()
    for vars in clique_vars:

        # get Q
        # TODO(FD) continue here, need to divide Q in overlapping areas! 
        Q_sub = # 

        # get A_list
        A_b_list = lifter.get_A_b_learned(var_dict=vars)

        # generate overlap
        clique = BaseClique(
            Q, [A_b[0] for A_b in A_b_list], [A_b[1] for A_b in A_b_list]
        )
        clique.left_slice_start = [[0, 0]]
        clique.left_slice_end = [[1 + lifter.get_x_dim(), 1 + lifter.get_x_dim()]]
        clique.right_slice_start = [[0, 0]]
        clique.right_slice_end = [[1 + lifter.get_x_dim(), 1 + lifter.get_x_dim()]]
        clique_list.append(clique)

    X_list, info = solve_oneshot(clique_list, primal=False)
    return X_list

if __name__ == "__main__":
    np.random.seed(0)
    # lifter = Stereo2DLifter(n_landmarks=3, param_level="ppT", level="urT")
    lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="no")
    # (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="quad")),
    # (Stereo3DLifter, dict(n_landmarks=4, param_level="ppT", level="urT")),
    # (WahbaLifter, dict(n_landmarks=5, d=3, robust=True, level="xwT", n_outliers=1)),
    # (MonoLifter, dict(n_landmarks=6, d=3, robust=True, level="xwT", n_outliers=1)),
    # (WahbaLifter, dict(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0)),
    # (MonoLifter, dict(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0)),

    import scipy.sparse as sp

    name = "known"

    try:
        title = f"{lifter}_{name}"
        df = pd.read_pickle(f"_results/{title}_mask.pkl")
        mask = sp.csr_array(
            (np.ones(len(df.indices_i)), [df.indices_i, df.indices_j]),
            shape=df.attrs["mask_shape"],
        )
        plt.matshow(mask.toarray())
        print("done")

    except FileNotFoundError:
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        dict_list, success = learner.run(verbose=True, plot=False)
        if not success:
            raise RuntimeError(
                f"{lifter}: did not achieve {learner.lifter.TIGHTNESS} tightness."
            )
        plot_sparsities(learner)
