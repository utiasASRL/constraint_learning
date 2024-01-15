import itertools
import pickle

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cert_tools.base_clique import BaseClique
from cert_tools.sparse_solvers import solve_oneshot

from auto_template.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.stereo_lifter import StereoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.state_lifter import StateLifter
from solvers.chordal import get_aggregate_sparsity, investigate_sparsity
from utils.plotting_tools import plot_aggregate_sparsity, savefig

COMPUTE_MINIMAL = False

NOISE_SEED = 5

USE_FUSION = False
USE_PRIMAL = True
USE_KNOWN = False

VERBOSE = False


def compute_sparsities(learner: Learner, appendix=""):
    templates_known = learner.get_known_templates(use_known=True)
    A_b_known = [(constraint.A_sparse_, 0) for constraint in templates_known]
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
        fname = f"_results/{title}_mask{appendix}.pkl"
        df.to_pickle(fname)
        print("wrote", fname)


def visualize_clique_list(clique_list, symmetric=True, fname=""):
    from poly_matrix import PolyMatrix

    mask = PolyMatrix(symmetric=symmetric)
    Q_vals = PolyMatrix(symmetric=symmetric)
    A_vals = PolyMatrix(symmetric=symmetric)
    for c in clique_list:
        assert isinstance(c, BaseClique)
        for key_i, key_j in itertools.combinations_with_replacement(c.var_dict, 2):
            Q_ij = c.get_Qij(key_i, key_j)
            A_ij = c.get_Aij_agg(key_i, key_j).astype(float)

            Q_vals[key_i, key_j] += Q_ij
            A_vals[key_i, key_j] += A_ij

            mask[key_i, key_j] += 1.0

    # fig, ax, im = Q_vals.matshow()
    # ax.set_title("Q")
    # fig, ax, im = A_vals.matshow()
    # ax.set_title("A agg")
    for X, title in zip([Q_vals, A_vals], ["Q", "A"]):
        fig, ax = plt.subplots()
        plot_X = np.abs(X.get_matrix(X.variable_dict_i).toarray())
        ax.matshow(np.log10(plot_X))
        ax.set_title(title)
        for i, c in enumerate(clique_list):
            X.plot_box(
                ax=ax, clique_keys=c.var_dict.keys(), symmetric=symmetric, color=f"C{i}"
            )
        if fname != "":
            savefig(fig, fname + f"_{title}.png")

    fig, ax, im = mask.matshow()
    ax.set_title("clique mask")
    if fname != "":
        savefig(fig, fname + f"_mask.png")


def solve_by_cliques(lifter: StateLifter, overlap_params, fname=""):
    # from _scripts.run_by_cliques_bkp import run_by_clique
    from _scripts.generate_cliques import create_clique_list, create_clique_list_slam

    np.random.seed(NOISE_SEED)
    Q, y = lifter.get_Q()
    theta_gt = lifter.get_vec_around_gt(delta=0)
    theta_est, info, cost = lifter.local_solver(theta_gt, lifter.y_)
    # xtheta_gt = lifter.get_x()
    # theta_est, info, cost = lifter.local_solver(xtheta_gt, lifter.y_)
    x = lifter.get_x(theta=theta_est)
    assert (x.T @ Q @ x - cost) / cost < 1e-7
    for overlap in overlap_params:
        print("creating cliques...", end="")
        # clique_list = create_clique_list(lifter, **overlap, use_known=USE_KNOWN)
        clique_list = create_clique_list_slam(lifter, use_known=USE_KNOWN)
        # visualize_clique_list(
        #    clique_list,
        #    fname=fname + f"_o{overlap['overlap_mode']:.0f}_c{overlap['n_vars']:.0f}",
        # )
        print("solving...", end="")
        X_list, info = solve_oneshot(
            clique_list, use_primal=USE_PRIMAL, use_fusion=USE_FUSION, verbose=VERBOSE
        )
        print(f"results for {overlap} : q={cost:.4e}, p={info['cost']:.4e}")


def read_saved_learner(lifter):
    fname = f"_results/scalability_{lifter}_order_dict.pkl"
    try:
        with open(fname, "rb") as f:
            order_dict = pickle.load(f)
            saved_learner = pickle.load(f)
    except FileNotFoundError:
        print(f"did not find saved learner for {lifter}. Run run_..._study.py first.")
        raise
    return saved_learner


def solve_in_one(lifter, overlap_params):
    # all_pairs = True
    # appendix = "_all"

    all_pairs = False
    appendix = ""

    try:
        names = [""]  # recompute
        # names = ["known", "learned", "suff"]
        for name in names:
            title = f"{lifter}_{name}"
            df = pd.read_pickle(f"_results/{title}_mask{appendix}.pkl")
            mask = sp.csr_array(
                (np.ones(len(df.indices_i)), [df.indices_i, df.indices_j]),
                shape=df.attrs["mask_shape"],
            )
            fig, ax = investigate_sparsity(mask)
            ax.set_title(title)
            savefig(fig, f"_plots/{title}_cliques{appendix}.png", verbose=True)
            fig, ax = plot_aggregate_sparsity(mask)
            ax.set_title(title)
            savefig(fig, f"_plots/{title}_mask{appendix}.png", verbose=True)

    except FileNotFoundError:
        # uses incremental learning
        saved_learner = read_saved_learner(lifter)
        overlap_params = overlap_params[0]
        if overlap_params["overlap_mode"] == 2:
            lifter.ALL_PAIRS = True
        elif overlap_params["overlap_mode"] == 1:
            lifter.ALL_PAIRS = False
            lifter.CLIQUE_SIZE = overlap_params["n_vars"]
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        learner.find_local_solution()  # populates solver_vars
        learner.templates = saved_learner.templates
        learner.apply_templates()
        learner.is_tight(verbose=True)
        # compute_sparsities(learner, appendix=appendix)

        # new_constraints = lifter.apply_templates(
        #    saved_learner.templates, var_dict=lifter.var_dict, all_pairs=False
        # )
        # remove_dependent_constraints(new_constraints)
        # A_list = [lifter.get_A0(var_dict)] + [t.A_sparse_ for t in new_constraints]

        # doesn't use incremental learning
        # _, success = learner.run(verbose=False, plot=False)
        # if not success:
        #     raise RuntimeError(
        #         f"{lifter}: did not achieve {learner.lifter.TIGHTNESS} tightness."
        #     )
        # compute_sparsities(learner)


if __name__ == "__main__":
    np.random.seed(0)
    # TODO(FD): Some running points
    # - currently Stereo3D works for seed=0 but fails for others (didn't test extensively)
    # - Mono robust doesn't become tight, but Wahba robust does. This is weird as they usually behave the same!
    lifters = [
        # Stereo2DLifter(n_landmarks=10, param_level="ppT", level="urT"),
        # Stereo3DLifter(n_landmarks=10, param_level="ppT", level="urT"),
        # RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="no"),
        # RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="quad"),
        # WahbaLifter(n_landmarks=10, d=2, robust=True, level="xwT", n_outliers=1),
        # WahbaLifter(n_landmarks=20, d=3, robust=True, level="xwT", n_outliers=14),
        # MonoLifter(n_landmarks=10, d=3, robust=True, level="xwT", n_outliers=1),
        # WahbaLifter(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0),
        # MonoLifter(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0),
        MatWeightLocLifter(n_landmarks=10, n_poses=10)
    ]
    # overlaps = [0, 1, 2]
    overlaps = [0, 1, 2]
    overlap_params = [
        # {"overlap_mode": 0, "n_vars": 1},
        {"overlap_mode": 1, "n_vars": 2},
        # {"overlap_mode": 1, "n_vars": 3},
        # {"overlap_mode": 1, "n_vars": 4},
        # {"overlap_mode": 1, "n_vars": 5},
        # {"overlap_mode": 1, "n_vars": 6},
        # {"overlap_mode": 1, "n_vars": 7},
        # {"overlap_mode": 2, "n_vars": 2},
    ]
    for lifter in lifters:
        print(f"=============={lifter}===============")
        # solve_in_one(lifter, overlap_params)
        solve_by_cliques(
            lifter, overlap_params=overlap_params, fname=f"_plots/{lifter}_cliques"
        )
    print("done")
