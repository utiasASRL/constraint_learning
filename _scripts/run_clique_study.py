import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from auto_template.learner import Learner
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from solvers.chordal import investigate_sparsity, get_aggregate_sparsity
from utils.plotting_tools import plot_aggregate_sparsity, savefig

from cert_tools.sparse_solvers import solve_oneshot
from cert_tools.base_clique import BaseClique

from poly_matrix import PolyMatrix

COMPUTE_MINIMAL = False
DEBUG = True

USE_KNOWN = True

NOISE_SEED = 5


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


def run_by_clique(lifter: WahbaLifter, overlap_mode=0):
    """
    :param overlap_mode:
        - 0: no overlap
        - 1: add adjacent landmarks as overlapping
        - 2: add all possible pairs of landmarks as overlapping

    """

    def create_clique(vars, Q_sub, A_list):
        if len(A_list) == 0:
            if USE_KNOWN:
                A_known_poly = lifter.get_A_known(var_dict=vars, output_poly=True)
                A_known = [A.get_matrix(vars) for A in A_known_poly]
                A_learned = lifter.get_A_learned_simple(
                    var_dict=vars, A_known=A_known_poly
                )
                A_list = [lifter.get_A0(vars)] + A_known + A_learned
            else:
                A_learned = lifter.get_A_learned(var_dict=vars)
                A_list = [lifter.get_A0(vars)] + A_learned
            print(
                f"number of total constraints: known {len(A_list) - len(A_learned)}"
                + f" + learned {len(A_learned)} = {len(A_list)}"
            )

        b_list = [1.0] + [0] * (len(A_list) - 1)

        indices = np.unique([int(v.split("_")[1]) for v in vars if "_" in v])
        left_slice_start = [[0, 0]]
        left_slice_end = [[1, o]]
        right_slice_start = [[0, 0]]
        right_slice_end = [[1, o]]
        # left_slice_start = [[0, 0]]
        # left_slice_end = [[o, o]]
        # right_slice_start = [[0, 0]]
        # right_slice_end = [[o, o]]

        # find if there are any neighboring cliques.
        n_elems = z  # lifter.d
        if np.any((indices[:, None] - indices[None, :]) == 1):
            print("constraining overlap of", indices)
            # the right part of left slice...
            left_slice_start += [[0, o + z]]
            left_slice_end += [[1, o + z + n_elems]]
            # equals the left part of right slice...
            right_slice_start += [[0, o]]
            right_slice_end += [[1, o + n_elems]]

        return (
            BaseClique(
                sp.csr_array(Q_sub),
                A_list,
                b_list,
                left_slice_start,
                left_slice_end,
                right_slice_start,
                right_slice_end,
            ),
            A_list,
        )

    def evaluate_clique(clique, vars):
        x = lifter.get_x(var_subset=vars)
        for A, b in zip(clique.A_list, clique.b_list):
            err = abs(x.T @ A @ x - b)
            assert err < 1e-10, err
        return x.T @ clique.Q @ x

    A_list = []
    clique_list = []
    cost_total = 0

    Q_list = []
    for i in range(lifter.n_landmarks):
        vars = lifter.get_clique_vars(i, n_overlap=0)
        Q, __ = lifter.get_Q(output_poly=True, use_cliques=[i])
        Q_list.append(Q.get_matrix(vars))

    if DEBUG:
        X_list = []
        for i in range(lifter.n_landmarks - 1):
            vars = lifter.get_clique_vars(i, n_overlap=overlap_mode)
            x = lifter.get_x(var_subset=vars)
            X_list.append(np.outer(x, x))

    if DEBUG:
        Q_test = PolyMatrix()

    o = lifter.base_size()
    z = lifter.landmark_size()
    if overlap_mode == 0:
        for i in range(lifter.n_landmarks - overlap_mode):
            vars = lifter.get_clique_vars(i, n_overlap=overlap_mode)
            Q_sub = Q_list[i]
            clique, A_list = create_clique(vars, Q_sub, A_list)
            clique_list.append(clique)
            cost_total += evaluate_clique(clique, vars)

    elif overlap_mode == 1:
        for i in range(lifter.n_landmarks - overlap_mode):
            vars = lifter.get_clique_vars(i, n_overlap=overlap_mode)
            Q_sub = np.zeros((o + 2 * z, o + 2 * z))
            fac0 = 1.0 if i == 0 else 0.5
            fac1 = 1.0 if i == lifter.n_landmarks - 2 else 0.5

            # fmt: off
            Q_sub[:o, :o] += fac0 * Q_list[i][:o, :o]  # Q_0
            Q_sub[:o, o:o+z] += fac0 * Q_list[i][:o, o:o+z]  # q_0
            Q_sub[o:o+z, :o] += fac0 * Q_list[i][o:o+z, :o]  # q_0.T
            Q_sub[o:o+z, o:o+z] += fac0 * Q_list[i][o:o+z, o:o+z]  # p_0
            Q_sub[:o, :o] += fac1 * Q_list[i+1][:o, :o] # Q_1
            Q_sub[:o, o+z:o+2*z] +=  fac1 * Q_list[i+1][:o, o:o+z] # q_1
            Q_sub[o+z:o+2*z, :o] +=  fac1 * Q_list[i+1][o:o+z, :o] # q_1.T
            Q_sub[o+z:o+2*z, o+z:o+2*z] += fac1 * Q_list[i+1][o:o+z, o:o+z] # p_1
            # fmt: on

            clique, A_list = create_clique(vars, Q_sub, A_list)
            clique_list.append(clique)
            cost_total += evaluate_clique(clique, vars)

            if DEBUG:
                print(f"{fac0} * Q{i}, {fac1} * Q{i+1}")
                if i < lifter.n_landmarks - 2:
                    Xi = X_list[i]
                    Xii = X_list[i + 1]
                    for l_s, l_e, r_s, r_e in zip(
                        clique.left_slice_start,
                        clique.left_slice_end,
                        clique.right_slice_start,
                        clique.right_slice_end,
                    ):
                        np.testing.assert_allclose(
                            Xi[l_s[0] : l_e[0], l_s[1] : l_e[1]],
                            Xii[r_s[0] : r_e[0], r_s[1] : r_e[1]],
                        )

                Qi = Q_list[i].toarray()
                Qii = Q_list[i + 1].toarray()
                Q_test["hx", "hx"] += fac0 * Qi[:o, :o]
                Q_test["hx", f"q_{i}"] += fac0 * Qi[:o, o : o + z]
                Q_test[f"q_{i}", f"q_{i}"] += fac0 * Qi[o : o + z, o : o + z]
                Q_test["hx", "hx"] += fac1 * Qii[:o, :o]
                Q_test["hx", f"q_{i+1}"] += fac1 * Qii[:o, o : o + z]
                Q_test[f"q_{i+1}", f"q_{i+1}"] += fac1 * Qii[o : o + z, o : o + z]

    elif overlap_mode == 2:
        for i in range(lifter.n_landmarks - 1):
            fac0 = fac1 = 1 / (lifter.n_landmarks - 1)
            left = Q_list[i]
            for j in range(i + 1, lifter.n_landmarks):
                right = Q_list[j]
                vars = lifter.get_clique_vars_ij(i, j)

                Q_sub = np.zeros((o + 2 * z, o + 2 * z))
                # fmt: off
                Q_sub[:o, :o] += fac0 * left[:o, :o]  # Q_0
                Q_sub[:o, o:o+z] += fac0 * left[:o, o:o+z]  # q_0
                Q_sub[o:o+z, :o] += fac0 * left[o:o+z, :o]  # q_0.T
                Q_sub[o:o+z, o:o+z] += fac0 * left[o:o+z, o:o+z]  # p_0
                Q_sub[:o, :o] += fac1 * right[:o, :o] # Q_1
                Q_sub[:o, o+z:o+2*z] +=  fac1 * right[:o, o:o+z] # q_1
                Q_sub[o+z:o+2*z, :o] +=  fac1 * right[o:o+z, :o] # q_1.T
                Q_sub[o+z:o+2*z, o+z:o+2*z] += fac1 * right[o:o+z, o:o+z] # p_1
                # fmt: on

                clique, A_list = create_clique(vars, Q_sub, A_list)
                clique_list.append(clique)
                cost_total += evaluate_clique(clique, vars)

                if DEBUG:
                    print(f"{fac0} * Q{i}, {fac1} * Q{j}")
                    if i < lifter.n_landmarks - 2:
                        Xi = X_list[i]
                        Xii = X_list[i + 1]
                        for l_s, l_e, r_s, r_e in zip(
                            clique.left_slice_start,
                            clique.left_slice_end,
                            clique.right_slice_start,
                            clique.right_slice_end,
                        ):
                            np.testing.assert_allclose(
                                Xi[l_s[0] : l_e[0], l_s[1] : l_e[1]],
                                Xii[r_s[0] : r_e[0], r_s[1] : r_e[1]],
                            )

                    Qi = Q_list[i].toarray()
                    Qii = Q_list[j].toarray()
                    Q_test["hx", "hx"] += fac0 * Qi[:o, :o]
                    Q_test["hx", f"q_{i}"] += fac0 * Qi[:o, o : o + z]
                    Q_test[f"q_{i}", f"q_{i}"] += fac0 * Qi[o : o + z, o : o + z]
                    Q_test["hx", "hx"] += fac1 * Qii[:o, :o]
                    Q_test["hx", f"q_{j}"] += fac1 * Qii[:o, o : o + z]
                    Q_test[f"q_{j}", f"q_{j}"] += fac1 * Qii[o : o + z, o : o + z]

    if DEBUG:
        Q, y = lifter.get_Q()
        if overlap_mode > 0:
            Q_mat = Q_test.get_matrix(
                ["hx"] + [f"q_{i}" for i in range(lifter.n_landmarks)]
            )
            np.testing.assert_allclose(Q_mat.toarray(), Q.toarray())
        x = lifter.get_x()
        cost_test = x.T @ Q @ x
        assert abs(cost_test - cost_total) < 1e-10, (cost_test, cost_total)

    X_list, info = solve_oneshot(
        clique_list, use_primal=True, use_fusion=True, verbose=True
    )
    return X_list, info


if __name__ == "__main__":
    import sys

    np.random.seed(0)
    # lifter = Stereo2DLifter(n_landmarks=5, param_level="ppT", level="urT")
    # lifter = Stereo3DLifter(n_landmarks=4, param_level="ppT", level="urT")
    # lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="no")
    # lifter = RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="quad")
    lifter = WahbaLifter(n_landmarks=5, d=3, robust=True, level="xwT", n_outliers=1)
    # lifter = MonoLifter(n_landmarks=6, d=3, robust=True, level="xwT", n_outliers=1)
    # lifter = WahbaLifter(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0)
    # lifter = MonoLifter(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0)

    np.random.seed(NOISE_SEED)
    Q, y = lifter.get_Q()
    theta_gt = lifter.theta
    theta_est, info, cost = lifter.local_solver(theta_gt, lifter.y_)
    # xtheta_gt = lifter.get_x()
    # theta_est, info, cost = lifter.local_solver(xtheta_gt, lifter.y_)
    x = lifter.get_x(theta_est)
    print(x.T @ Q @ x, cost)
    for overlap in [0, 1, 2]:
        X_list, info_sdp = run_by_clique(lifter, overlap_mode=overlap)
        print(f"overlap {overlap} : q={cost:.4e}, p={info_sdp['cost']:.4e}")
    sys.exit()

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
