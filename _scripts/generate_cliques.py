import itertools

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp

from cert_tools.base_clique import BaseClique
from poly_matrix import PolyMatrix

from lifters.stereo_lifter import StereoLifter
from lifters.stereo3d_lifter import Stereo3DLifter

DEBUG = True
USE_KNOWN = True


def create_clique_list(
    lifter: Stereo3DLifter, overlap_mode=0, use_known=USE_KNOWN, verbose=False
):
    """
    :param overlap_mode:
        - 0: no overlap
        - 1: add adjacent landmarks as overlapping
        - 2: add all possible pairs of landmarks as overlapping

    """

    def create_clique(var_dict, Q_sub, A_list, X_sub):
        if recreate_A_list or (len(A_list) == 0):
            if use_known:
                A_known_poly = lifter.get_A_known(var_dict=var_dict, output_poly=True)
                A_known = [A.get_matrix(var_dict) for A in A_known_poly]
                A_learned = lifter.get_A_learned_simple(
                    var_dict=var_dict, A_known=A_known_poly
                )
                A_list = [lifter.get_A0(var_dict)] + A_known + A_learned
            else:
                A_learned = lifter.get_A_learned_simple(var_dict=var_dict)
                A_list = [lifter.get_A0(var_dict)] + A_learned
            if verbose:
                print(
                    f"number of total constraints: known {len(A_list) - len(A_learned)}"
                    + f" + learned {len(A_learned)} = {len(A_list)}"
                )

        b_list = [1.0] + [0] * (len(A_list) - 1)
        clique = BaseClique(
            sp.csr_array(Q_sub), A_list, b_list, var_dict=var_dict, X=X_sub
        )
        return clique, A_list

    def evaluate_clique(clique, vars):
        x = lifter.get_x(var_subset=vars)
        for A, b in zip(clique.A_list, clique.b_list):
            err = abs(x.T @ A @ x - b)
            assert err < 1e-6, err
        return x.T @ clique.Q @ x

    def create_Q(Q_list, factors):
        # each element Qi of Q_list corresponds to one clique: {"h", "x", "z_i"}
        Q_sub = np.zeros((o + len(factors) * z, o + len(factors) * z))
        for k, (Q, fac) in enumerate(zip(Q_list, factors)):
            Q_sub[:o, :o] += fac * Q[:o, :o]  # Q_0
            Q_sub[:o, o + k * z : o + (k + 1) * z] += fac * Q[:o, o : o + z]  # q_0
            Q_sub[o + k * z : o + (k + 1) * z, :o] += fac * Q[:o, o : o + z].T  # q_0
            Q_sub[o + k * z : o + (k + 1) * z, o + k * z : o + (k + 1) * z] += (
                fac * Q[o : o + z, o : o + z]
            )  # p_0
        return Q_sub

    recreate_A_list = isinstance(lifter, StereoLifter)
    o = lifter.base_size()
    z = lifter.landmark_size()

    Q_subs = []
    clique_vars = []
    clique_list = []
    cost_total = 0

    Q_list = []
    for i in range(lifter.n_landmarks):
        vars = lifter.get_clique_vars(i, n_overlap=0)
        Q, __ = lifter.get_Q(output_poly=True, use_cliques=[i])
        Q_list.append(Q.get_matrix(vars))

    if DEBUG:
        Q_test = PolyMatrix()

    o = lifter.base_size()
    z = lifter.landmark_size()
    if overlap_mode == 0:
        for i in range(lifter.n_landmarks):
            vars = lifter.get_clique_vars(i, n_overlap=overlap_mode)
            Q_sub = Q_list[i]
            Q_subs.append(Q_sub)
            clique_vars.append(vars)
    elif overlap_mode > 0:
        if overlap_mode == 1:
            tuples = zip(range(lifter.n_landmarks - 1), range(1, lifter.n_landmarks))
        else:
            tuples = itertools.combinations(range(lifter.n_landmarks), overlap_mode)

        for tuple_ in tuples:
            vars = lifter.get_clique_vars_ij(*tuple_)

            # factors represents how many times each variable group is represented.
            if overlap_mode == 1:
                faci = 1.0 if tuple_[0] == 0 else 0.5
                facj = 1.0 if tuple_[-1] == lifter.n_landmarks - 1 else 0.5
                factors = [faci, facj]
            elif overlap_mode == 2:
                factors = [1 / (lifter.n_landmarks - 1)] * len(tuple_)
            elif overlap_mode == 3:
                factors = [
                    1 / ((lifter.n_landmarks - 1) * (lifter.n_landmarks - 2) / 2)
                ] * len(tuple_)

            Q_sub = create_Q([Q_list[t] for t in tuple_], factors)
            Q_subs.append(Q_sub)
            clique_vars.append(vars)

            if DEBUG:
                if verbose:
                    print([f"{faci:.2f} * Q{t}" for t, faci in zip(tuple_, factors)])
                for t, faci in zip(tuple_, factors):
                    Qi = Q_list[t].toarray()
                    Q_test["hx", "hx"] += faci * Qi[:o, :o]
                    Q_test["hx", f"q_{t}"] += faci * Qi[:o, o : o + z]
                    Q_test[f"q_{t}", f"q_{t}"] += faci * Qi[o : o + z, o : o + z]

    A_list = []
    cost_total = 0
    for i in range(len(Q_subs)):
        vars = clique_vars[i]
        x = lifter.get_x(var_subset=vars)
        X_sub = np.outer(x, x)

        clique, A_list = create_clique(vars, Q_subs[i], A_list, X_sub)
        clique.index = i
        clique_list.append(clique)
        if DEBUG:
            cost_total += evaluate_clique(clique, vars)

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

    return clique_list
