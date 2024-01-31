import itertools
import pickle

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from cert_tools.admm_clique import ADMMClique, initialize_overlap
from progressbar import ProgressBar

from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo_lifter import StereoLifter
from lifters.wahba_lifter import WahbaLifter
from poly_matrix import PolyMatrix
from utils.constraint import remove_dependent_constraints

DEBUG = False
USE_KNOWN = False
USE_AUTOTEMPLATE = False
ADD_REDUNDANT = False


def create_clique_list(
    lifter: WahbaLifter,
    overlap_mode=0,
    n_vars=1,
    use_known=USE_KNOWN,
    use_autotemplate=USE_AUTOTEMPLATE,
    verbose=False,
):
    """
    :param overlap_mode:
        - 0: no overlap
        - 1: add adjacent landmarks as overlapping
        - 2: add all possible tuples of landmarks as overlapping

    :param n_vars: how many landmarks to include per clique.
    """

    def create_Q(Q_list, factors):
        # each element Qi of Q_list corresponds to one clique: {"h", "x", "z_i"}
        Q_sub = np.zeros((o + len(factors) * x_dim, o + len(factors) * x_dim))
        for k, (Q, fac) in enumerate(zip(Q_list, factors)):
            Q_sub[:o, :o] += fac * Q[:o, :o]  # Q_0
            Q_sub[:o, o + k * x_dim : o + (k + 1) * x_dim] += (
                fac * Q[:o, o : o + x_dim]
            )  # q_0
            Q_sub[o + k * x_dim : o + (k + 1) * x_dim, :o] += (
                fac * Q[:o, o : o + x_dim].T
            )  # q_0
            Q_sub[
                o + k * x_dim : o + (k + 1) * x_dim, o + k * x_dim : o + (k + 1) * x_dim
            ] += (
                fac * Q[o : o + x_dim, o : o + x_dim]
            )  # p_0
        return Q_sub

    if use_autotemplate:
        fname = f"_results/scalability_{lifter}_order_dict.pkl"
        try:
            with open(fname, "rb") as f:
                order_dict = pickle.load(f)
                saved_learner = pickle.load(f)
        except FileNotFoundError:
            print(
                f"did not find saved learner for {lifter}. Run run_..._study.py first."
            )
            raise
        templates = saved_learner.templates

    recreate_A_list = isinstance(lifter, StereoLifter)
    o = lifter.base_size()
    x_dim = lifter.node_size()

    Q_subs = []
    clique_vars = []
    clique_list = []
    cost_total = 0

    Q_list = []
    for i in range(lifter.n_poses):
        vars = lifter.get_clique_vars(i, n_overlap=1)

        use_cliques = [f"x_{i}"]
        Q, _ = lifter.get_Q(output_poly=True, use_cliques=use_cliques)
        Q_list.append(Q.get_matrix(vars))

    if DEBUG:
        Q_test = PolyMatrix()

    if overlap_mode == 0:
        if lifter.n_poses % n_vars != 0:
            raise ValueError(
                "Cannot have cliques of different sizes (fusion doesn't like it): n_poses has to be multiple of n_vars"
            )
        for i in np.arange(lifter.n_poses, step=n_vars):
            indices = list(range(i, min(i + n_vars, lifter.n_poses)))
            vars = lifter.get_clique_vars_ij(*indices)
            Q, _ = lifter.get_Q(output_poly=True, use_cliques=indices)
            Q_sub = Q.get_matrix(vars)
            Q_subs.append(Q_sub)
            clique_vars.append(vars)
    elif (overlap_mode > 0) and (n_vars == 1):
        raise ValueError("cannot have overlap with n_vars=1")
    elif (overlap_mode > 0) and (n_vars > 1):
        if overlap_mode == 1:
            # {z_0, z_1, z_2}, {z_1, z_2, z_3}, ... , {z_{N-3}, z_{N-2}, z_{N-1}}
            indices_list = [
                list(range(i, i + n_vars)) for i in range(lifter.n_poses - n_vars + 1)
            ]
        elif overlap_mode == 2:
            # all possible combinations of n_vars variables.
            indices_list = list(itertools.combinations(range(lifter.n_poses), n_vars))
        else:
            raise ValueError("unknown overlap mode, must be 0, 1, or 2")

        # counts how many times each variable group is represented.
        factors = [
            1 / sum(i in idx for idx in indices_list) for i in range(lifter.n_poses)
        ]

        for indices in indices_list:
            vars = lifter.get_clique_vars_ij(*indices)

            Q_sub = create_Q(
                [Q_list[t] for t in indices], [factors[t] for t in indices]
            )
            Q_subs.append(Q_sub)
            clique_vars.append(vars)

            if DEBUG:
                for i in indices:
                    faci = factors[i]
                    if verbose:
                        print(f"{faci:.2f} * Q{i}")
                    Qi = Q_list[i].toarray()
                    Q_test["hx", "hx"] += faci * Qi[:o, :o]
                    Q_test["hx", f"q_{i}"] += faci * Qi[:o, o : o + x_dim]
                    Q_test[f"q_{i}", f"q_{i}"] += (
                        faci * Qi[o : o + x_dim, o : o + x_dim]
                    )

    A_list = []
    cost_total = 0
    for i in range(len(Q_subs)):
        var_dict = clique_vars[i]
        x = lifter.get_x(var_subset=var_dict)
        X_sub = np.outer(x, x)

        if recreate_A_list or (len(A_list) == 0):
            if use_autotemplate:
                new_constraints = lifter.apply_templates(
                    templates, var_dict=var_dict, all_pairs=True
                )
                remove_dependent_constraints(new_constraints)
                A_list = [lifter.get_A0(var_dict)] + [
                    t.A_sparse_ for t in new_constraints
                ]
            elif use_known:
                A_known_poly = lifter.get_A_known(var_dict=var_dict, output_poly=True)
                A_known = [A.get_matrix(var_dict) for A in A_known_poly]
                # A_learned = lifter.get_A_learned_simple(
                #    var_dict=var_dict, A_known=A_known_poly
                # )
                A_list = [lifter.get_A0(var_dict)] + A_known  # + A_learned
            else:
                A_learned = lifter.get_A_learned_simple(var_dict=var_dict)
                A_list = [lifter.get_A0(var_dict)] + A_learned
            if verbose:
                print(
                    f"number of total constraints: known {len(A_list) - len(A_learned)}"
                    + f" + learned {len(A_learned)} = {len(A_list)}"
                )

        # A_agg = np.sum([np.abs(A) > 1e-10 for A in A_list])
        # plt.matshow(A_agg.toarray())
        b_list = [1.0] + [0] * (len(A_list) - 1)
        clique = ADMMClique(
            sp.csr_array(Q_subs[i]),
            A_list,
            b_list,
            var_dict=var_dict,
            X=X_sub,
            index=i,
            x_dim=x_dim,
            hom="h",
        )
        clique_list.append(clique)
        if DEBUG:
            x = lifter.get_x(var_subset=var_dict)
            for A, b in zip(clique.A_list, clique.b_list):
                err = abs(x.T @ A @ x - b)
                assert err < 1e-6, err
            cost_total += x.T @ clique.Q @ x

    if DEBUG:
        Q, y = lifter.get_Q()
        if overlap_mode > 0:
            Q_mat = Q_test.get_matrix(
                ["hx"] + [f"q_{i}" for i in range(lifter.n_poses)]
            )
            # np.testing.assert_allclose(Q_mat.toarray(), Q.toarray())
        x = lifter.get_x()
        cost_test = x.T @ Q @ x
        # assert abs(cost_test - cost_total) < 1e-10, (cost_test, cost_total)

    return clique_list


def create_clique_list_loc(
    lifter: MatWeightLocLifter,
    # lifter: RangeOnlyLocLifter,
    use_known=USE_KNOWN,
    use_autotemplate=USE_AUTOTEMPLATE,
    add_redundant=ADD_REDUNDANT,
    verbose=False,
):
    """
    Simplified clique creation for SLAM. Corresponds to overlap_mode==1, n_vars==2
    """
    if use_autotemplate:
        fname = f"_results/scalability_{lifter}_order_dict.pkl"
        try:
            with open(fname, "rb") as f:
                order_dict = pickle.load(f)
                saved_learner = pickle.load(f)
        except FileNotFoundError:
            print(
                f"did not find saved learner for {lifter}. Run run_..._study.py first."
            )
            raise
        templates = saved_learner.templates

    recreate_A_list = isinstance(lifter, StereoLifter)

    clique_list = []
    cost_total = 0

    if DEBUG:
        Q_test = PolyMatrix(symmetric=True)

    # create the list of variables
    clique_vars = []
    for i in range(lifter.n_cliques):
        vars = lifter.get_clique_vars_ij(i, i + 1)
        clique_vars.append(vars)

    m = len(clique_vars)
    A_list = []
    cost_total = 0
    p = ProgressBar(max_value=m - 1)
    for i in range(m):
        p.update(i)
        var_dict = clique_vars[i]

        if recreate_A_list or (len(A_list) == 0):
            if use_autotemplate:
                new_constraints = lifter.apply_templates(
                    templates, var_dict=var_dict, all_pairs=True
                )
                remove_dependent_constraints(new_constraints)
                A_list = [lifter.get_A0(var_dict)] + [
                    t.A_sparse_ for t in new_constraints
                ]
            elif use_known:
                A_known_poly = lifter.get_A_known(
                    var_dict=var_dict, output_poly=True, add_redundant=add_redundant
                )
                A_known = [A.get_matrix(var_dict) for A in A_known_poly]
                # A_learned = lifter.get_A_learned_simple(
                #    var_dict=var_dict, A_known=A_known_poly
                # )
                A_list = [lifter.get_A0(var_dict)] + A_known  # + A_learned
            else:
                A_learned = lifter.get_A_learned_simple(var_dict=var_dict)
                A_list = [lifter.get_A0(var_dict)] + A_learned
            if verbose:
                print(
                    f"number of total constraints: known {len(A_list) - len(A_learned)}"
                    + f" + learned {len(A_learned)} = {len(A_list)}"
                )
        Qi = lifter.get_clique_cost(i)
        if DEBUG:
            Q_test += Qi
            x = lifter.get_x(var_subset=var_dict)
            X_sub = np.outer(x, x)
        else:
            X_sub = None

        # A_agg = np.sum([np.abs(A) > 1e-10 for A in A_list])
        # plt.matshow(A_agg.toarray())
        b_list = [1.0] + [0] * (len(A_list) - 1)
        Qi = Qi.get_matrix(var_dict)
        x_dim = lifter.node_size()
        clique = ADMMClique(
            Qi,
            A_list,
            b_list,
            var_dict=var_dict,
            X=X_sub,
            index=i,
            x_dim=x_dim,
            hom="h",
            N=lifter.n_cliques + 1,
        )
        clique_list.append(clique)
        if DEBUG:
            x = lifter.get_x(var_subset=var_dict)
            for i, (A, b) in enumerate(zip(clique.A_list, clique.b_list)):
                err = abs(x.T @ A @ x - b)
                assert err < 1e-6, f"constraint {i}: {err}"
            cost_total += x.T @ clique.Q @ x

    initialize_overlap(clique_list)

    if DEBUG:
        Q, y = lifter.get_Q()
        x = lifter.get_x()
        cost_test = x.T @ Q @ x
        Q_mat = Q_test.get_matrix(lifter.get_all_variables()[0])

        np.testing.assert_allclose(Q_mat.toarray(), Q.toarray())
        assert abs(cost_test - cost_total) / cost_total < 1e-5, (cost_test, cost_total)

        for i, clique in enumerate(clique_list):
            assert isinstance(clique, ADMMClique)
            left = clique_list[i - 1].X if i > 0 else None
            right = clique_list[i + 1].X if i < len(clique_list) - 1 else None
            g = clique.generate_g(left=left, right=right)
            error = clique.F @ clique.X.flatten() - g
            np.testing.assert_allclose(error, 0)

    return clique_list
