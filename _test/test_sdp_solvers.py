from copy import deepcopy

import numpy as np
from cert_tools.admm_solvers import solve_alternating
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp, solve_sdp_homqcqp
from cert_tools.sparse_solvers import solve_dsdp, solve_oneshot

from decomposition.sim_experiments import extract_solution, get_relative_gap
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter, Reg
from poly_matrix import PolyMatrix

TOL_DSDP = 1e-10
TOL_SDP = 1e-10
EVR_THRESH = 1e2
TEST_ERROR = 0.1
TEST_OPT = 1e-3
TEST_RDG = 1e-2
TEST_SDP = 0.1  # because not rank 1, has to be higher

VERBOSE = False


def test_matrices():
    def test_equal(Q_list, Q_list_test):
        for Q, Q_test in zip(Q_list, Q_list_test):
            np.testing.assert_allclose(Q.toarray(), Q_test.toarray())

    n_params = 4
    lifters = [
        MatWeightLocLifter(n_landmarks=8, n_poses=n_params),
        RangeOnlyLocLifter(
            n_landmarks=8,
            n_positions=n_params,
            reg=Reg.CONSTANT_VELOCITY,
            d=2,
            level="no",
        ),
    ]
    noises = [1.0, 1e-4]
    sparsity = 1.0
    seed = 0
    for lifter, noise in zip(lifters, noises):

        np.random.seed(seed)
        lifter.generate_random_setup()
        lifter.simulate_y(noise=noise, sparsity=sparsity)

        Q = lifter.get_Q_from_y(lifter.y_)
        A_list = lifter.get_A_learned_simple()

        problem = HomQCQP.init_from_lifter(lifter)
        test_equal([problem.C.get_matrix(lifter.var_dict)], [Q])
        test_equal([A.get_matrix(lifter.var_dict) for A in problem.As], A_list)

        Q_test, Constraints_list_test = problem.get_problem_matrices()
        A_list_test = [C[0] for C in Constraints_list_test]
        test_equal([Q], [Q_test])
        test_equal(A_list, A_list_test)


def test_highlevel():
    n_params = 4
    lifters = [
        MatWeightLocLifter(n_landmarks=8, n_poses=n_params),
        RangeOnlyLocLifter(
            n_landmarks=8,
            n_positions=n_params,
            reg=Reg.CONSTANT_VELOCITY,
            d=2,
            level="no",
        ),
    ]
    noises = [1.0, 1e-4]
    sparsity = 1.0
    seed = 0
    for lifter, noise in zip(lifters, noises):

        np.random.seed(seed)
        lifter.generate_random_setup()
        lifter.simulate_y(noise=noise, sparsity=sparsity)

        theta_gt = lifter.get_vec_around_gt(delta=0)
        theta_rand = lifter.get_vec_around_gt(delta=0.5)

        ## =================== solve with gt init =================== ##
        theta_opt, info, cost_gt = lifter.local_solver(
            theta_gt, lifter.y_, verbose=VERBOSE, method="GN"  # could be BFGS for RO
        )
        print(f"GN converged in {info['n_iter']} from ground truth, cost: {cost_gt}")
        assert info["success"]
        if not isinstance(lifter, RangeOnlyLocLifter):
            for key, val in lifter.get_error(theta_hat=theta_opt).items():
                assert val < TEST_ERROR, f"error {key} failed for local-gt"

        ## =================== solve with random init =================== ##
        theta_est, info, cost = lifter.local_solver(
            theta_rand, lifter.y_, verbose=VERBOSE
        )
        print(f"GN converged in {info['n_iter']} from random init, cost: {cost}")
        assert info["success"]
        for key, val in lifter.get_error(
            theta_hat=theta_est, theta_gt=theta_opt
        ).items():
            assert val < TEST_OPT, f"error {key} failed for local"

        ## =================== solve with SDP the old way =================== ##
        Q, _ = lifter.get_Q()
        Constraints = [(lifter.get_A0(), 1.0)] + [
            (A, 0.0) for A in lifter.get_A_learned_simple()
        ]
        X, info = solve_sdp(
            Q,
            Constraints,
            adjust=True,
            primal=True,
            use_fusion=True,
            tol=TOL_SDP,
            verbose=VERBOSE,
        )

        rdg = get_relative_gap(info["cost"], cost_gt)
        x_SDP, info_rank = rank_project(X, p=1)
        theta_SDP = lifter.get_theta_from_x(x=x_SDP)
        for key, val in lifter.get_error(
            theta_hat=theta_SDP, theta_gt=theta_opt
        ).items():
            assert val < TEST_SDP, f"error {key} failed for SDP"
        assert info_rank["EVR"] > EVR_THRESH
        print(f"SDP converged to cost={info['cost']}")

        ## =================== solve with SDP from HomQCQP =================== ##
        problem = HomQCQP.init_from_lifter(lifter)
        np.testing.assert_allclose(
            problem.C.get_matrix_dense(lifter.var_dict), Q.toarray()
        )
        problem.get_asg(var_list=lifter.var_dict)
        # clique_data = HomQCQP.get_chain_clique_data(fixed="h", variables=["x", "z"])
        problem.clique_decomposition()

        Q_homqcqp, Constraints_homqcqp = problem.get_problem_matrices()
        X, info = solve_sdp(
            Q_homqcqp,
            Constraints_homqcqp,
            adjust=True,
            primal=True,
            use_fusion=True,
            tol=TOL_SDP,
            verbose=VERBOSE,
        )

        rdg = get_relative_gap(info["cost"], cost_gt)
        x_SDP, info_rank = rank_project(X, p=1)
        theta_SDP = lifter.get_theta_from_x(x=x_SDP)
        for key, val in lifter.get_error(
            theta_hat=theta_SDP, theta_gt=theta_opt
        ).items():
            assert val < TEST_SDP, f"error {key} failed for SDP"
        assert info_rank["EVR"] > EVR_THRESH
        print(f"SDP converged to cost={info['cost']}")

        ## =================== solve with dSDP =================== ##

        X_list, info = solve_dsdp(
            problem,
            use_primal=True,
            verbose=VERBOSE,
            tol=TOL_DSDP,
        )

        (
            x_dSDP,
            ranks,
            factors,
        ) = problem.get_mr_completion(X_list)

        rdg = get_relative_gap(info["cost"], cost_gt)
        assert rdg < TEST_RDG

        theta_dSDP = lifter.get_theta_from_x(x=x_dSDP)
        if not isinstance(lifter, RangeOnlyLocLifter):
            for key, val in lifter.get_error(
                theta_hat=theta_dSDP, theta_gt=theta_opt
            ).items():
                assert val < TEST_OPT, f"error {key} failed for dSDP"
        np.testing.assert_allclose(ranks, 1.0)
        print(f"dSDP converged to cost={info['cost']}")

        return
        ## =================== solve with ADMM =================== ##
        if lifter.ADMM_INIT_XHAT:
            X0 = []
            for c in problem.cliques:
                x_clique = lifter.get_x(theta=theta_est, var_subset=c.var_dict)
                X0.append(np.outer(x_clique, x_clique))
        else:
            X0 = None
        # do deepcopy to make sure we can can use clique_list again.
        lifter.ADMM_OPTIONS["maxiter"] = 30
        lifter.ADMM_OPTIONS["early_stop"] = True
        X_list, info = solve_alternating(
            deepcopy(problem),
            X0=X0,
            verbose=False,
            **lifter.ADMM_OPTIONS,
        )
        rdg = get_relative_gap(info["cost"], cost_gt)
        assert rdg < TEST_RDG

        x_ADMM, evr_mean = extract_solution(lifter, X_list)
        theta_ADMM = lifter.get_theta_from_x(x=np.hstack([1.0, x_ADMM.flatten()]))
        if not isinstance(lifter, RangeOnlyLocLifter):
            for key, val in lifter.get_error(theta_hat=theta_ADMM).items():
                assert val < TEST_ERROR, f"error {key} failed for dSDP-admm"
        assert evr_mean > EVR_THRESH
        print(f"dSDP-admm converged to cost={info['cost']}")


if __name__ == "__main__":
    # test_matrices()
    test_highlevel()
