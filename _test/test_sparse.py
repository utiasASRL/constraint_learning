from copy import deepcopy

import numpy as np
from cert_tools.admm_solvers import solve_alternating
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.sparse_solvers import solve_oneshot

from decomposition.generate_cliques import create_clique_list_loc
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

        ## =================== solve with SDP =================== ##
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

        # test problem without redundant constraints
        clique_list = create_clique_list_loc(
            lifter,
            use_known=True,
            use_autotemplate=False,
            add_redundant=True,
            verbose=VERBOSE,
        )
        Q_test = PolyMatrix(symmetric=False)
        for c in clique_list:
            Ci, __ = PolyMatrix.init_from_sparse(c.Q, var_dict=c.var_dict)
            Q_test += Ci
        np.testing.assert_allclose(Q.toarray(), Q_test.toarray(lifter.var_dict))

        ## =================== solve with dSDP =================== ##
        X_list, info = solve_oneshot(
            clique_list,
            use_primal=True,
            use_fusion=True,
            verbose=VERBOSE,
            tol=TOL_DSDP,
        )
        rdg = get_relative_gap(info["cost"], cost_gt)
        assert rdg < TEST_RDG
        x_dSDP, evr_mean = extract_solution(lifter, X_list)
        theta_dSDP = lifter.get_theta_from_x(x=np.hstack([1.0, x_dSDP.flatten()]))
        if not isinstance(lifter, RangeOnlyLocLifter):
            for key, val in lifter.get_error(
                theta_hat=theta_dSDP, theta_gt=theta_opt
            ).items():
                assert val < TEST_OPT, f"error {key} failed for dSDP"
        assert evr_mean > EVR_THRESH
        print(f"dSDP converged to cost={info['cost']}")

        ## =================== solve with ADMM =================== ##
        if lifter.ADMM_INIT_XHAT:
            X0 = []
            for c in clique_list:
                x_clique = lifter.get_x(theta=theta_est, var_subset=c.var_dict)
                X0.append(np.outer(x_clique, x_clique))
        else:
            X0 = None
        # do deepcopy to make sure we can can use clique_list again.
        lifter.ADMM_OPTIONS["maxiter"] = 30
        lifter.ADMM_OPTIONS["early_stop"] = True
        X_list, info = solve_alternating(
            deepcopy(clique_list),
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
    test_highlevel()
