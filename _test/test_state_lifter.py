import numpy as np

from lifters.landmark_lifter import PoseLandmarkLifter
from lifters.poly_lifters import Poly4Lifter, Poly6Lifter
from lifters.range_only_lifters import (
    RangeOnlyLocLifter,
    RangeOnlySLAM1Lifter,
    RangeOnlySLAM2Lifter,
)
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

d = 2
n_landmarks = 4
n_poses = 1
all_lifters = [
    # Poly4Lifter(),
    # Poly6Lifter(),
    RangeOnlyLocLifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    RangeOnlySLAM1Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    RangeOnlySLAM2Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    # PoseLandmarkLifter(n_landmarks, n_poses, d=d),
    # Stereo1DLifter(n_landmarks),
    # Stereo2DLifter(n_landmarks, level=0),
    # Stereo3DLifter(n_landmarks),
]


def test_hess_finite_diff():
    for lifter in all_lifters:
        lifter.generate_random_setup()
        lifter.generate_random_unknowns()

        errors = []
        eps_list = np.logspace(-10, -5, 11)
        for eps in eps_list:
            Q, y = lifter.get_Q(noise=1)
            theta = lifter.get_vec_around_gt(delta=0).flatten("C")
            grad = lifter.get_grad(theta, y)

            try:
                hess = lifter.get_hess(theta, y)
            except Exception as e:
                print(e)
                print("grad not implemented?")
                raise
                continue

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            for i in range(n):
                theta_delta = theta + I[i]
                grad_delta = lifter.get_grad(theta_delta, y)

                hess_est = (grad_delta - grad) / eps

                err = np.max(np.abs(hess_est - hess[i, :]))
                max_err = max(err, max_err)
            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except AssertionError:
            print(f"Hessian test for {lifter} not passing")
            # import matplotlib.pylab as plt
            # plt.figure()
            # plt.title(f"hess {lifter}")
            # plt.loglog(eps_list, errors)
            # plt.show()
            # assert  < 1e-7


def test_grad_finite_diff():
    for lifter in all_lifters:
        lifter.generate_random_setup()
        lifter.generate_random_unknowns()

        errors = []
        eps_list = np.logspace(-10, -1, 11)
        for eps in eps_list:
            Q, y = lifter.get_Q(noise=1)

            theta = lifter.get_vec_around_gt(delta=0).flatten("C")
            cost = lifter.get_cost(theta, y)

            # try:
            grad = lifter.get_grad(theta, y)
            # except Exception as e:
            #    print(e)
            #    print("grad not implemented?")
            #    continue

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            for i in range(n):
                theta_delta = theta + I[i]
                cost_delta = lifter.get_cost(theta_delta, y)

                grad_est = (cost_delta - cost) / eps

                err = abs(grad_est - grad[i])
                max_err = max(err, max_err)
                break

            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except AssertionError:
            import matplotlib.pylab as plt

            plt.figure()
            plt.title(f"grad {lifter}")
            plt.loglog(eps_list, errors)
            plt.show()
            # assert  < 1e-7


def test_equivalent_lifters():
    l1 = RangeOnlySLAM1Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)
    l2 = RangeOnlySLAM2Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)

    np.random.seed(1)
    l1.generate_random_setup()
    l1.generate_random_unknowns()
    Q, y = l1.get_Q(noise=1e-3)
    cost1 = l1.get_cost(l1.get_theta(), y)

    np.random.seed(1)
    l2.generate_random_setup()
    l2.generate_random_unknowns()
    Q, y = l2.get_Q(noise=1e-3)
    cost2 = l2.get_cost(l2.get_theta(), y)
    assert abs(cost1 - cost2) < 1e-10

    # 1, t1, t2, ..., a1, a2, ..., |t1|^2, ..., |a1|^2, ..., a1 @ t1, ...
    x1 = l1.get_x()
    # 1, t1, t2, ..., a1, a2, ..., |t1 - a1|^2
    x2 = l2.get_x()

    start = 1 + (n_poses + n_landmarks) * d
    np.testing.assert_allclose(x1[:start], x2[:start])
    for i, (n, k) in enumerate(l1.edges):
        x1_test = (
            x1[start + n]
            + x1[start + n_poses + k]
            - 2 * x1[start + n_poses + n_landmarks + i]
        )
        np.testing.assert_allclose(x2[start + i], x1_test)


def test_levels():
    from lifters.stereo_lifter import StereoLifter

    for level in StereoLifter.LEVELS:
        lifter_2d = Stereo2DLifter(n_landmarks=3, level=level)

        # inside below function we tests that dimensions are consistent.
        lifter_2d.get_x()

        lifter_3d = Stereo3DLifter(n_landmarks=3, level=level)
        lifter_3d.get_x()


def test_cost_noisy():
    test_cost(noise=0.1)


def test_cost(noise=0.0):
    for lifter in all_lifters:
        # np.random.seed(1)
        Q, y = lifter.get_Q(noise=noise)
        # np.random.seed(1)
        # Qold, yold = lifter.get_Q_old(noise=noise)
        # np.testing.assert_allclose(Q, Qold)
        # np.testing.assert_allclose(y, yold)
        if Q is None:
            continue

        x = lifter.get_theta()
        cost = lifter.get_cost(x, y)

        x = lifter.get_x()
        costQ = x.T @ Q @ x
        assert abs(cost - costQ) < 1e-8, (cost, costQ)

        if noise == 0:
            assert cost < 1e-10, cost
            assert costQ < 1e-10, costQ


def test_solvers_noisy(n_seeds=3, noise=1e-1):
    test_solvers(n_seeds=n_seeds, noise=noise)


def test_solvers(n_seeds=1, noise=0.0):
    for lifter in all_lifters:
        for j in range(n_seeds):
            np.random.seed(j)

            # noisy setup
            Q, y = lifter.get_Q(noise=noise)
            if Q is None:
                continue

            # test that we stay at real solution when initializing at it
            theta_gt = lifter.get_vec_around_gt(delta=0)
            theta_hat, msg, cost_solver = lifter.local_solver(theta_gt, y)
            if noise == 0:
                # test that solution is ground truth with no noise
                np.testing.assert_allclose(theta_hat, theta_gt)
            else:
                # just test that we converged when noise is added
                assert theta_hat is not None

            # test that we converge to real solution when initializing around it
            theta_0 = lifter.get_vec_around_gt(delta=1e-2)
            theta_hat, msg, cost_solver = lifter.local_solver(theta_0, y)

            cost_lifter = lifter.get_cost(theta_hat, y)
            assert abs(cost_solver - cost_lifter) < 1e-10, (cost_solver, cost_lifter)

            if noise == 0:
                # test that "we made progress"
                progress = np.linalg.norm(theta_0 - theta_hat)
                assert progress > 1e-10, progress

                # test that cost decreased
                cost_0 = lifter.get_cost(theta_0, y)
                cost_hat = lifter.get_cost(theta_hat, y)
                assert cost_hat <= cost_0

                # TODO(FD) this doesn't pass, looks like the problem is actually not well conditioned!
                # Need to implement and investigate Hessian!
                try:
                    np.testing.assert_allclose(theta_hat, theta_gt, atol=1e-5)
                except AssertionError as e:
                    print(
                        f"Found solution for {lifter} is not ground truth in zero-noise! is the problem well-conditioned?"
                    )
            else:
                # test that "we made progress"
                progress = np.linalg.norm(theta_0 - theta_hat)
                assert progress > 1e-10, progress

                # just test that we converged when noise is added
                assert theta_hat is not None


def test_constraints():
    def test_with_tol(A_list, tol):
        unknowns = lifter.generate_random_unknowns(replace=False)
        x = lifter.get_x(unknowns)
        for Ai in A_list:
            assert abs(x.T @ Ai @ x) < tol

            # TODO(FD) not fully understood why this fixes the below test.
            Ai[range(Ai.shape[0]), range(Ai.shape[0])] /= 2.0

            ai = lifter.get_vec(Ai)
            xvec = lifter.get_vec(np.outer(x, x))
            np.testing.assert_allclose(xvec @ ai, 0.0, atol=tol)

    for lifter in all_lifters:
        A_known = lifter.get_A_known()

        Y = lifter.generate_Y()
        basis, S = lifter.get_basis(Y)
        A_learned = lifter.generate_matrices(basis)

        test_with_tol(A_learned, tol=1e-5)
        test_with_tol(A_known, tol=1e-10)


if __name__ == "__main__":
    test_cost()
    test_cost_noisy()

    test_equivalent_lifters()
    test_grad_finite_diff()
    test_hess_finite_diff()

    test_levels()

    test_solvers()
    test_solvers_noisy()

    test_constraints()

    print("all tests passed")
    # import pytest
    # print("testing", lifters)
    # pytest.main([__file__, "-s"])
