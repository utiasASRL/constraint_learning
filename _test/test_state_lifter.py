import numpy as np

from lifters.custom_lifters import Poly4Lifter, Poly6Lifter, RangeOnlyLifter
from lifters.landmark_lifter import PoseLandmarkLifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

n_landmarks = 3
n_poses = 2
lifters = [
    # Poly4Lifter(),
    # Poly6Lifter(),
    # RangeOnlyLifter(n_positions=n_poses, d=2),
    # PoseLandmarkLifter(n_landmarks, n_poses, d=2),
    # Stereo1DLifter(n_landmarks),
    # Stereo2DLifter(n_landmarks, level=0),
    Stereo3DLifter(n_landmarks),
]


def test_cost_noisy():
    test_cost(noise=0.1)


def test_cost(noise=0.0):
    for lifter in lifters:
        # np.random.seed(1)
        Q, y = lifter.get_Q(noise=noise)
        # np.random.seed(1)
        # Qold, yold = lifter.get_Q_old(noise=noise)
        # np.testing.assert_allclose(Q, Qold)
        # np.testing.assert_allclose(y, yold)
        if Q is None:
            continue

        x = lifter.unknowns
        cost = lifter.get_cost(lifter.landmarks, y, x, W=lifter.W)

        x = lifter.get_x()
        costQ = x.T @ Q @ x
        assert abs(cost - costQ) < 1e-8

        if noise == 0:
            assert cost < 1e-10, cost
            assert costQ < 1e-10, costQ


def test_solvers_noisy(n_seeds=3, noise=1e-1):
    test_solvers(n_seeds=n_seeds, noise=noise)


def test_solvers(n_seeds=1, noise=0.0):
    for lifter in lifters:
        for j in range(n_seeds):
            np.random.seed(j)

            # noisy setup
            Q, y = lifter.get_Q(noise=noise)
            if Q is None:
                continue

            # test that we converge to real solution when initializing around it
            theta_0 = lifter.get_vec_around_gt(delta=1e-3)
            theta_gt = lifter.get_vec_around_gt(delta=0)

            theta_hat, msg, cost_solver = lifter.local_solver(
                lifter.landmarks, y, theta_0, W=lifter.W
            )

            cost_lifter = lifter.get_cost(lifter.landmarks, y, t=theta_hat, W=lifter.W)
            assert abs(cost_solver - cost_lifter) < 1e-10, (cost_solver, cost_lifter)

            if noise == 0:
                # test that solution is ground truth with no noise
                np.testing.assert_allclose(theta_hat, theta_gt)
            else:
                # just test that we converged when noise is added
                assert theta_hat is not None


def test_constraints():
    for lifter in lifters:
        x = lifter.get_x()

        A_known = lifter.get_A_known()
        for Ai in A_known:
            assert abs(x.T @ Ai @ x) < 1e-10
            print("passed constraints test 1")

            # TODO(FD) not fully understood why this fixes the below test.
            Ai[range(Ai.shape[0]), range(Ai.shape[0])] /= 2.0

            ai = lifter.get_vec(Ai)
            xvec = lifter.get_vec(np.outer(x, x))
            np.testing.assert_allclose(xvec @ ai, 0.0, atol=1e-10)
            print("passed constraints test 2")
    print("done")


if __name__ == "__main__":
    test_cost()
    test_cost_noisy()

    test_solvers()
    test_solvers_noisy()

    test_constraints()

    # import pytest
    # print("testing", lifters)
    # pytest.main([__file__, "-s"])
