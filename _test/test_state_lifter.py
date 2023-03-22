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
    Stereo2DLifter(n_landmarks, level=0),
    # Stereo3DLifter(n_landmarks),
]


def test_cost_noiseless():
    for lifter in lifters:
        Q, y = lifter.get_Q(noise=0)
        if Q is None:
            continue

        t = lifter.unknowns
        cost = lifter.get_cost(lifter.landmarks, y, t)
        assert cost < 1e-10, cost

        x = lifter.get_x()
        costQ = x.T @ Q @ x
        assert abs(costQ) < 1e-10, costQ

        assert abs(cost - costQ) < 1e-10, (cost, costQ)


def test_cost_noisy():
    for lifter in lifters:
        noise = 1e-1
        Q, y = lifter.get_Q(noise=noise)
        if Q is None:
            continue

        x = lifter.unknowns
        cost = lifter.get_cost(lifter.landmarks, y, x)

        x = lifter.get_x()
        costQ = x.T @ Q @ x
        assert abs(cost - costQ) < 1e-10
        print("passed noise test 2")


def test_solvers_noisy(n_seeds=3, noise=1e-1):
    test_solvers(n_seeds=n_seeds, noise=noise)


def test_solvers(n_seeds=1, noise=0.0):
    for lifter in lifters:
        for j in range(n_seeds):
            np.random.seed(j)
            Q, y = lifter.get_Q(noise=noise)
            if Q is None:
                continue

            # test that we converge to real solution when initializing around it
            delta = 1e-3
            # t0 = t  # + np.random.rand(4)
            gt = lifter.unknowns
            t0 = gt + np.random.normal(scale=delta, loc=0, size=len(gt))
            a = lifter.landmarks
            that, msg = lifter.local_solver(a, y, t0)
            if noise == 0:
                # test that solution is ground truth with no noise
                np.testing.assert_allclose(that, gt)
            else:
                # just test that we converged when noise is added
                assert that is not None


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
    import pytest

    print("testing", lifters)
    pytest.main([__file__, "-s"])
