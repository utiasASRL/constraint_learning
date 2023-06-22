import matplotlib.pylab as plt
import numpy as np

from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from lifters.range_only_slam2 import RangeOnlySLAM2Lifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

d = 3
n_landmarks = 5
n_poses = 5


def test_equivalent_lifters():
    np.random.seed(1)
    l1 = RangeOnlySLAM1Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)
    np.random.seed(1)
    l2 = RangeOnlySLAM2Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)

    np.random.seed(1)
    l1.generate_random_setup()
    Q, y = l1.get_Q(noise=1e-3)
    cost1 = l1.get_cost(l1.theta, y)

    np.random.seed(1)
    l2.generate_random_setup()
    Q, y = l2.get_Q(noise=1e-3)
    cost2 = l2.get_cost(l2.theta, y)
    assert abs(cost1 - cost2) < 1e-10

    # 1, t1, t2, ..., a1, a2, ..., |t1|^2, ..., |a1|^2, ..., a1 @ t1, ...
    x1 = l1.get_x()
    # 1, t1, t2, ..., a1, a2, ..., |t1 - a1|^2
    x2 = l2.get_x()

    start = 1 + l1.N
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


def test_gauge():
    import itertools

    # TODO(FD): understand why 3D needs 5 landmarks and positions
    params = [
        dict(n_landmarks=5, n_positions=5, d=3),
        dict(n_landmarks=3, n_positions=3, d=2),
    ]
    remove_gauge_list = ["hard", "cost", None]

    for param, remove_gauge in itertools.product(params, remove_gauge_list):
        lifter = RangeOnlySLAM1Lifter(**param, remove_gauge=remove_gauge)

        Q, y = lifter.get_Q(noise=0)
        theta = lifter.get_vec_around_gt(delta=0).flatten("C")

        grad = lifter.get_grad(theta, y)
        hess = lifter.get_hess(theta, y)

        np.testing.assert_almost_equal(grad, 0.0)

        try:
            S, V = np.linalg.eig(hess)
        except:
            S, V = np.linalg.eig(hess.toarray())
        mask = np.abs(S) < 1e-10
        n_null = np.sum(mask)

        if remove_gauge == "hard":
            assert n_null == 0
        else:
            assert n_null > 0
        continue

        eigvecs = V[:, mask]
        print(f"Hessian has nullspace of dimension {n_null}! Eigenvectors:", eigvecs)

        fig, ax = lifter.plot_setup(title=f"Nullspace dim: {n_null}")
        for i in range(n_null):
            vec = eigvecs[:, i]  # p0_x, p0_y, ...
            lifter.plot_nullvector(vec, ax, color=f"C{i}")

            # theta_delta = theta + eigvec
            # cost_delta = lifter.get_cost(theta_delta, y)
            # assert cost_delta < 1e-10, cost_delta
    print("done")
    return None


if __name__ == "__main__":
    test_equivalent_lifters()
    test_gauge()
    test_levels()
    print("all tests passed")
