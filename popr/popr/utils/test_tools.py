import numpy as np
from popr.examples import (
    MonoLifter,
    Poly4Lifter,
    Poly6Lifter,
    RangeOnlyLocLifter,
    RangeOnlySLAM1Lifter,
    RangeOnlySLAM2Lifter,
    Stereo1DLifter,
    Stereo2DLifter,
    Stereo3DLifter,
    WahbaLifter,
)
from popr.utils.common import get_vec

d = 2
n_landmarks = 3
n_poses = 4
Lifters = [
    (Poly4Lifter, dict()),
    (Poly6Lifter, dict()),
    (WahbaLifter, dict(n_landmarks=3, d=2, robust=False, level="no", n_outliers=0)),
    (MonoLifter, dict(n_landmarks=5, d=2, robust=False, level="no", n_outliers=0)),
    (WahbaLifter, dict(n_landmarks=5, d=2, robust=True, level="xwT", n_outliers=1)),
    (MonoLifter, dict(n_landmarks=6, d=2, robust=True, level="xwT", n_outliers=1)),
    (
        RangeOnlyLocLifter,
        dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no"),
    ),
    (
        RangeOnlyLocLifter,
        dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad"),
    ),
    (Stereo1DLifter, dict(n_landmarks=n_landmarks)),
    (Stereo1DLifter, dict(n_landmarks=n_landmarks, param_level="p")),
    (Stereo2DLifter, dict(n_landmarks=n_landmarks)),
    (Stereo3DLifter, dict(n_landmarks=n_landmarks)),
]

ExampleLifters = [
    (WahbaLifter, dict(n_landmarks=5, d=2, robust=False, level="no", n_outliers=1)),
    (
        RangeOnlyLocLifter,
        dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad"),
    ),
    (Stereo1DLifter, dict(n_landmarks=n_landmarks)),
    (Stereo2DLifter, dict(n_landmarks=n_landmarks)),
]


def _test_with_tol(lifter, A_list, tol):
    x = lifter.get_x().astype(float).reshape((-1, 1))
    for Ai in A_list:
        err = abs((x.T @ Ai @ x)[0, 0])
        assert err < tol, err

        ai = get_vec(Ai.toarray())
        xvec = get_vec(np.outer(x, x))
        np.testing.assert_allclose(ai @ xvec, 0.0, atol=tol)

        ai = get_vec(Ai)
        xvec = get_vec(np.outer(x, x))
        np.testing.assert_allclose(ai @ xvec, 0.0, atol=tol)


# Below, we always reset seeds to make sure tests are reproducible.
def all_lifters(seed=1):
    for Lifter, kwargs in Lifters:
        np.random.seed(seed)
        yield Lifter(**kwargs)


def example_lifters(seed=1, param_level="no"):
    for Lifter, kwargs in ExampleLifters:
        np.random.seed(seed)
        kwargs["param_level"] = param_level
        yield Lifter(**kwargs)
