import numpy as np

from auto_tight.lifters.examples import (
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


# Below, we always reset seeds to make sure tests are reproducible.
def all_lifters():
    for Lifter, kwargs in Lifters:
        np.random.seed(1)
        yield Lifter(**kwargs)
