import numpy as np

from lifters.poly_lifters import Poly4Lifter, Poly6Lifter, PolyLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from lifters.range_only_slam2 import RangeOnlySLAM2Lifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

d = 2
n_landmarks = 4
n_poses = 4
Lifters = {
    # Poly4Lifter: dict(),
    # Poly6Lifter: dict(),
    # Stereo1DLifter: dict(n_landmarks=n_landmarks),
    Stereo2DLifter: dict(n_landmarks=n_landmarks),
    # Stereo3DLifter: dict(n_landmarks=n_landmarks),
    RangeOnlySLAM1Lifter: dict(
        n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="inner"
    ),
    # RangeOnlySLAM1Lifter: dict(
    #    n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="outer"
    # ),
    # RangeOnlySLAM2Lifter: dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    # RangeOnlyLocLifter: dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
}


# Below, we always reset seeds to make sure tests are reproducible.
def all_lifters():
    for Lifter, kwargs in Lifters.items():
        np.random.seed(1)
        yield Lifter(**kwargs)
