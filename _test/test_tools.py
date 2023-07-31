import numpy as np

from lifters.poly_lifters import Poly4Lifter, Poly6Lifter, PolyLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from lifters.range_only_slam2 import RangeOnlySLAM2Lifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter

d = 2
n_landmarks = 3
n_poses = 4
Lifters = [
    #(Poly4Lifter, dict()),
    #(Poly6Lifter, dict()),
    #(MonoLifter, dict(n_landmarks=n_landmarks, d=d, robust=False, level="no")),
    #(MonoLifter, dict(n_landmarks=n_landmarks, d=d, robust=True, level="xxT")),
    #(MonoLifter, dict(n_landmarks=n_landmarks, d=d, robust=True, level="xwT")),
    (WahbaLifter, dict(n_landmarks=n_landmarks, d=d, robust=False, level="no")),
    (WahbaLifter, dict(n_landmarks=n_landmarks, d=d, robust=True, level="xxT")),
    (WahbaLifter, dict(n_landmarks=n_landmarks, d=d, robust=True, level="xwT")),
    #(RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no")),
    #(RangeOnlyLocLifter, dict( n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad")),
    #(Stereo1DLifter, dict(n_landmarks=n_landmarks)),
    #(Stereo1DLifter, dict(n_landmarks=n_landmarks, param_level="p")),
    #(Stereo2DLifter, dict(n_landmarks=n_landmarks)),
    #(Stereo3DLifter, dict(n_landmarks=n_landmarks)),
]


# Below, we always reset seeds to make sure tests are reproducible.
def all_lifters():
    for Lifter, kwargs in Lifters:
        np.random.seed(1)
        yield Lifter(**kwargs)
