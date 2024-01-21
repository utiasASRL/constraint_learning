import numpy as np

from lifters.matweight_lifter import MatWeightLocLifter
from lifters.mono_lifter import MonoLifter
from lifters.poly_lifters import Poly4Lifter, Poly6Lifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from lifters.range_only_slam2 import RangeOnlySLAM2Lifter
from lifters.state_lifter import StateLifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.wahba_lifter import WahbaLifter
from ro_certs.problem import Reg

d = 2
n_landmarks = 3
n_poses = 4
# fmt: off
Lifters = [
    (Poly4Lifter, dict()),
    (Poly6Lifter, dict()),
    (WahbaLifter, dict(n_landmarks=3, d=2, robust=False, level="no", n_outliers=0)),
    (MonoLifter, dict(n_landmarks=5, d=2, robust=False, level="no", n_outliers=0)),
    (WahbaLifter, dict(n_landmarks=5, d=2, robust=True, level="xwT", n_outliers=1)),
    (MatWeightLocLifter, dict(n_landmarks=5, n_poses=n_poses)),
    (MonoLifter, dict(n_landmarks=6, d=2, robust=True, level="xwT", n_outliers=1)),
    (Stereo1DLifter, dict(n_landmarks=n_landmarks)),
    (Stereo1DLifter, dict(n_landmarks=n_landmarks, param_level="p")),
    (Stereo2DLifter, dict(n_landmarks=n_landmarks)),
    (Stereo3DLifter, dict(n_landmarks=n_landmarks)),
]
ROLifters = [
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no", reg=Reg.NONE)),
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no", reg=Reg.ZERO_VELOCITY)),
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no", reg=Reg.CONSTANT_VELOCITY)),
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad", reg=Reg.NONE)),
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad", reg=Reg.ZERO_VELOCITY)),
    (RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="quad", reg=Reg.CONSTANT_VELOCITY)),
]
# Lifters = [(RangeOnlyLocLifter, dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d, level="no", reg=Reg.CONSTANT_VELOCITY))]
# Lifters = []
# ROLifters = []
# fmt: on


# Below, we always reset seeds to make sure tests are reproducible.
def all_lifters() -> StateLifter:
    for Lifter, kwargs in Lifters + ROLifters:
        np.random.seed(1)
        yield Lifter(**kwargs)
