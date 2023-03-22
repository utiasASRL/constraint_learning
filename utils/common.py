import numpy as np
def get_T(t: np.ndarray):
    """
    :param t: is composed of x, y, alpha
    """
    if len(t) == 6:
        T = np.zeros((3, 3))
        T[-1, -1] = 1.0
        T[:2, 0] = t[:2]
        T[:2, 1] = t[2:4]
        T[:2, -1] = t[-2:]
    else:
        raise NotImplementedError
    return T


def get_rot_matrix(rot):
    from scipy.spatial.transform import Rotation as R

    """ return the desired parameterization """
    if np.ndim(rot) <= 1:
        if np.ndim(rot) == 1:
            rot = float(rot)
        r = R.from_euler("z", rot)
        return r.as_matrix()[:2, :2]
    elif len(rot) == 2:
        r = R.from_euler("xyz", rot)
        return r.as_matrix()
