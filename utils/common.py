import numpy as np

import os.path


def get_fname(name, extension="pkl"):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_results", f"{name}.{extension}")
    )


def get_rot_matrix(rot):
    from scipy.spatial.transform import Rotation as R

    """ return the desired parameterization """
    if np.ndim(rot) == 0 or len(rot) == 1:
        if np.ndim(rot) == 1:
            rot = float(rot)
        r = R.from_euler("z", rot)
        return r.as_matrix()[:2, :2]
    elif len(rot) == 3:
        r = R.from_euler("xyz", rot)
        return r.as_matrix()
    else:
        raise ValueError(rot)


def setup_tex():
    import shutil
    import matplotlib.pylab as plt

    usetex = True if shutil.which("latex") else False
    print("found latex:", usetex)
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )
    import matplotlib

    # matplotlib.use("ps")
    plt.rc("text.latex", preamble=r"\usepackage{bm}\usepackage{color}")
    return plt
