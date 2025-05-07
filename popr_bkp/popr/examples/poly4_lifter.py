import numpy as np
from popr.lifters import PolyLifter


class Poly4Lifter(PolyLifter):
    @property
    def VARIABLE_LIST(self):
        return [[self.HOM, "t", "z0"]]

    def __init__(self):
        # actual minimum
        super().__init__(degree=4)

    def get_Q_mat(self):
        Q = np.r_[np.c_[2, 1, 0], np.c_[1, -1 / 2, -1 / 3], np.c_[0, -1 / 3, 1 / 4]]
        return Q

    def get_A_known(self, target_dict=None, output_poly=False):
        from poly_matrix import PolyMatrix

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1[self.HOM, "z0"] = -1
        A_1["t", "t"] = 2
        if output_poly:
            return [A_1]
        else:
            return [A_1.get_matrix(self.var_dict)]

    def generate_random_setup(self):
        self.theta_ = np.array([-1])

    def get_D(self, that):
        # TODO(FD) generalize and move to PolyLifter
        D = np.array(
            [
                [1.0, 0.0, 0.0],
                [that, 1.0, 0.0],
                [that**2, 2 * that, 1.0],
            ]
        )
        return D
