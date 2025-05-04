import numpy as np
from popr.lifters import PolyLifter


class Poly6Lifter(PolyLifter):
    @property
    def VARIABLE_LIST(self):
        return [[self.HOM, "t", "z0", "z1"]]

    def __init__(self, poly_type="A"):
        assert poly_type in ["A", "B"]
        self.poly_type = poly_type
        super().__init__(degree=6)

    def get_Q_mat(self):
        if self.poly_type == "A":
            return 0.1 * np.array(
                [
                    [25, 12, 0, 0],
                    [12, -13, -2.5, 0],
                    [0, -2.5, 6.25, -0.9],
                    [0, 0, -0.9, 1 / 6],
                ]
            )
        elif self.poly_type == "B":
            return np.array(
                [
                    [5.0000, 1.3167, -1.4481, 0],
                    [1.3167, -1.4481, 0, 0.2685],
                    [-1.4481, 0, 0.2685, -0.0667],
                    [0, 0.2685, -0.0667, 0.0389],
                ]
            )

    def get_A_known(self, target_dict=None, output_poly=False):
        from poly_matrix import PolyMatrix

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1[self.HOM, "z0"] = -1
        A_1["t", "t"] = 2

        # z_1 = t^3 = t z_0
        A_2 = PolyMatrix(symmetric=True)
        A_2[self.HOM, "z1"] = -1
        A_2["t", "z0"] = 1

        # t^4 = z_1 t = z_0 z_0
        B_0 = PolyMatrix(symmetric=True)
        B_0["z0", "z0"] = 2
        B_0["t", "z1"] = -1
        if output_poly:
            return [A_1, A_2, B_0]
        else:
            return [A_i.get_matrix(self.var_dict) for A_i in [A_1, A_2, B_0]]

    def get_D(self, that):
        # TODO(FD) generalize and move to PolyLifter
        D = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [that, 1.0, 0.0, 0.0],
                [that**2, 2 * that, 1.0, 0.0],
                [that**3, 3 * that**2, 3 * that, 1.0],
            ]
        )
        return D

    def generate_random_setup(self):
        self.theta_ = np.array([-1])
