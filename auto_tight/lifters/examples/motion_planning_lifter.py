import numpy as np
from auto_tight.lifters.state_lifter import StateLifter
from utils.common import upper_triangular

from mp_certs.setups import Setup


class MotionPlanningLifter(StateLifter):
    """Lifter for simple motion planning problem."""

    HOM = "l"
    LEVELS = ["no"]

    def __init__(self, setup: Setup):
        self.setup = setup
        super().__init__(param_level="p")

    @property
    def var_dict(self):
        var_dict = {self.HOM: 1}
        var_dict.update(
            {f"x_{i}": 2 * self.setup.d for i in range(1, self.setup.N - 1)}
        )
        return var_dict

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        trajectory = theta

        # note that we start at x_1 in var_dict here, therefore we use n-1
        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif "x" in key:
                n = int(key.split("_")[-1])
                x_data += list(trajectory[n - 1])
            elif "z" in key:
                n = int(key.split("_")[-1])
                if self.level == "no":
                    x_data.append(np.linalg.norm(trajectory[n - 1]) ** 2)
                elif self.level == "quad":
                    x_data += list(upper_triangular(trajectory[n - 1]))
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def get_p(self, parameters=None, var_subset=None):
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        if self.param_level == "no":
            return np.array([1.0])

        parameters_here = parameters.reshape((2, -1))
        indices = self.get_variable_indices(var_subset)
        if len(indices) == 0:
            return np.array([1.0])
        else:  # either start or end position
            assert len(indices) == 1
            idx = indices[0]

        sub_p = np.hstack([1.0, parameters_here[idx]])
        if self.param_level == "p":
            return sub_p
        elif self.param_level == "ppT":
            return upper_triangular(sub_p)
        else:
            raise ValueError("unnown parameters")

    def sample_parameters(self, theta: np.ndarray = None) -> dict | np.ndarray:
        return np.concatenate([self.current_source, self.current_target])

    def sample_theta(self) -> dict | np.ndarray:
        assert isinstance(self.setup, Setup)
        trajectory = self.setup.generate_random_trajectory()
        self.current_source = trajectory[0, :]
        self.current_target = trajectory[-1, :]
        return trajectory[1:-1, :]

    ### Below functions are only necessary for AutoTemplate.
    def get_Q(self, output_poly=False):
        Q_poly = self.setup.get_Q_min_length()
        if output_poly:
            return Q_poly, None
        return Q_poly.get_matrix(self.var_dict), None

    def get_A_known(self, var_dict=None, output_poly: bool = False) -> list:
        if var_dict is None:
            var_dict = self.var_dict
        A_known_poly = self.setup.get_A_known(variable_dict=var_dict)
        A_known_poly += self.setup.get_A_known_redundant(
            level=1, variable_dict=var_dict
        )
        if output_poly:
            return A_known_poly
        return [A.get_matrix(var_dict) for A in A_known_poly]

    def get_variable_indices(self, var_subset):
        if "x_1" in var_subset:
            return [0]
        elif f"x_{self.setup.N - 1}" in var_subset:
            return [1]
        return []
