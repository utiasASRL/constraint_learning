import numpy as np

from auto_tight.lifters.state_lifter import StateLifter
from utils.common import upper_triangular


class MotionPlanningLifter(StateLifter):
    """Lifter for simple motion planning problem."""

    HOM = "h"
    LEVELS = ["no"]

    def __init__(self, setup):
        from mp_certs.setups import Setup

        assert isinstance(setup, Setup)
        self.setup = setup
        super().__init__(param_level="p")

    @property
    def var_dict(self):
        var_dict = {self.HOM: 1}
        var_dict.update(
            {f"x_{i}": 2 * self.setup.d for i in range(1, self.setup.N - 1)}
        )
        return var_dict

    @property
    def param_dict(self):
        param_dict = {self.HOM: 1}
        if self.param_level == "p":
            param_dict["x_s"] = 4
            param_dict["x_t"] = 4
        elif self.param_level == "ppT":
            # add all second-order terms: 4 * 5 / 2
            param_dict["x_s"] = 10
            param_dict["x_t"] = 10
        return param_dict

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

    def get_p(self, parameters: dict = None, param_subset: dict | list = None):
        if parameters is None:
            parameters = self.parameters
        if param_subset is None:
            param_subset = self.param_dict
        assert isinstance(parameters, dict)

        p_data = [1.0]
        for key in param_subset:
            if key == self.HOM:
                continue
            if np.ndim(parameters[key]) > 0:
                p_data += list(parameters[key])
            else:
                p_data.append(parameters[key])
        assert len(p_data) == self.get_dim_p(param_subset)
        return np.array(p_data)

    def sample_parameters(self, theta: np.ndarray = None) -> dict | np.ndarray:
        return {self.HOM: 1.0, "x_s": self.current_source, "x_t": self.current_target}

    def sample_theta(self) -> dict | np.ndarray:
        from mp_certs.setups import Setup

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
        return super().get_variable_indices(var_subset, variable="x")

    def get_involved_param_dict(self, var_subset):
        if "x_1" in var_subset:
            return ["x_s"]
        elif f"x_{self.setup.N - 1}" in var_subset:
            return ["x_t"]
        else:
            return []
