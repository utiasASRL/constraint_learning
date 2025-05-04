import numpy as np
from popr.lifters import StateLifter


class ExampleLifter(StateLifter):
    """Example Lifter class.

    This class implements the bare minimum to use AutoTight.

    To create a new Lifter for your problem formulation, create
    a copy of this file and fill in the missing parts.

    You can take a look at the example files in `auto_tight.lifters.examples` for
    inspiration.

    """

    # you can choose your homogenization variable here
    HOM = "h"

    # chose the "lifting" levels when going up in the sparse Lasserre's hierarchy.
    LEVELS = ["no"]

    def __init__(self, param_level="p"):
        # you can choose if you want to use "parameters". Otherwise remove param_level or set to "no"
        super().__init__(param_level=param_level)

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        var_dict = {self.HOM: 1}
        return var_dict

    @property
    def param_dict(self):
        """Return key,size pairs of all parameters."""
        param_dict = {self.HOM: 1}
        return param_dict

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        """
        Return the lifted vector x.

        :param theta: the vector theta to use. If not given, the ground truth theta is used.
        :param parameters: list of all parameters
        :param var_subset: subset of variables that we care about (will extract corresponding data)
        """
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            # fill in the rest of x according to var_subset.
            # elif "x" in key:
            # elif "z" in key:
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def sample_parameters(self, theta: np.ndarray = None) -> dict | np.ndarray:
        """
        Return the array of parameters.

        :param theta: The vector theta to use. If not given, the ground truth theta is used.
                      Often we don't actually need this because the parameters might not depend
                      on theta (e.g. when parameters = landmarks)
        """
        return

    def sample_theta(self) -> dict | np.ndarray:
        """
        Create a new random instance of theta.
        """
        return
