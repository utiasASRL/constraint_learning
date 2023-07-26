from lifters.state_lifter import StateLifter

import numpy as np

class NewLifter(StateLifter):
    LEVELS = ["no"] 
    PARAM_LEVELS = ["no"]

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(self, level="no", param_level="no", d=3):
        super().__init__(level=level, param_level=param_level)

    def var_dict(self):
        """Return key,size pairs of all variables."""
        return 

    def get_param_idx_dict(self, var_subset=None):
        """Return key,index pairs of all parameters touched by var_subset"""
        return

    def get_level_dims(self, n=1):
        """Return the dimension of the chosen lifting level, for n parameters"""
        return

    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        return

    def generate_random_theta(self, factor=1.0):
        """Generate a random new feasible point, this is the ground truth. """
        pass

    def sample_theta(self):
        """Sample a new feasible theta."""
        return

    def sample_parameters(self, x=None):
        """Sample new parameters, given x."""
        return

    def get_x(self, theta, var_subset=None) -> np.ndarray:
        """Get the lifted vector x given theta."""
        return

    def get_p(self, parameters=None, var_subset=None) -> np.ndarray:
        """Get the lifted vector p given the parameters."""
        return

    def get_parameters(self, var_subset=None) -> list:
        """Get the current paratmers given the (fixed) setup."""
        return

    def get_grad(self, t, y):
        Warning("get_grad not implement yet")
        return None

    def get_Q(self, noise=1e-3):
        Warning("get_Q not implemented yet")
        return None, None