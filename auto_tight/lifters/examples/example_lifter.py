import numpy as np

from auto_tight.lifters.state_lifter import StateLifter


class ExampleLifter(StateLifter):
    """Example Lifter class.

    To create a new Lifter for your problem formulation, create
    a copy of this file and fill in the missing parts.

    You can take a look at the example files in `lifters.examples` for
    inspiration.

    """

    def __init__(self):
        pass

    def get_Q(self, y):
        """ """
        pass

    def get_x(self, y):
        pass

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        return

    def get_x(self, theta, var_subset=None) -> np.ndarray:
        return

    def get_p(self, parameters=None, var_subset=None) -> np.ndarray:
        return

    def sample_theta(self) -> dict | np.ndarray:
        return
