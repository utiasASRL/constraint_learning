from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class BaseClass(ABC):
    """
    This BaseClass is to be used as a template for creating new Lifters.
    All the listed functionalities need to be defined by inheriting functions.
    """

    LEVELS = ["no"]
    PARAM_LEVELS = ["no"]
    VARIABLE_LIST = ["h"]

    def __init__(
        self,
        level="no",
        param_level="no",
        d=2,
        variable_list=None,
        robust=False,
        n_outliers=0,
    ):
        self.robust = robust
        self.n_outliers = n_outliers

        assert level in self.LEVELS
        self.level = level

        assert param_level in self.PARAM_LEVELS
        self.param_level = param_level

        if variable_list is not None:
            self.variable_list = variable_list
        else:
            self.variable_list = self.VARIABLE_LIST

        # variables that get overwritten upon initialization
        self.parameters = [1.0]
        self.theta_ = None
        self.var_dict_ = None
        self.y_ = None

        self.d = d

    @abstractproperty
    def var_dict(self):
        """Return key,size pairs of all variables."""
        return

    @abstractproperty
    def theta(self):
        """Return ground truth theta."""
        return

    @abstractmethod
    def get_x(self, theta, var_subset=None) -> np.ndarray:
        return

    @abstractmethod
    def get_p(self, parameters=None, var_subset=None) -> np.ndarray:
        return

    @abstractmethod
    def sample_theta(self):
        return

    @abstractmethod
    def get_Q(self, noise=1e-3):
        Warning("get_Q not implemented yet")
        return None, None

    def get_A_known(self):
        return []

    def sample_parameters(self, x=None):
        if self.param_level == "no":
            return [1.0]

    def get_parameters(self, var_subset=None) -> list:
        if self.param_level == "no":
            return [1.0]

    def get_grad(self, t, y):
        Warning("get_grad not implement yet")
        return None
