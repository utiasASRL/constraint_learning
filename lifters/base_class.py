from abc import ABC, abstractproperty, abstractmethod

import numpy as np


class BaseClass(ABC):
    LEVELS = ["no"]
    PARAM_LEVELS = ["no"]

    def __init__(self, level="no", param_level="no"):
        assert level in self.LEVELS
        self.level = level

        assert param_level in self.PARAM_LEVELS
        self.param_level = param_level

        # variables that get overwritten upon initialization
        self.var_dict_ = None
        self.theta_ = None

        self.generate_random_setup()

    @property
    def d(self):
        return self.d_

    @d.setter
    def d(self, var):
        assert var in [1, 2, 3]
        self.d_ = var

    @abstractproperty
    def var_dict(self):
        """Return key,size pairs of all variables."""
        return

    @abstractmethod
    def get_param_idx_dict(self, var_subset=None):
        """Return key,index pairs of all parameters touched by var_subset"""
        return

    @abstractmethod
    def get_level_dims(self, n=1):
        return

    @abstractmethod
    def generate_random_setup(self):
        return

    @abstractmethod
    def generate_random_theta(self, factor=1.0):
        pass

    @abstractmethod
    def sample_theta(self):
        return

    @abstractmethod
    def sample_theta(self):
        return

    @abstractmethod
    def sample_parameters(self, x=None):
        return

    @abstractmethod
    def get_x(self, theta, var_subset=None) -> np.ndarray:
        return

    @abstractmethod
    def get_p(self, parameters=None, var_subset=None) -> np.ndarray:
        return

    @abstractmethod
    def get_parameters(self, var_subset=None) -> list:
        return

    def get_grad(self, t, y):
        Warning("get_grad not implement yet")
        return None

    def get_Q(self, noise=1e-3):
        Warning("get_Q not implemented yet")
        return None, None
