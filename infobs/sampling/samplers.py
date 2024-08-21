from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = ["Sampler", "Constant", "Uniform", "LogUniform", "BoundedPowerLaw"]


class Sampler(ABC):
    """abstract sampler class"""

    @abstractmethod
    def get(self, n: int) -> np.ndarray:
        """samples from the considered sampler

        Parameters
        ----------
        n : int
            number of samples to draw

        Returns
        -------
        np.ndarray of shape (n,)
            sampled physical parameter values
        """
        pass

    @abstractmethod
    def copy_other_bounds(self, lower: float, upper: Optional[float] = None):
        """generates a copy of the considered sampler with new lower and upper bounds

        Parameters
        ----------
        lower : float
            new lower bound
        upper : Optional[float], optional
            new upper bound, by default None
        """
        pass


class Constant(Sampler):
    """simplest possible probability distribution: a Dirac at a given value"""

    def __init__(self, value: float):
        """

        Parameters
        ----------
        value : float
            considered constant value for the physical parameter
        """
        self.value = value

    def get(self, n: int) -> np.ndarray:
        return self.value * np.ones(n, dtype=float)

    def copy_other_bounds(self, value: float):
        return Constant(value)

    def __str__(self):
        return f"Constant(value={self.value})"


class Uniform(Sampler):
    """uniform distribution on a possible open-ended interval"""

    def __init__(self, lower: float, upper: Optional[float] = None):
        """

        Parameters
        ----------
        lower : float
            lower bound of the uniform distribution
        upper : Optional[float], optional
            upper bound of the uniform distribution, by default None
        """
        assert upper is None or upper >= lower

        self.lower = lower
        self.upper = upper or lower

    def get(self, n: int) -> np.ndarray:
        return np.random.uniform(self.lower, self.upper, n)

    def copy_other_bounds(self, lower: float, upper: Optional[float] = None):
        return Uniform(lower, upper)

    def __str__(self):
        return f"Uniform(lower={self.lower}, upper={self.upper})"


class LogUniform(Sampler):
    """log-uniform distribution on a possible open-ended interval"""

    def __init__(self, lower: float, upper: Optional[float] = None, base: float = 10.0):
        """

        Parameters
        ----------
        lower : float
            lower bound of the log-uniform distribution
        upper : Optional[float], optional
            upper bound of the log-uniform distribution, by default None
        base : float, optional
            logarithm base, by default 10.
        """
        assert lower > 0
        assert upper is None or upper >= lower
        assert base > 1

        self.lower = lower
        self.upper = upper or lower
        self.base = base

    def get(self, n: int) -> np.ndarray:
        a = np.log(self.lower) / np.log(self.base)
        b = np.log(self.upper) / np.log(self.base)
        return self.base ** np.random.uniform(a, b, n)

    def copy_other_bounds(self, lower: float, upper: Optional[float] = None):
        return LogUniform(lower, upper, base=self.base)

    def __str__(self):
        return f"LogUniform(lower={self.lower}, upper={self.upper}, base={self.base})"


class BoundedPowerLaw(Sampler):
    """bounded power law distribution on a possible open-ended interval"""

    def __init__(self, alpha: float, lower: float, upper: Optional[float] = None):
        """

        Parameters
        ----------
        alpha : float
            exponent value of the power law distribution
        lower : float
            lower bound of the bounded power law distribution
        upper : Optional[float], optional
            upper bound of the bounded power law distribution, by default None
        """
        assert lower > 0
        assert upper is None or upper >= lower

        self.alpha = alpha
        self.lower = lower
        self.upper = upper or lower

    def get(self, n: int) -> np.ndarray:
        a = self.lower**self.alpha
        b = self.upper**self.alpha - a
        return (a + b * np.random.rand(n)) ** (1 / self.alpha)

    def copy_other_bounds(self, lower: float, upper: Optional[float] = None):
        return BoundedPowerLaw(self.alpha, lower, upper)

    def __str__(self):
        return f"BoundedPowerLaw(alpha={self.alpha}, lower={self.lower}, upper={self.upper})"
