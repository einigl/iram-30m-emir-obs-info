from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = [
    "Sampler",
    "Constant",
    "Uniform",
    "LogUniform",
    "BoundedPowerLaw"
]

class Sampler(ABC):

    @abstractmethod
    def get(self, n: int) -> np.ndarray:
        pass
    
class Constant(Sampler):

    def __init__(self, value: float):
        """
        TODO
        """
        self.value = value

    def get(self, n: int) -> np.ndarray:
        """
        TODO
        """
        return self.value * np.ones(n, dtype=float)
    
    def __str__(self):
        return f"Constant(value={self.value})"
    
class Uniform(Sampler):

    def __init__(self, lower: float, upper: Optional[float]=None):
        """
        TODO
        """
        assert upper is None or upper >= lower

        self.lower = lower
        self.upper = upper or lower

    def get(self, n: int) -> np.ndarray:
        """
        TODO
        """
        return np.random.uniform(self.lower, self.upper, n)
    
    def copy_other_bounds(self, lower: float, upper: Optional[float]=None):
        """
        TODO
        """
        return Uniform(lower, upper)

    def __str__(self):
        return f"Uniform(lower={self.lower}, upper={self.upper})"

class LogUniform(Sampler):

    def __init__(self, lower: float, upper: Optional[float]=None, base: float=10.):
        """
        TODO
        """
        assert lower > 0
        assert upper is None or upper >= lower
        assert base > 1

        self.lower = lower
        self.upper = upper or lower
        self.base = base

    def get(self, n: int) -> np.ndarray:
        """
        TODO
        """
        a = np.log(self.lower) / np.log(self.base)
        b = np.log(self.upper) / np.log(self.base)
        return self.base**np.random.uniform(a, b, n)
    
    def copy_other_bounds(self, lower: float, upper: Optional[float]=None):
        """
        TODO
        """
        return LogUniform(lower, upper, base=self.base)
    
    def __str__(self):
        return f"LogUniform(lower={self.lower}, upper={self.upper}, base={self.base})"
    
class BoundedPowerLaw(Sampler):

    def __init__(self, alpha: float, lower: float, upper: Optional[float]=None):
        """
        TODO
        """
        assert lower > 0
        assert upper is None or upper >= lower

        self.alpha = alpha
        self.lower = lower
        self.upper = upper or lower

    def get(self, n: int) -> np.ndarray:
        """
        TODO
        """
        a = self.lower**self.alpha
        b = self.upper**self.alpha - a
        return (a + b*np.random.rand(n))**(1/self.alpha)

    def copy_other_bounds(self, lower: float, upper: Optional[float]=None):
        """
        TODO
        """
        return BoundedPowerLaw(self.alpha, lower, upper)

    def __str__(self):
        return f"BoundedPowerLaw(alpha={self.alpha}, lower={self.lower}, upper={self.upper})"