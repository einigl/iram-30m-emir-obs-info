from typing import List, Optional

import numpy as np

from .samplers import Sampler

__all__ = ["Mixture"]


class Mixture(Sampler):
    """mixture of probability distributions. To sample a value from this distribution:
    1) select which probability distribution to use (each probability distribution has a weight that determines its selection probability)
    2) draw a value from the selected probability distribution
    """

    def __init__(self, samplers: List[Sampler], weights: Optional[List[float]] = None):
        """
        Uniform
                Parameters
                ----------
                samplers : List[Sampler]
                    list of probability distributions that are mixed
                weights : Optional[List[float]], optional
                    selection probabilities for each sampler (if None, uniform selection probabilities are considered), by default None
        """
        assert all([isinstance(el, Sampler) for el in samplers])
        self.samplers = samplers

        if weights is None:
            weights = np.ones(len(samplers))
        assert len(samplers) == len(weights)

        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self.p = weights / weights.sum()

    def get(self, n: int) -> np.ndarray:
        samples = np.column_stack([sampler.get(n) for sampler in self.samplers])
        idx = np.random.choice(np.arange(self.p.size), size=n, p=self.p)
        return samples[np.arange(n), idx]
