from typing import List, Optional

import numpy as np

from .samplers import Sampler

__all__ = [
    "Mixture"
]

class Mixture(Sampler):

    def __init__(
        self,
        samplers: List[Sampler],
        weights: Optional[List[float]]=None
    ):
        assert all([isinstance(el, Sampler) for el in samplers])
        self.samplers = samplers

        if weights is None:
            weights = np.ones(len(samplers))
        assert len(samplers) == len(weights)

        if not isinstance(weights, np.ndarray):
            weights = np.ndarray(weights)
        self.p = weights / weights.sum()

    def get(self, n: int) -> np.ndarray:
        samples = np.column_stack([
            sampler.get(n) for sampler in self.samplers
        ])
        idx = np.random.choice(
            np.arange(self.weights.size), size=n, p=self.p
        )
        return samples[np.arange(n), idx]
