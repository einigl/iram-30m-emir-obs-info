from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .model import MeudonPDR
from .instruments import Instrument
from .sampling import Sampler, Constant

from .simulator import Simulator

import infovar


__all__ = [
    "PDRGetter"
]

class PDRGetter(infovar.StandardGetter):
    """
    TODO
    """

    def __init__(
        self,
        instrument: Instrument,
        n_samples: int,
        samplers: List[Sampler],
        obs_time: Union[float, List[float]],
        noise: bool=True,
        kappa: Sampler=Constant(1.),
        seed: Optional[int]=None
    ):
        
        # Simulator
        self.simulator = Simulator(
            instrument
        )

        # Load and preprocess data
        df = self.simulator.simulate(
            n_samples,
            samplers,
            instrument.lines,
            obs_time,
            noise=noise,
            kappa=kappa,
            seed=seed
        )

        # Attributes
        self.x_names = MeudonPDR.parameters
        self.y_names = instrument.lines
        self.x = df.loc[:, self.x_names].values
        self.y = df.loc[:, self.y_names].values


class Infobs:

    def __init__(
        self,
        instrument: Instrument,
        n_samples: int,
        samplers: List[Sampler],
        obs_time: Union[float, List[float]],
        noise: bool=True,
        kappa: Sampler=Constant(1.),
        seed: Optional[int]=None
    ):
        raise NotImplementedError("TODO")

    def reset(self):
        raise NotImplementedError("TODO")
