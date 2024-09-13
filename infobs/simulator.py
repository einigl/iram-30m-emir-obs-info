from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .instruments import Instrument
from .model import MeudonPDR
from .sampling import Constant, Sampler

__all__ = ["Simulator"]


class Simulator:
    """
    A class to generate synthetic observations of an environment.
    """

    def __init__(self, instrument: Instrument):
        self.model = MeudonPDR(instrument.kelvin)
        self.instrument = instrument

    @staticmethod
    def params_sampling(
        n_samples: int, samplers: Dict[str, Sampler], seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulates `n_samples` samples of physical parameter vectors from the probability distributions set in `samplers`

        Parameters
        ----------
        n_samples : int
            number of samples to generate
        samplers : Dict[str, Sampler]
            probability distribution to sample from for each physical parameter
        seed : Optional[int], optional
            random seed, by default None

        Returns
        -------
        pd.DataFrame
            dataframe containing the sampled physical parameters
        """
        np.random.seed(seed)

        Theta = []
        for name in samplers:
            Theta.append(samplers[name].get(n_samples))

        return pd.DataFrame(np.column_stack(Theta), columns=list(samplers.keys()))

    def simulate(
        self,
        n_samples: int,
        samplers: Dict[str, Sampler],
        lines: Optional[List[str]] = None,
        obstime: Union[float, List[float]] = None,
        kappa: Sampler = Constant(1.0),
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """simulates `n_samples` samples from the probability distributions set in `samplers`, and then predicts the integrated intensities for all or a subset of lines, for a given integration time `obstime`.

        Parameters
        ----------
        n_samples : int
            number of samples to generate
        samplers : Dict[str, Sampler]
            probability distribution to sample from for each physical parameter
        lines : Optional[List[str]], optional
            list of the names of the lines to predict, by default None
        obstime : Union[float, List[float]], optional
            integration time, by default None
        kappa : Sampler, optional
            multiplicative factor that includes e.g., beam dilution effects, by default Constant(1.)
        seed : Optional[int], optional
            random seed, by default None

        Returns
        -------
        pd.DataFrame
            dataframe containing the sampled physical parameters and the associated predicted integrated intensities
        """
        MeudonPDR._check_parameters(pd.DataFrame(columns=list(samplers.keys())))

        # Parameters sampling
        df_params = self.params_sampling(n_samples, samplers, seed=seed)

        # PDR code predictions
        df = self.model.predict(df_params, lines=lines, kappa=kappa)

        # Receiver measurement
        return self.instrument.measure(df, obstime)
