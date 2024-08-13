from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .model import MeudonPDR
from .instruments import Instrument
from .sampling import Sampler, Constant


__all__ = [
    "Simulator"
]

class Simulator:

    def __init__(
        self,
        instrument: Instrument  
    ):
        self.model = MeudonPDR()
        self.instrument = instrument

    @staticmethod
    def params_sampling(
        n_samples: int,
        samplers: Dict[str, Sampler],
        seed: Optional[int]=None
    ) -> pd.DataFrame:
        """
        TODO
        """

        np.random.seed(seed)

        Theta = [None] * len(samplers)

        for i, name in enumerate(samplers):
            Theta[i] = samplers[name].get(n_samples)

        return pd.DataFrame(np.column_stack(Theta), columns=samplers.keys())

    def simulate(
        self,
        n_samples: int,
        samplers: Dict[str, Sampler],
        lines: Optional[List[str]]=None,
        obstime: Union[float, List[float]]=None,
        kappa: Sampler=Constant(1.),
        seed: Optional[int]=None,
    ) -> pd.DataFrame:
        """
        TODO
        """
        MeudonPDR.check_parameters(
            list(samplers.keys())
        )

        # Parameters sampling
        df_params = self.params_sampling(
            n_samples,
            samplers,
            seed=seed
        )

        # PDR code predictions
        df = self.model.predict(
            df_params,
            lines=lines,
            kappa=kappa
        )

        # Receiver measurement
        return self.instrument.measure(
            df,
            obstime
        )
