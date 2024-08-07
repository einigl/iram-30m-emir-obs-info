import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from nnbma import NeuralNetwork

from ..sampling.samplers import Sampler, Constant
from ..util import erg_to_kelvin


__all__ = [
    "MeudonPDR"
]

class MeudonPDR:

    parameters: List[str] = ["Av", "G0", "Pth", "angle", "kappa"]

    def __init__(
        self,
        kelvin: bool=True
    ):
        self.kelvin = kelvin

        # Load neural network
        model_name = "meudon_pdr_emulator"
        path_model = os.path.abspath(os.path.dirname(__file__))
        
        self.net = NeuralNetwork.load(
            model_name,
            path_model
        )

        # Reference dataframe
        self.full_df = pd.read_csv(os.path.join(
            os.path.dirname(__file__), "full_table.csv"
        ))

        # Remove lines whose frequency is not available
        self.full_df = self.full_df.dropna(axis=0)

    @staticmethod
    def check_parameters(
        parameters: List[str]
    ) -> None:
        assert set(parameters) == {"Av", "G0", "Pth", "angle"} 

    def predict(
        self,
        df_params: pd.DataFrame,
        lines: Optional[List[str]]=None,
        kappa: Sampler=Constant(1.),
    ) -> pd.DataFrame:
        """
        TODO
        """    
        if lines is None:
            lines = self.full_df["line_id"].to_list()

        # Restrictions

        self.net.restrict_to_output_subset(
            lines
        )

        # Use the appropriate names for the network

        self.check_parameters(
            df_params.columns.to_list()
        )

        df_params_net = df_params.rename(columns={
            "Av": "Avmax",
            "G0": "radm",
            "Pth": "P",
            "angle": "angle"
        })

        # Conversion from G0 to radm

        conv_fact = 1.2786 / 2  # G0 = 1.2786 * radm / 2
        df_params_net["radm"] = df_params_net["radm"] / conv_fact

        # Reorder inputs

        df_params_net = df_params_net[self.net.inputs_names]

        # PDR code predictions

        Y = 10**self.net.evaluate(df_params_net.values, transform_inputs=True)
        
        # Unit conversion
        freqs = 1e9 * self.full_df[self.full_df["line_id"].isin(lines)]["freq"].to_numpy()
        if self.kelvin:
            Y = erg_to_kelvin(Y, freqs)

        # Apply kappa

        _kappa = kappa.get(Y.shape[0]).reshape(-1, 1)

        df = pd.DataFrame(
            np.hstack((df_params.values, _kappa, _kappa * Y)),
            columns=df_params.columns.to_list() + ["kappa"] + self.net.current_output_subset
        )

        return df

    @staticmethod
    def _get_table() -> pd.DataFrame:
        # Reference dataframe
        full_df = pd.read_csv(os.path.join(
            os.path.dirname(__file__), "full_table.csv"
        )).set_index("line_id")

        # Remove lines whose frequency is not available
        full_df = full_df.dropna(axis=0)

        return full_df

    @staticmethod
    def all_lines() -> List[str]:
        full_df = MeudonPDR._get_table()
        return full_df.index.to_list()
    
    @staticmethod
    def frequencies(
        lines: List[str]
    ) -> Dict[str, float]:
        full_df = MeudonPDR._get_table()
        return {line: full_df.loc[line, "freq"] for line in lines}
