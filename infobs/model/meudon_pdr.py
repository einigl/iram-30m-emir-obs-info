import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nnbma import NeuralNetwork

from ..sampling.samplers import Constant, Sampler
from ..util import erg_to_kelvin, g0_to_radm, radm_to_g0

__all__ = ["MeudonPDR"]


class ValidityDomainWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class MeudonPDR:

    parameters: List[str] = ["Av", "G0", "Pth", "angle", "kappa"]
    bounds: Dict[str, Tuple[float, float]] = {
        "Av": (1, 40),
        "G0": (radm_to_g0(1e0), radm_to_g0(1e5)),
        "Pth": (1e5, 1e9),
        "angle": (0, 60),
    }

    def __init__(self, kelvin: bool = True):
        """

        Parameters
        ----------
        kelvin : bool, optional
            if True, predicted integrated intensities are expressed in K * km / s. Otherwise, they are expressed in erg cm-2 s-1 sr-1, by default True
        """
        if not isinstance(kelvin, bool):
            raise TypeError(f"kelvin must be a boolean, not {type(kelvin)}")
        self.kelvin = kelvin

        # Load neural network
        model_name = "meudon_pdr_emulator"
        path_model = os.path.abspath(os.path.dirname(__file__))

        self.net = NeuralNetwork.load(model_name, path_model)

        # Reference dataframe
        self.full_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "full_table.csv")
        )

        # Remove lines whose frequency is not available
        self.full_df = self.full_df.dropna(axis=0)

    @staticmethod
    def _check_parameters(df_params: pd.DataFrame) -> None:
        if not isinstance(df_params, pd.DataFrame):
            raise TypeError(f"df_params must be a DataFrame, not {type(df_params)}")

        params = df_params.columns.to_list()
        if set(params) != {"Av", "G0", "Pth", "angle"}:
            raise ValueError(
                f"Input parameters must be [Av, G0, Pth, angle], not {params}"
            )

        if any(
            [
                not df_params.loc[:, p].between(a, b).all()
                for p, (a, b) in MeudonPDR.bounds.items()
            ]
        ):
            warnings.warn(
                "Input parameters must fall within the model's validity domain. See `MeudonPDR.bounds` for the limits of the validity domain.",
                ValidityDomainWarning,
            )

    def predict(
        self,
        df_params: pd.DataFrame,
        lines: Optional[List[str]] = None,
        kappa: Sampler = Constant(1.0),
    ) -> pd.DataFrame:
        """predicts

        _extended_summary_

        Parameters
        ----------
        df_params : pd.DataFrame
            dataframe containing of physical parameter values
        lines : Optional[List[str]], optional
            list of lines to prodict (if None, all lines are predicted), by default None
        kappa : Sampler, optional
            scaling factor that includes, e.g., beam dilution, by default Constant(1.)

        Returns
        -------
        pd.DataFrame
            predicted integrated intensities for each physical parameter vector
        """
        if lines is None:
            lines = self.full_df["line_id"].to_list()

        # Restrictions
        self.net.restrict_to_output_subset(lines)

        # Use the appropriate names for the network
        self._check_parameters(df_params)

        df_params_net = df_params.rename(
            columns={"Av": "Avmax", "G0": "radm", "Pth": "P", "angle": "angle"}
        )

        # Conversion from G0 to radm
        df_params_net["radm"] = g0_to_radm(df_params_net["radm"])

        # Reorder inputs
        df_params_net = df_params_net[self.net.inputs_names]

        # PDR code predictions
        Y = 10 ** self.net.evaluate(df_params_net.values, transform_inputs=True)

        # Unit conversion
        freqs = (
            1e9 * self.full_df[self.full_df["line_id"].isin(lines)]["freq"].to_numpy()
        )
        if self.kelvin:
            Y = erg_to_kelvin(Y, freqs)

        # Apply kappa
        _kappa = kappa.get(Y.shape[0]).reshape(-1, 1)

        df = pd.DataFrame(
            np.hstack((df_params.values, _kappa, _kappa * Y)),
            columns=df_params.columns.to_list()
            + ["kappa"]
            + self.net.current_output_subset,
        )
        return df

    @staticmethod
    def _get_table() -> pd.DataFrame:
        # Reference dataframe
        full_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "full_table.csv")
        ).set_index("line_id")

        # Remove lines whose frequency is not available
        full_df = full_df.dropna(axis=0)

        return full_df

    @staticmethod
    def all_lines() -> List[str]:
        full_df = MeudonPDR._get_table()
        return full_df.index.to_list()

    @staticmethod
    def frequencies(lines: List[str]) -> Dict[str, float]:
        if not isinstance(lines, (List, Tuple)):
            raise TypeError(f"lines must be a list, not {type(lines)}")
        full_df = MeudonPDR._get_table()
        return {line: full_df.loc[line, "freq"] for line in lines}
