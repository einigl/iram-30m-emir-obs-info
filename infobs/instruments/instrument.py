from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .. import util
from ..model import MeudonPDR
from ..util import kelvin_to_erg
from ..util.ism_lines_helpers import filter_molecules, molecules_among_lines

__all__ = ["Instrument", "StandardInstrument", "IdealInstrument", "MergedInstrument"]


class Instrument(ABC):
    """abstract class for instruments, used to generate synthetic observations with the instrument noise properties"""

    def __init__(self, lines: List[str], kelvin: bool = True):
        assert isinstance(lines, list)
        assert len(set(lines)) == len(lines)  # Verifies that lines has no duplicates
        assert isinstance(kelvin, bool)

        self.lines = lines
        self.kelvin = kelvin
        self.freqs = MeudonPDR.frequencies(lines)

    @abstractmethod
    def measure(self, df: pd.DataFrame, obstime: float) -> pd.DataFrame:
        """adds realistic noise to noise-free observation, using the properties of the instrument

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of noise-free observations
        obstime : float
            integration time

        Returns
        -------
        pd.DataFrame
            dataframe of noisy observations
        """
        pass

    def filter_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, df.columns.isin(MeudonPDR.parameters + self.lines)]

    def restrict_lines(self, lines: List[str]) -> None:
        assert set(lines) <= set(self.lines)
        assert isinstance(lines, list)
        assert len(set(lines)) == len(lines)  # Verifies that lines has no duplicates
        self.lines = lines

    def _split_dataframe(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        filt = df.columns.isin(MeudonPDR.parameters)
        df_params = df.loc[:, filt]
        df_lines = df.loc[:, ~filt]

        line_names = df_lines.columns.to_list()
        if not (set(line_names) <= set(self.lines)):
            raise ValueError(
                f"DataFrame contains lines that are not available for this instrument: {list(set(line_names) & set(self.lines))}"
            )

        return df_params, df_lines

    def available_molecules(self) -> List[str]:
        return molecules_among_lines(self.lines)

    def available_lines(
        self, molecules: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        if molecules is None:
            return self.lines.copy()
        return filter_molecules(self.lines, molecules)

    @abstractmethod
    def __str__(self):
        pass


class StandardInstrument(Instrument, ABC):
    """abstract class for standard instruments, used to generate synthetic observations with the instrument noise properties"""

    def __init__(
        self, lines: List[str], linewidth: Optional[float] = None, kelvin: bool = True
    ):
        if (linewidth is None) != (self.dv is None):
            raise ValueError(
                "self.dv and linewidth must be None or not None simultaneously"
            )
        self.linewidth = linewidth

        super().__init__(lines, kelvin)

    @property
    @abstractmethod
    def dv(self) -> Optional[float]:
        """Reference velocity channel width"""
        pass

    @property
    @abstractmethod
    def ref_obstime(self) -> Dict[str, float]:
        """Reference integration time [s]"""
        pass

    @property
    @abstractmethod
    def rms(self) -> Dict[str, float]:
        """Noise RMS [K] for each line"""
        pass

    @property
    @abstractmethod
    def percent(self) -> Dict[str, float]:
        """Calibration error [%] for each line"""
        pass

    def get_rms(
        self, obstime: float, lines: Optional[List[str]] = None
    ) -> Dict[str, float]:
        if isinstance(lines, str):
            lines = [lines]
        lines = lines or self.lines

        rms = np.array([self.rms[line] for line in lines])
        ref_obstime = np.array([self.ref_obstime[line] for line in lines])

        rms = rms / (obstime / ref_obstime) ** 0.5
        rms = util.integrate_noise(rms, self.linewidth / self.dv, self.dv)

        # Unit conversion
        freqs = 1e9 * np.array([self.freqs[l] for l in lines])
        if not self.kelvin:
            rms = kelvin_to_erg(rms, freqs)

        return {l: v for l, v in zip(lines, rms.tolist())}

    def measure(self, df: pd.DataFrame, obstime: float) -> pd.DataFrame:
        df_params, df_lines = self._split_dataframe(df)

        rms_dict = self.get_rms(obstime)
        percent = np.array([self.percent[line] for line in df_lines.columns])

        sigma_a = np.array(
            [rms_dict[l] for l in df_lines.columns.to_list()]
        )  # Atmospheric noise
        sigma_m = np.log(1 + percent / 100)  # Calibration error

        eps_a = np.random.normal(loc=0.0, scale=sigma_a, size=df_lines.shape)
        eps_m = np.random.lognormal(
            mean=-(sigma_m**2) / 2, sigma=sigma_m, size=df_lines.shape
        )

        y = eps_m * df_lines.values + eps_a

        df_lines_noise = pd.DataFrame(y, columns=df_lines.columns)
        return pd.concat([df_params, df_lines_noise], axis=1)

    def __str__(self):
        return "User-defined instrument"


class IdealInstrument(Instrument):
    """Ideal instrument without atmospheric noise and calibration error."""

    def __init__(self, kelvin: bool = True):
        super().__init__(MeudonPDR.all_lines(), kelvin)

    def measure(self, df: pd.DataFrame, _: Optional[float] = None) -> pd.DataFrame:
        self._split_dataframe(df)  # Only to check lines validity
        return df.copy()

    def __str__(self):
        return "Ideal instrument"


class MergedInstrument(Instrument):
    """combinations of multiple instruments"""

    def __init__(self, instruments: List[Instrument]):
        assert len(instruments) != 0
        self.instruments = instruments

        kelvin = instruments[0].kelvin
        assert all([instru.kelvin == kelvin for instru in instruments])

        lines = []
        for instru in instruments:
            intersect = list(set(lines) & set(instru.lines))
            if len(intersect) != 0:
                raise ValueError(
                    f"Instrument {instru} contains lines that are already handled by previous instrument: {intersect}"
                )
            lines.extend(instru.lines)

        super().__init__(lines, kelvin)

    def measure(
        self, df: pd.DataFrame, obstime: Union[float, List[float]]
    ) -> pd.DataFrame:
        if isinstance(obstime, (float, int)):
            obstime = [obstime] * len(self.instruments)
        elif len(obstime) != len(self.instruments):
            raise ValueError(
                "Number of integration times must be the same that the number of instruments"
            )

        df_params, df_lines = self._split_dataframe(df)

        df_list = [
            instru.measure(instru.filter_lines(df_lines), t)
            for instru, t in zip(self.instruments, obstime)
        ]
        df_out = pd.concat([df_params] + df_list, axis=1)
        return df_out[df.columns.to_list()]  # Reorder columns

    def __str__(self):
        return "MergedInstrument: " + ", ".join(
            [str(instru) for instru in self.instruments]
        )
