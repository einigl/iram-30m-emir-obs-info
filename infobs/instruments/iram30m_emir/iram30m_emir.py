import os
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from ..instrument import StandardInstrument

__all__ = [
    "IRAM30mEMIR"
]

class IRAM30mEMIR(StandardInstrument):

    def __init__(
        self,
        linewidth: float,
        kelvin: bool=True
    ):
        """
        Reference velocity channels : 0.5 km/s.
        Linewidth [km/s]
        """

        emir_df = pd.read_csv(os.path.join(
            os.path.dirname(__file__), "emir_table_noise.csv"
        ))
        self.emir_df = emir_df.set_index("line_id")
        
        lines = self.emir_df.index.to_list()
        super().__init__(
            lines,
            linewidth,
            kelvin
        )

    @property
    def ref_obstime(self) -> Dict[str, float]:
        return {line: 60 for line in self.lines} # Seconds

    @property
    def dv(self) -> float:
        return 0.5 # km/s

    @property
    def rms(self) -> Dict[str, float]:
        return self.emir_df["Noise RMS (K) [1 min]"].to_dict() # Kelvins
            
    @property
    def percent(self) -> Dict[str, float]:
        return self.emir_df["Calibration error (%)"].to_dict() # Percents


    @staticmethod
    def bands():
        return {
            "E090": [73, 117], # GHz
            "E150": [125, 184],
            "E230": [202, 274],
            "E330": [277, 375] # [277, 350]
        }
    
    from ._display import plot_bands

    def __str__(self):
        return "IRAM 30m EMIR"
