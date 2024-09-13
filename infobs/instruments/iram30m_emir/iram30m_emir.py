import os
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

# from ...graphics import latex
from ...model import MeudonPDR
from ..instrument import StandardInstrument
from . import _display

__all__ = ["IRAM30mEMIR"]


class IRAM30mEMIR(StandardInstrument):
    def __init__(
        self, linewidth: float, kelvin: bool = True, ipwv: Literal[0, 1, 2] = 1
    ):
        """
        Reference velocity channels : 0.5 km/s
        Integrated precipitable water vapor [mm]
        Linewidth [km/s]
        """
        ipwv = int(ipwv)
        assert ipwv in [0, 1, 2]
        self.ipwv = ipwv

        self.emir_df = IRAM30mEMIR._get_table()[
            [
                "EMIR band",
                "freq",
                "Calibration error (%)",
                f"Noise RMS (K) [1 min, IPWV={ipwv}mm]",
            ]
        ].rename(
            columns={f"Noise RMS (K) [1 min, IPWV={ipwv}mm]": "Noise RMS (K) [1 min]"}
        )
        lines = self.emir_df.index.to_list()

        super().__init__(lines, linewidth, kelvin)

    @property
    def ref_obstime(self) -> Dict[str, float]:
        return {line: 60 for line in self.lines}  # Seconds

    @property
    def dv(self) -> float:
        return 0.5  # km/s

    @property
    def rms(self) -> Dict[str, float]:
        return self.emir_df["Noise RMS (K) [1 min]"].to_dict()  # Kelvins

    @property
    def percent(self) -> Dict[str, float]:
        return self.emir_df["Calibration error (%)"].to_dict()  # Percents

    @staticmethod
    def _get_table():
        emir_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "emir_table_noise.csv")
        ).set_index("line_id")

        return emir_df

    @staticmethod
    def all_lines():
        emir_df = IRAM30mEMIR._get_table()
        return emir_df.index.to_list()

    @staticmethod
    def bands():
        """
        3mm (E090), 2mm (E150), 1mm (E230), 0.9mm (E330)
        """
        return {
            "3mm": [73, 117],  # GHz
            "2mm": [125, 184],
            "1mm": [202, 274],
            "0.9mm": [277, 375],  # [277, 350]
        }

    @staticmethod
    def get_bands(lines: Optional[Union[str, List[str]]] = None):
        """
        TODO
        """
        single = isinstance(lines, str)
        if single:
            lines = [lines]
        lines = lines or IRAM30mEMIR.all_lines()

        assert set(lines) <= set(IRAM30mEMIR.all_lines())

        emir_df = IRAM30mEMIR._get_table()
        freqs = emir_df.loc[lines, "freq"].to_list()
        emir_bands = IRAM30mEMIR.bands()

        bands = [None] * len(lines)
        for i, f in enumerate(freqs):
            for b, (low, upp) in emir_bands.items():
                if low <= f <= upp:
                    bands[i] = b
                    break

        return bands[0] if single else bands

    def plot_band(
        self,
        band: Literal["3mm", "2mm", "1mm", "0.9mm", "all"],
        lines: List[str],
        obstime: Optional[float] = None,
        transitions: bool = True,
        rotation: int = 60,
        rms_min: Optional[float] = None,
        rms_max: Optional[float] = None,
        min_gap: Optional[float] = None,
        global_offset: Optional[float] = None,
        fontsize=12,
        lines_fontsize=10,
        legend_fontsize=10,
    ):
        band = band.lower()
        assert band in ["3mm", "2mm", "1mm", "0.9mm", "all"]

        emir_lines = IRAM30mEMIR.all_lines()
        emir_freqs = MeudonPDR.frequencies(lines)

        lines = list(set(lines))  # Remove duplicates
        assert set(lines) <= set(
            emir_lines
        )  # Check that lines are observable with EMIR

        # lines = [latex.remove_hyperfine(l) for l in lines]

        if band == "all":
            return _display.plot_all_bands(
                lines,
                emir_freqs,
                obstime,
                self.ipwv,
                rms_min,
                rms_max,
                fontsize,
                legend_fontsize,
            )
        return _display.plot_specific_band(
            band,
            lines,
            emir_freqs,
            obstime,
            self.ipwv,
            transitions,
            rotation,
            rms_min,
            rms_max,
            min_gap,
            global_offset,
            fontsize,
            lines_fontsize,
            legend_fontsize,
        )

    def __str__(self):
        return "IRAM 30m EMIR"
