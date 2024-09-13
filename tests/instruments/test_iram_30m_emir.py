import unittest
from typing import List

import pandas as pd

from infobs.instruments import IRAM30mEMIR


class TestStandardInstrument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instru_ref = IRAM30mEMIR(linewidth=1)
        cls.instru_linewidth = IRAM30mEMIR(linewidth=4)
        cls.instru_erg = IRAM30mEMIR(linewidth=1, kelvin=False)
        cls.instru_ipwv = IRAM30mEMIR(linewidth=1, ipwv=0)

        cls.line = "13c_o_j1__j0"
        cls.df = pd.DataFrame(
            {
                "Av": [1, 5, 10],
                "G0": [1e2, 1e2, 1e2],
                "Pth": [1e5, 1e5, 1e5],
                "angle": [0, 0, 0],
                cls.line: [1e-7, 1e-6, 1e-5],
            }
        )

    def test_init(self):
        with self.assertRaises(Exception):
            IRAM30mEMIR(None)
        with self.assertRaises(Exception):
            IRAM30mEMIR(1, "kelvin")
        with self.assertRaises(Exception):
            IRAM30mEMIR(1, ipwv=None)

    def test_get_rms(self):
        line = self.line
        d_ref = self.instru_ref.get_rms(1, line)

        d_linewidth = self.instru_linewidth.get_rms(1, line)
        self.assertAlmostEqual(d_linewidth[line], 2 * d_ref[line])

        d_ipwv = self.instru_ipwv.get_rms(1, line)
        self.assertGreater(d_ref[line], d_ipwv[line])

        d_erg = self.instru_erg.get_rms(1, line)
        self.assertNotEqual(d_erg[line], d_ref[line])

    def test_measure(self):
        df1 = self.instru_ref.measure(self.df[[self.line]], 1.0)
        df2 = self.instru_ref.measure(self.df[[self.line]], 1.0)
        self.assertFalse(df1.loc[:, self.line].equals(df2[[self.line]]))

    def test_measure_param(self):
        df1 = self.instru_ref.measure(self.df, 1.0)
        df2 = self.instru_ref.measure(self.df, 1.0)
        self.assertTrue(set(df1.columns) == {"Av", "G0", "Pth", "angle", self.line})
        self.assertFalse(df1.loc[:, self.line].equals(df2.loc[:, self.line]))

    def test_get_bands(self):
        lines = ["13c_o_j1__j0", "13c_o_j2__j1", "13c_o_j3__j2"]
        bands = IRAM30mEMIR.get_bands(lines)
        self.assertIsInstance(bands, List)
        self.assertTrue(len(bands) == len(lines))

    def test_plot_band_single(self):
        self.instru_ref.plot_band(
            "3mm", ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"]
        )
        self.instru_ref.plot_band(
            "3mm", ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"], obstime=60
        )
        self.instru_ref.plot_band(
            "3mm", ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"], min_gap=2
        )

    def test_plot_band(self):
        self.instru_ref.plot_band(
            "all", ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"]
        )
        self.instru_ref.plot_band(
            "all", ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"], obstime=60
        )
