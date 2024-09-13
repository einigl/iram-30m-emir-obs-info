import unittest

import pandas as pd

from infobs.instruments import IdealInstrument, MergedInstrument, StandardInstrument


class Instru1(StandardInstrument):
    def __init__(self, linewidth: float, kelvin: bool = True):
        super().__init__(
            ["c_el3p_j1__el3p_j0", "c_el3p_j2__el3p_j1"], linewidth, kelvin
        )

    @property
    def dv(self):
        return 1.0

    @property
    def ref_obstime(self):
        return {"c_el3p_j1__el3p_j0": 1, "c_el3p_j2__el3p_j1": 1}

    @property
    def rms(self):
        return {"c_el3p_j1__el3p_j0": 0.5, "c_el3p_j2__el3p_j1": 0.5}

    @property
    def percent(self):
        return {"c_el3p_j1__el3p_j0": 20, "c_el3p_j2__el3p_j1": 20}


class Instru2(StandardInstrument):
    def __init__(self, linewidth: float, kelvin: bool = True):
        super().__init__(["cp_el2p_j3_2__el2p_j1_2"], linewidth, kelvin)

    @property
    def dv(self):
        return 0.193

    @property
    def ref_obstime(self):
        return {"cp_el2p_j3_2__el2p_j1_2": 1}

    @property
    def rms(self):
        return {"cp_el2p_j3_2__el2p_j1_2": 2.25}

    @property
    def percent(self):
        return {"cp_el2p_j3_2__el2p_j1_2": 5}


class TestStandardInstrument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instru_ref = Instru1(linewidth=1)
        cls.instru_linewidth = Instru1(linewidth=4)
        cls.instru_erg = Instru1(linewidth=1, kelvin=False)

    def test_init(self):
        with self.assertRaises(Exception):
            self.Instru(None)
        with self.assertRaises(Exception):
            self.Instru(1, "kelvin")

    def test_get_rms(self):
        # TODO
        pass

    def test_measure(self):
        df = pd.DataFrame(
            {
                "c_el3p_j1__el3p_j0": [1e-6, 1e-7, 1e-8],
                "c_el3p_j2__el3p_j1": [1e-7, 1e-8, 1e-9],
            }
        )
        df_out = self.instru_ref.measure(df, 60)
        self.assertEqual(len(df), len(df_out))
        self.assertEqual(list(df_out.columns), list(df.columns))

    def test_measure_param(self):
        df = pd.DataFrame(
            {
                "Av": [1, 5, 10],
                "c_el3p_j1__el3p_j0": [1e-6, 1e-7, 1e-8],
                "c_el3p_j2__el3p_j1": [1e-7, 1e-8, 1e-9],
            }
        )
        df_out = self.instru_ref.measure(df, 60)
        self.assertEqual(len(df), len(df_out))
        self.assertEqual(list(df_out.columns), list(df.columns))
        self.assertTrue(df["Av"].equals(df_out["Av"]))

    def test_measure_erg(self):
        df = pd.DataFrame(
            {
                "c_el3p_j1__el3p_j0": [1e-6, 1e-7, 1e-8],
                "c_el3p_j2__el3p_j1": [1e-7, 1e-8, 1e-9],
            }
        )
        df_out = self.instru_erg.measure(df, 60)
        self.assertEqual(len(df), len(df_out))
        self.assertEqual(list(df_out.columns), list(df.columns))


class TestIdealInstrument(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(Exception):
            IdealInstrument("kelvin")

    def test_measure(self):
        instru = IdealInstrument()
        df = pd.DataFrame(
            {
                "13c_o_j1__j0": [1e-6, 1e-7, 1e-8],
                "c_18o_j1__j0": [1e-7, 1e-8, 1e-9],
            }
        )
        self.assertTrue(instru.measure(df).equals(df))


class TestMergedInstrument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instru = MergedInstrument([Instru1(10), Instru2(10)])

    def test_init(self):
        with self.assertRaises(Exception):
            MergedInstrument([Instru1(10, kelvin=True), Instru2(10, kelvin=False)])
        with self.assertRaises(Exception):
            MergedInstrument([Instru1(10), IdealInstrument(10)])

    def test_measure(self):
        df = pd.DataFrame(
            {
                "cp_el2p_j3_2__el2p_j1_2": [1e-6, 1e-7, 1e-8],
                "c_el3p_j1__el3p_j0": [1e-7, 1e-8, 1e-9],
            }
        )
        df_out = self.instru.measure(df, 60)
        self.assertEqual(len(df), len(df_out))
        self.assertEqual(list(df_out.columns), list(df.columns))

    def test_measure_obstime(self):
        df = pd.DataFrame(
            {
                "cp_el2p_j3_2__el2p_j1_2": [1e-6, 1e-7, 1e-8],
                "c_el3p_j1__el3p_j0": [1e-7, 1e-8, 1e-9],
            }
        )
        df_out = self.instru.measure(df, [60, 30])
        self.assertEqual(len(df), len(df_out))
        self.assertEqual(list(df_out.columns), list(df.columns))
