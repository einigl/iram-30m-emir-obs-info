import unittest
from typing import List

import numpy as np
import pandas as pd

from infobs.model import MeudonPDR


class TestMeudonPDR(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            MeudonPDR("kelvin")

    def test_predict(self):
        meudonpdr = MeudonPDR()
        df_params = pd.DataFrame(
            {
                "Av": [1.0, 1.0, 5.0],
                "G0": [1e2, 1e2, 1e2],
                "Pth": [1e5, 1e5, 1e5],
                "angle": [0.0, 0.0, 0.0],
            }
        )

        df = meudonpdr.predict(df_params)
        self.assertEqual(df.shape, (3, 5 + len(meudonpdr.all_lines())))
        self.assertTrue(np.isclose(df.iloc[0].values, df.iloc[1].values).all())
        self.assertFalse(np.isclose(df.iloc[0].values, df.iloc[2].values).all())
        self.assertTrue((df.loc[:, "kappa"] == 1.0).all())

        lines = ["13c_o_j1__j0"]
        subdf = meudonpdr.predict(df_params, lines)
        self.assertEqual(subdf.shape, (3, 6))
        self.assertTrue(
            np.isclose(df.loc[:, lines].values, subdf.loc[:, lines].values).all()
        )

    def test_predict_kappa(self):
        meudonpdr = MeudonPDR(kelvin=False)
        # TODO

    def test_predict_erg(self):
        meudonpdr = MeudonPDR(kelvin=False)
        # TODO

    def test_predict_warning(self):
        df_params = pd.DataFrame(
            {
                "Av": [50.0],
                "G0": [1e2],
                "Pth": [1e5],
                "angle": [0.0],
            }
        )

        meudonpdr = MeudonPDR()
        with self.assertWarns(Warning):
            meudonpdr.predict(df_params)

    def test_static(self):
        lines = MeudonPDR.all_lines()
        self.assertIsInstance(lines, List)
        self.assertTrue(all([isinstance(l, str) for l in lines]))
        self.assertTrue(len(lines) == len(set(lines)))

        freqs = MeudonPDR.frequencies(lines)
        self.assertTrue(set(freqs.keys()) == set(lines))

        with self.assertRaises(TypeError):
            MeudonPDR.frequencies(lines[0])
        subfreqs = MeudonPDR.frequencies(lines[0:1])
        self.assertTrue(len(subfreqs) == 1 and subfreqs[lines[0]] == freqs[lines[0]])
