import os
import unittest

import numpy as np
import pandas as pd

from infobs.graphics import PDRPlotter


class TestPDRPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plotter = PDRPlotter()

    @classmethod
    def tearDownClass(cls):
        path = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(path, "profiles.csv")
        output_dir = os.path.join(path, "out-profiles")
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
            os.rmdir(output_dir)

        filename = os.path.join(path, "slices.csv")
        output_dir = os.path.join(path, "out-slices")
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
            os.rmdir(output_dir)

    def test_plot_profile(self):
        self.plotter.plot_profile(
            ["co_v0_j1__v0_j0", "13c_o_j1__j0", "c_18o_j1__j0"],
            G0=1e2,
            Pth=1e5,
            angle=0,
            n_samples=10,
        )

    def test_plot_slice(self):
        self.plotter.plot_slice("13c_o_j1__j0", Pth=1e5, angle=0, n_samples=5)

    def test_save_profiles_from_csv(self):
        path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(path, "profiles.csv")
        output_dir = os.path.join(path, "out-profiles")
        df = pd.DataFrame(
            {
                "Av": [np.nan, 1, 1, 1],
                "G0": [1e2, np.nan, 1e2, 1e2],
                "Pth": [1e5, 1e5, np.nan, 1e5],
                "angle": [0, 0, 0, np.nan],
                "lines": [
                    "13c_o_j1__j0 c_18o_j1__j0",
                    "13c_o_j1__j0 c_18o_j1__j0",
                    "13c_o_j1__j0 c_18o_j1__j0",
                    "13c_o_j1__j0 c_18o_j1__j0",
                ],
            }
        )
        df.to_csv(filename)

        self.plotter.save_profiles_from_csv(filename, output_dir, n_samples=10)
        self.assertTrue(len(os.listdir(output_dir)) == len(df))

    def test_save_slices_from_csv(self):
        path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(path, "slices.csv")
        output_dir = os.path.join(path, "out-slices")
        df = pd.DataFrame(
            {
                "Av": [np.nan, np.nan, 1],
                "G0": [np.nan, 1e2, np.nan],
                "Pth": [1e5, np.nan, np.nan],
                "angle": [0, 0, 0],
                "lines": [
                    "13c_o_j1__j0 c_18o_j1__j0",
                    "13c_o_j1__j0 c_18o_j1__j0",
                    "13c_o_j1__j0 c_18o_j1__j0",
                ],
            }
        )
        df.to_csv(filename)

        self.plotter.save_slices_from_csv(filename, output_dir, n_samples=5)
        self.assertTrue(len(os.listdir(output_dir)) == len(df))
