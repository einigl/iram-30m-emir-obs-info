import unittest

from infobs import Simulator
from infobs.instruments import IdealInstrument
from infobs.sampling import LogUniform, Uniform


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = Simulator(IdealInstrument())
        cls.samplers = {
            "Av": LogUniform(1, 40),
            "G0": LogUniform(1, 1e4),
            "Pth": LogUniform(1e5, 1e9),
            "angle": Uniform(0, 60),
        }

    def test_params_sampling(self):
        df_params = self.simulator.params_sampling(10, self.samplers)
        self.assertTrue(len(df_params) == 10)
        self.assertTrue(df_params["Av"].between(1, 40).all())
        self.assertTrue(df_params["G0"].between(1, 1e5).all())
        self.assertTrue(df_params["Pth"].between(1e5, 1e9).all())
        self.assertTrue(df_params["angle"].between(0, 60).all())

    def test_simulate(self):
        df = self.simulator.simulate(
            10, self.samplers, ["13c_o_j1__j0", "c_18o_j1__j0"]
        )
        self.assertTrue(len(df) == 10)
        self.assertTrue(
            set(df.columns)
            == {"Av", "G0", "Pth", "angle", "kappa", "13c_o_j1__j0", "c_18o_j1__j0"}
        )

        self.assertTrue(df["Av"].between(1, 40).all())
        self.assertTrue(df["G0"].between(1, 1e5).all())
        self.assertTrue(df["Pth"].between(1e5, 1e9).all())
        self.assertTrue(df["angle"].between(0, 60).all())
        self.assertTrue(df["kappa"].between(1, 1).all())
