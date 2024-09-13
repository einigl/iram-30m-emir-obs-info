import unittest

from infobs import Infobs
from infobs.instruments import IdealInstrument
from infobs.model import MeudonPDR
from infobs.sampling import Constant, LogUniform


class TestInfobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samplers = {
            "Av": LogUniform(*MeudonPDR.bounds["Av"]),
            "G0": LogUniform(*MeudonPDR.bounds["G0"]),
            "Pth": LogUniform(*MeudonPDR.bounds["Pth"]),
            "angle": Constant(0.0),
        }
        cls.infobs = Infobs(IdealInstrument(), 1_000, samplers, obstime=60)

    def test_init(self):
        pass

    def test_plot_info_map(self):
        pass
