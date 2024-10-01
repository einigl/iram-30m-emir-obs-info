import unittest

import pandas as pd

from infobs.graphics import InfoPlotter
from infobs.graphics.latex import latex_line, latex_param

class TestInfoPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plotter = InfoPlotter(
            latex_line, latex_param
        )

    def test_plot_prob_bar(self):
        pass # TODO
