import unittest
from typing import List

import numpy as np
import pandas as pd

from infobs.sampling import Constant, Mixture


class TestMixtures(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(Exception):
            Mixture(None)
        with self.assertRaises(Exception):
            Mixture(Constant(0.0))
        with self.assertRaises(Exception):
            Mixture([Constant(0.0), Constant(1.0)], [1, 1, 1])

    def test_get_single(self):
        mxt = Mixture([Constant(0.0)])
        self.assertTrue((mxt.get(10) == 0.0).all())

    def test_get_weight(self):
        mxt = Mixture([Constant(0.0), Constant(1.0)], [1, 0])
        self.assertTrue((mxt.get(10) == 0.0).all())
