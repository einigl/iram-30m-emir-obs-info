import unittest

import numpy as np

from infobs.sampling import samplers


class TestConstant(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 10
        cls.smp = samplers.Constant(0.0)

    def test_get(self):
        self.assertListEqual(self.smp.get(self.n).tolist(), np.zeros(self.n).tolist())

    def test_copy_other_bounds(self):
        new_smp = self.smp.copy_other_bounds(1.0)
        self.assertListEqual(new_smp.get(self.n).tolist(), np.ones(self.n).tolist())

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestUniform(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 10
        cls.smp = samplers.Uniform(1, 2)

    def test_get(self):
        x = self.smp.get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestLogUniform(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 10
        cls.smp = samplers.LogUniform(1, 2)

    def test_get(self):
        x = self.smp.get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestBoundedPowerLaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 10
        alpha = 2
        cls.smp = samplers.BoundedPowerLaw(alpha, 1, 2)

    def test_get(self):
        x = self.smp.get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(self.n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)
