import unittest

import numpy as np

from infobs.sampling import samplers

n = 10


class TestConstant(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.smp = samplers.Constant(0.0)

    def test_get(self):
        self.assertListEqual(self.smp.get(n).tolist(), np.zeros(n).tolist())

    def test_copy_other_bounds(self):
        new_smp = self.smp.copy_other_bounds(1.0)
        self.assertListEqual(new_smp.get(n).tolist(), np.ones(n).tolist())

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestUniform(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.smp = samplers.Uniform(1, 2)

    def test_get(self):
        x = self.smp.get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestLogUniform(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.smp = samplers.LogUniform(1, 2)

    def test_get(self):
        x = self.smp.get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


class TestBoundedPowerLaw(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        alpha = 2
        self.smp = samplers.BoundedPowerLaw(alpha, 1, 2)

    def test_get(self):
        x = self.smp.get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 2)

    def test_copy_other_bounds(self):
        x = self.smp.copy_other_bounds(1, 10).get(n)
        self.assertGreaterEqual(np.min(x), 1)
        self.assertLessEqual(np.max(x), 10)

    def test_str(self):
        self.assertIsInstance(str(self.smp), str)


if __name__ == "__main__":
    unittest.main()
