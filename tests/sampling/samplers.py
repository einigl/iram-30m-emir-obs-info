import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join("../.."))

from infobs.sampling import samplers

n = 10


lower = 15478.5
lower_arr = lower * np.ones((n,))
constant_sampler = samplers.Constant(lower)

upper = lower + 100

new_lower = 1.2
new_constant_sampler = constant_sampler.copy_other_bounds(lower)

uniform_sampler = samplers.Uniform(lower, upper)
new_uniform_sampler = uniform_sampler.copy_other_bounds(new_lower, upper)





class TestConstant(unittest.TestCase):

    def test_get(self):
        self.assertEqual(np.max(np.abs(constant_sampler.get(n) - lower_arr)), 0.0)

    def test_copy_other_bounds(self):
        self.assertEqual(new_constant_sampler.value, new_lower)

    def test_str(self):
        self.assertEqual(constant_sampler.__str__(), f"Constant(value={lower})")
        self.assertEqual(new_constant_sampler.__str__(), f"Constant(value={new_lower})")


class TestUniform(unittest.TestCase):

    def test_get(self):
        x = uniform_sampler.get(n)
        self.assertEqual(np.max(np.abs(constant_sampler.get(n) - lower_arr)), 0.0)

    def test_copy_other_bounds(self):
        self.assertEqual(new_constant_sampler.value, new_lower)

    def test_str(self):
        self.assertEqual(constant_sampler.__str__(), f"Constant(value={lower})")
        self.assertEqual(new_constant_sampler.__str__(), f"Constant(value={new_lower})")


if __name__ == '__main__':
    unittest.main()
