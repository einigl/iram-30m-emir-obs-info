import os
import sys
import unittest

sys.path.insert(0, os.path.join("../.."))

from infobs.util import util


class TestUtil(unittest.TestCase):
    def test_radm_to_g0(self):
        self.assertAlmostEqual(util.radm_to_g0(0.0), 0.0, delta=1e-8)
        self.assertAlmostEqual(util.radm_to_g0(1.0), 1.2786 / 2, delta=1e-8)

    def test_g0_to_radm(self):
        self.assertAlmostEqual(util.g0_to_radm(0.0), 0.0, delta=1e-8)
        self.assertAlmostEqual(util.g0_to_radm(1.0), 2 / 1.2786, delta=1e-8)

        x = 13542.12355
        self.assertAlmostEqual(util.g0_to_radm(util.radm_to_g0(x)), x, delta=1e-8)

    def test_kelvin_to_erg(self):
        self.assertAlmostEqual(util.kelvin_to_erg(0.0, 1.0), 0.0, delta=1e-8)

    def test_erg_to_kelvin(self):
        self.assertAlmostEqual(util.erg_to_kelvin(0.0, 1.0), 0.0, delta=1e-8)

        x = 7895453.225
        nu = 123456.32232
        self.assertAlmostEqual(
            util.kelvin_to_erg(util.erg_to_kelvin(x, nu), nu), x, delta=1e-8
        )

    def test_integrate_noise(self):
        rsm = 123.0
        self.assertAlmostEqual(util.integrate_noise(rsm, 1, 1.0), rsm, delta=1e-8)
        self.assertAlmostEqual(
            util.integrate_noise(rsm, 100, 1.0), 10 * rsm, delta=1e-8
        )
        self.assertAlmostEqual(util.integrate_noise(rsm, 1, 4.0), 4.0 * rsm, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
