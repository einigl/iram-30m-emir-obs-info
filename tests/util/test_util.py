import unittest
import os
import sys

sys.path.insert(0, os.path.join("../.."))

from infobs.util import util

class TestUtil(unittest.TestCase):

    def test_radm_to_g0(self):
        self.assertEqual(util.radm_to_g0(0.), 0.)
        self.assertEqual(util.radm_to_g0(1.), 1.2786 / 2)

    def test_g0_to_radm(self):
        self.assertEqual(util.g0_to_radm(0.), 0.)
        self.assertEqual(util.g0_to_radm(1.), 2 / 1.2786 )

        x = 13542.12355
        self.assertEqual(util.g0_to_radm(util.radm_to_g0(x)), x)


    def test_kelvin_to_erg(self):
        self.assertEqual(util.kelvin_to_erg(0., 1.), 0.)

    def test_erg_to_kelvin(self):
        self.assertEqual(util.erg_to_kelvin(0., 1.), 0.)

        x = 7895453.225
        nu = 123456.32232
        self.assertEqual(util.kelvin_to_erg(util.erg_to_kelvin(x, nu), nu), x)

    def test_integrate_noise(self):
        rsm = 123.0
        self.assertEqual(util.integrate_noise(rsm, 1, 1.0), rsm)
        self.assertEqual(util.integrate_noise(rsm, 100, 1.0), 10 * rsm)
        self.assertEqual(util.integrate_noise(rsm, 1, 4.0), 2.0 * rsm)

if __name__ == '__main__':
    unittest.main()
