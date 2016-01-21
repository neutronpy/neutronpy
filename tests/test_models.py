from neutronpy import models
import unittest


class ModelTests(unittest.TestCase):
    def test_harmonic_oscillator(self):
        models.harmonic_oscillator()

    def test_acoustic_phonon(self):
        models.acoustic_phonon()

    def test_optical_phonon(self):
        models.optical_phonon()


if __name__ == '__main__':
    unittest.main()
