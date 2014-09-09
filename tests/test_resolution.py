from neutronpy import resolution
import numpy as np
import unittest


class ResolutionTest(unittest.TestCase):
    def test_cooper_nathans(self):
        ResVol0 = 3.59321990862
        R0 = 1.42573667
        RM = np.array([[23867.70517271, 20739.82283639, 0., 1427.28556717],
                       [20739.82283639, 22311.93077865, 0., 746.61192804],
                       [0., 0., 1049.13250378, 0.],
                       [1427.28556717, 746.61192804, 0., 187.09685691]])

        sample = resolution.Sample(3.81, 3.81, 6.25, 90, 90, 90, 70.)
        sample.u = [1, 0, 0]
        sample.v = [0, 1, 0]

        EXP = resolution.Instrument(5., sample, hcol=[32, 80, 120, 120], mono='pg(002)', ana='pg(002)')

        EXP.arms = np.array([150, 150, 150, 150, 105])
        EXP.infin = 1
        EXP.horifoc = -1

        hkle = [1., 1., 0., 0.]

        EXP.calc_resolution([[hkle[0], 0.5], [hkle[1], 0.5], hkle[2], hkle[3]])

        NP = EXP.RMS[:, :, 0]
        R = EXP.R0[0]
        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP)) * (2. / 1.e-5)

        self.assertAlmostEqual(np.sum(NP), np.sum(RM), 6)
        self.assertAlmostEqual(R, R0, 6)
        self.assertAlmostEqual(ResVol, ResVol0, 6)

    def popovici_test(self):
        pass


if __name__ == '__main__':
    unittest.main()
