from neutronpy import resolution
import numpy as np
import unittest


def angle2(x, y, z, h, k, l, lattice):
    [V, Vstar, latticestar] = resolution._star(lattice)
    
    return np.arccos(2 * np.pi * (h * x + k * y + l * z) / resolution._modvec([x, y, z], lattice) / resolution._modvec([h, k, l], latticestar))


def SqwDemo(H, K, L, W, p):
    Deltax = p[0]
    Deltay = p[1]
    Deltaz = p[2]
    cc = p[3] 
    Gamma = p[4]

    omegax = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltax ** 2)
    omegay = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltay ** 2)
    omegaz = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltaz ** 2)
    
    lorx = 1 / np.pi * Gamma / ((W - omegax) ** 2 + Gamma ** 2)
    lory = 1 / np.pi * Gamma / ((W - omegay) ** 2 + Gamma ** 2)
    lorz = 1 / np.pi * Gamma / ((W - omegaz) ** 2 + Gamma ** 2)
    
    sqw0 = lorx * (1 - np.cos(np.pi * H)) / omegax / 2
    sqw1 = lory * (1 - np.cos(np.pi * H)) / omegay / 2
    sqw2 = lorz * (1 - np.cos(np.pi * H)) / omegaz / 2
    
    sqw = np.vstack((sqw0, sqw1, sqw2))
    
    return sqw


def SMADemo(H, K, L, p):
    Deltax = p[0]
    Deltay = p[1]
    Deltaz = p[2]
    cc = p[3] 
    Gamma = p[4]

    omegax = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltax ** 2)
    omegay = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltay ** 2)
    omegaz = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltaz ** 2)
    w0 = np.vstack((omegax, omegay, omegaz))

    S = np.vstack(((1. - np.cos(np.pi * H.flatten())) / omegax / 2.,
                   (1. - np.cos(np.pi * H.flatten())) / omegay / 2.,
                   (1. - np.cos(np.pi * H.flatten())) / omegaz / 2.))

    HWHM = np.ones(S.shape) * Gamma
    
    return [w0, S, HWHM]


def PrefDemo(H, K, L, EXP, p):    
    [sample, rsample] = resolution._GetLattice(EXP)
    
    q2 = resolution._modvec([H, K, L], rsample) ** 2
    
    sd = q2 / (16 * np.pi ** 2)
    ff = 0.0163 * np.exp(-35.883 * sd) + 0.3916 * np.exp(-13.223 * sd) + 0.6052 * np.exp(-4.339 * sd) - 0.0133
    
    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)
    
    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2
    
    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.*polx * p[5]
    prefactor[1, :] = ff ** 2.*poly * p[5]
    prefactor[2, :] = ff ** 2.*polz * p[5]

    bgr = np.ones(H.shape) * p[6]

    return [prefactor, bgr]


class ResolutionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ResolutionTest, self).__init__(*args, **kwargs)
        
        self.sumI11 = 2551.51194680064
        self.sumI12 = 2466.94158939588
        self.sumI13 = 2480.92389914159
        self.sumI14 = 2477.64921252374
        self.sumI15 = 2468.68169056761
        
    def test_cooper_nathans(self):
        R0 = 1799.58795290577
        RM = np.array([[6612.20619533415, 1.19238109481724e-11, 1.61862681767924e-12, 0],
                        [4.70281555083552e-11, 94579.1213987331, 11782.2595636634, 0],
                        [6.01386133616247e-12, 11782.2595636634, 1471.46842759222, 0],
                        [0, 0, 0, 634.724632931705]])
        ResVol0 = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(RM)) * (2. / 1.e-5)
        
        sample = resolution.Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = resolution.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')    

        hkle = [1., 1., 0., 0.]

        EXP.calc_resolution(hkle)

        NP = EXP.RMS[:, :, 0]
        R = EXP.R0[0]
        
        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP)) * (2. / 1.e-5)
        

        self.assertAlmostEqual(np.sum(NP), np.sum(RM), 6)
        self.assertAlmostEqual(R, R0, 6)
        self.assertAlmostEqual(ResVol, ResVol0, 6)

    def test_popovici(self):
        R0 = 1920.87188227816
        RMS = np.array([[7564.48470203361, 1.40133282636945e-11, 1.91484370090147e-12, 0],
                       [1.26749364029288e-10, 131741.994875255, 16416.2315817137, 0],
                       [1.59709520917375e-11, 16416.2315817137, 2050.87837885871, 0],
                       [0, 0, 0, 2705.71659797677]])
        ResVol0 = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(RMS)) * (2. / 1.e-5)
        
        sample = resolution.Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = resolution.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')
        EXP.arms = [150, 150, 150, 150]
        EXP.method = 1   

        hkle = [1., 1., 0., 0.]

        EXP.calc_resolution(hkle)

        NP = EXP.RMS[:, :, 0]
        R = EXP.R0[0]
        
        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP)) * (2. / 1.e-5)

        self.assertAlmostEqual(np.sum(NP), np.sum(RMS), 6)
        self.assertAlmostEqual(R, R0, 6)
        self.assertAlmostEqual(ResVol, ResVol0, 6)
        
    def test_4d_conv(self):            
        sample = resolution.Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = resolution.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')
                
        p = np.array([3, 3, 3, 30, 0.4, 6e4, 40])
        H1, K1, L1, W1 = 1.5, 0, 0.35, np.arange(20, -0.5, -0.5)
        
        I11 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'fix', [5, 0], p)
        I12 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'fix', [15, 0], p)
        I13 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'mc', None, p)
                
        sumI11, sumI12, sumI13 = np.sum(I11), np.sum(I12), np.sum(I13)
        
        print(np.abs(self.sumI11 - sumI11))
        self.assertTrue(np.abs(self.sumI11 - sumI11) < 25)
        self.assertTrue(np.abs(self.sumI12 - sumI12) < 25)
        self.assertTrue(np.abs(self.sumI13 - sumI13) < 50)
    
    def test_sma_conv(self):
        sample = resolution.Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = resolution.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')
                
        p = np.array([3, 3, 3, 30, 0.4, 6e4, 40])
        H1, K1, L1, W1 = 1.5, 0, 0.35, np.arange(20, -0.5, -0.5)
        
        I14 = EXP.resolution_convolution_SMA(SMADemo,PrefDemo,2,(H1,K1,L1,W1),'fix',[15,0],p)
        I15 = EXP.resolution_convolution_SMA(SMADemo,PrefDemo,2,(H1,K1,L1,W1),'mc',[1],p)
            
        sumI14, sumI15 = np.sum(I14), np.sum(I15)

        self.assertTrue(np.abs(self.sumI14 - sumI14) < 25)
        self.assertTrue(np.abs(self.sumI15 - sumI15) < 50)


if __name__ == '__main__':
    unittest.main()
