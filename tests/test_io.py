'''Tests for FileIO

'''
import os
import unittest
from mock import patch
import numpy as np
from neutronpy import functions
from neutronpy.fileio import load_data, save_data, detect_filetype
from neutronpy import Data


def build_data(clean=True):
    '''Builds data object
    '''
    p = np.array([20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2])
    x = np.linspace(-1, 1, 81)

    if clean:
        y = functions.voigt(p, x)
        mon = 1e5
        tim = 15
    else:
        y = functions.voigt(p, x) + np.random.normal(loc=0., scale=5, size=len(x))
        mon = 1e3
        tim = 5

    output = Data(Q=np.vstack((item.ravel() for item in np.meshgrid(x, 0., 0., 0., 300.))).T,
                  detector=y, monitor=np.full(x.shape, mon, dtype=float), time=np.full(x.shape, tim, dtype=float))

    return output


class IOTests(unittest.TestCase):
    '''Unit tests for fileIO
    '''
    @patch('sys.stdout')
    def test_load_files(self, mock_stdout):
        '''Tests file loading
        '''
        try:
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0001.dat'),
                       os.path.join(os.path.dirname(__file__), 'filetypes/scan0002.dat')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0003.ng5')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0004.bt7')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0007.bt7')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/000000.nxs')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/000001.dat')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.iexy')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.spe')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.xyie')))

        except:
            self.fail('Data loading failed')

        self.assertRaises(KeyError, load_data, (os.path.join(os.path.dirname(__file__), 'filetypes/scan0006.test')), filetype='blah')

    def test_save_file(self):
        '''Tests file saving
        '''
        data_out = build_data()
        save_data(data_out, 'test.out', fileformat='ascii')
        save_data(data_out, 'test.out', fileformat='hdf5')
        save_data(data_out, 'test.out', fileformat='pickle')
        self.assertRaises(ValueError, save_data, data_out, 'test.out', fileformat='blah')

    def test_filetype_detection(self):
        '''Test filetype detection
        '''
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0001.dat')) == 'spice')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0003.ng5')) == 'icp')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0004.bt7')) == 'ice')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')) == 'mad')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.iexy')) == 'dcs_mslice')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/000001.dat')) == 'grasp')
        self.assertRaises(ValueError, detect_filetype, os.path.join(os.path.dirname(__file__), 'filetypes/scan0006.test'))


if __name__ == '__main__':
    unittest.main()
