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
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0007.bt7')))
            load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')))
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
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0001.dat')) == 'SPICE')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0003.ng5')) == 'ICP')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0004.bt7')) == 'ICE')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')) == 'MAD')
        self.assertRaises(ValueError, detect_filetype, os.path.join(os.path.dirname(__file__), 'filetypes/scan0006.test'))


if __name__ == '__main__':
    unittest.main()
