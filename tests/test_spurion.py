r'''Tests spurion search

'''
import unittest
import warnings
from mock import patch
from neutronpy import spurion
from neutronpy import Instrument


class SpurionTest(unittest.TestCase):
    '''Unit test Spurions
    '''
    def test_aluminum_rings(self):
        '''Check aluminum ring finder
        '''
        try:
            spurion.aluminum()
        except:
            self.fail('Aluminum ring finder failed')

    def test_currat_axe(self):
        with warnings.catch_warnings(record=True) as w:
            spurion.currat_axe_peaks(Instrument(), [[0.8, 0.8, 0], [1.2, 1.2, 0], 17], [[1, 1, 0]], angle_tol=1)
            self.assertTrue(len(w) == 3)

if __name__ == "__main__":
    unittest.main()
