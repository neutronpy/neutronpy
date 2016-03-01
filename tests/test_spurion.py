r'''Tests spurion search

'''
import unittest
from neutronpy import spurion


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


if __name__ == "__main__":
    unittest.main()
