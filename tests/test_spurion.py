from neutronpy import spurion
import unittest


class SpurionTest(unittest.TestCase):
    def test_aluminum_rings(self):
        spurion.aluminum()
    

if __name__ == "__main__":
    unittest.main()