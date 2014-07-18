from neutronpy import tools
from numpy import testing


def test_energy():
    '''Test the Neutron class, which takes property of a
    neutron (such as energy or something related to energy)
    and provides conversions
    '''
    energy = tools.Neutron(e=25.)

    testing.assert_almost_equal(energy.e, 25.0, 4)
    testing.assert_almost_equal(energy.l, 1.8089, 4)
    testing.assert_almost_equal(energy.k, 3.473, 3)
    testing.assert_almost_equal(energy.v, 2187., 0)
    testing.assert_almost_equal(energy.temp, 290.113, 3)
    testing.assert_almost_equal(energy.freq, 6.045, 3)


if __name__ == '__main__':
    test_energy()
