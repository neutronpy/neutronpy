# -*- coding: utf-8 -*-
"""Tests for FileIO

"""
import os

import numpy as np
import pytest
from mock import patch
from neutronpy import Data, Instrument, functions
from neutronpy.fileio import (detect_filetype, load_data, load_instrument,
                              save_data, save_instrument)
from neutronpy.fileio.exceptions import DataIOError, InstrumentIOError


def build_data(clean=True):
    """Builds data object
    """
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


@patch('sys.stdout')
def test_load_data_files(mock_stdout):
    """Tests file loading
    """
    try:
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0001.dat'),
                   os.path.join(os.path.dirname(__file__), 'filetypes/scan0002.dat')), load_instrument=True)
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0003.ng5')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0004.bt7')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0007.bt7')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/000000.nxs')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/000001.dat')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.iexy')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.spe')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.xyie')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.npy')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.h5')))
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/test_save_load_spice.npy')), load_instrument=True)
        load_data(os.path.join(os.path.dirname(__file__), u'filetypes/scan0001º£.dat'))
    except:
        pytest.fail('Data loading failed')

    with pytest.raises(DataIOError):
        load_data((os.path.join(os.path.dirname(__file__), 'filetypes/scan0006.test')), filetype='blah')

    try:
        a = load_data(os.path.join(os.path.dirname(__file__), 'filetypes/scan0001a.dat'))
        h, k, l, e = a.h, a.k, a.l, a.e
    except:
        pytest.fail('Data loading failed (missing column problem)')

    with pytest.raises(KeyError):
        a = load_data(os.path.join(os.path.dirname(__file__), 'filetypes/scan0001a.dat'))
        temp = a.temp


def test_save_data_file():
    """Tests data object saving
    """
    data_out = build_data()

    try:
        save_data(data_out, 'test.out', filetype='ascii', overwrite=True)
        save_data(data_out, 'test.out', filetype='hdf5', overwrite=True)
        save_data(data_out, 'test.out', filetype='pickle', overwrite=True)
    except Exception:
        pytest.fail('Data saving failed')

    with pytest.raises(ValueError):
        save_data(data_out, 'test.out', filetype='hdf5', overwrite=False)
        save_data(data_out, 'test.out', filetype='blah', overwrite=True)


def test_load_instrument_file():
    """Tests instrument file loading
    """
    try:
        load_instrument((os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.par'),
                         os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.cfg')), filetype='parcfg')
        load_instrument(os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.instr'), filetype='ascii')
        load_instrument(os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.h5'), filetype='hdf5')
        load_instrument(os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.taz'), filetype='taz')
    except Exception:
        pytest.fail('Instrument file loading failed')

    with pytest.raises(InstrumentIOError):
        load_instrument(os.path.join(os.path.dirname(__file__), 'filetypes/test_instr.instr'), filetype='blah')


def test_save_instrument_file():
    """Tests instrument object saving
    """
    instr = Instrument()
    try:
        save_instrument(instr, 'test.out', filetype='ascii', overwrite=True)
        save_instrument(instr, 'test.out', filetype='hdf5', overwrite=True)
        save_instrument(instr, 'test.out', filetype='taz', overwrite=True)
    except Exception:
        pytest.fail('Instrument saving failed')

    with pytest.raises(ValueError):
        save_instrument(instr, 'test.out', filetype='hdf5', overwrite=False)


def test_filetype_detection():
    """Test filetype detection
    """
    assert (detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0001.dat')) == 'spice')
    assert (detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0003.ng5')) == 'icp')
    assert (detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0004.bt7')) == 'ice')
    assert (detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0005')) == 'mad')
    assert (
        detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/test_filetypes.iexy')) == 'dcs_mslice')
    assert (detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/000001.dat')) == 'grasp')
    with pytest.raises(DataIOError):
        detect_filetype(os.path.join(os.path.dirname(__file__), 'filetypes/scan0006.test'))


if __name__ == '__main__':
    pytest.main()
