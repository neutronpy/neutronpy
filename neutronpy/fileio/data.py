# -*- coding: utf-8 -*-
import copy
import numbers
import numpy as np
from .loaders import DcsMslice, Grasp, Icp, Ice, Mad, Spice


def load_data(files, filetype='auto', tols=1e-4, build_hkl=True, load_instrument=False):
    r'''Loads one or more files and creates a :class:`Data` object with the
    loaded data.

    Parameters
    ----------
    files : str or tuple of str
        A file or non-keyworded list of files containing data for input.

    filetype : str, optional
        Default: `'auto'`. Specify file type; Currently supported file types
        are SPICE, ICE, and ICP. By default, the function will attempt to
        determine the filetype automatically.

    tols : float or array_like
        Default: `1e-4`. A float or array of shape `(5,)` giving tolerances
        for combining multiple files. If multiple points are within the given
        tolerances then they will be combined into a single point. If a float
        is given, tolerances will all be the same for all variables in **Q**.
        If an array is given tolerances should be in the format
        `[h, k, l, e, temp]`.

    Returns
    -------
    Data : object
        A :class:`Data` object populated with the data from the input file or
        files.

    '''
    load_filetype = {'dcs_mslice': DcsMslice,
                     'grasp': Grasp,
                     'ice': Ice,
                     'icp': Icp,
                     'mad': Mad,
                     'spice': Spice}

    if isinstance(files, str):
        files = (files,)

    for filename in files:
        if filetype == 'auto':
            try:
                filetype = detect_filetype(filename)
            except ValueError:
                raise

        try:
            _data_object_temp = load_filetype[filetype.lower()]()
            _data_object_temp.load(filename, build_hkl=build_hkl, load_instrument=load_instrument)
        except KeyError:
            raise KeyError('Filetype not supported.')

        print()
        if isinstance(tols, numbers.Number):
            tols = [tols for i in range(len(_data_object_temp._data) - len(_data_object_temp.data_keys))]

        try:
            _data_object.combine_data(_data_object_temp, tols=tols)
        except (NameError, UnboundLocalError):
            _data_object = copy.deepcopy(_data_object_temp)

    return _data_object


def save_data(obj, filename, fileformat='ascii', **kwargs):
    '''Saves a given object to a file in a specified format.

    Parameters
    ----------
    obj : :class:`Data`
        A :class:`Data` object to be saved to disk

    filename : str
        Path to file where data will be saved

    fileformat : str
        Default: `'ascii'`. Data can either be saved in `'ascii'`,
        human-readable format, binary `'hdf5'` format, or binary
        `'pickle'` format.
    '''
    output = np.hstack((obj.Q, obj.detector.reshape(obj.detector.shape[0], 1),
                        obj.monitor.reshape(obj.monitor.shape[0], 1),
                        obj.time.reshape(obj.time.shape[0], 1)))

    if fileformat == 'ascii':
        np.savetxt(filename, output, **kwargs)
    elif fileformat == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('data', output.shape,
                                    maxshape=(None, output.shape[1]),
                                    dtype='float64')
            dset = output
    elif fileformat == 'pickle':
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(output, f)
    else:
        raise ValueError("""Format not supported. Please use 'ascii', 'hdf5', or 'pickle'""")


def detect_filetype(filename):
    u'''Simple method for quickly determining filetype of a given input file.

    Parameters
    ----------
    file : str
        File path

    Returns
    -------
    filetype : str
        The filetype of the given input file
    '''
    if filename[-3:] == 'nxs':
        return 'grasp'
    elif filename[-4:].lower() == 'iexy' or filename[-3:].lower() == 'spe' or filename[-3:].lower() == 'xye' or filename[-4:] == 'xyie':
        return 'dcs_mslice'
    else:
        with open(filename) as f:
            first_line = f.readline()
            second_line = f.readline()
            if '#ICE' in first_line:
                return 'ice'
            elif '# scan' in first_line:
                return 'spice'
            elif 'GRASP' in first_line.upper():
                return 'grasp'
            elif 'Filename' in second_line:
                return 'icp'
            elif 'RRR' in first_line or 'AAA' in first_line or 'VVV' in first_line:
                return 'mad'
            else:
                raise ValueError('Unknown filetype.')
