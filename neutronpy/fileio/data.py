# -*- coding: utf-8 -*-
import numbers
import re
import numpy as np
from ..data import Data


def build_Q(args, **kwargs):
    u'''Method for constructing **Q**\ (*q*, ℏω, temp) from h, k, l,
    energy, and temperature

    Parameters
    ----------
    args : dict
        A dictionary of the `h`, `k`, `l`, `e` and `temp` arrays to form into
        a column oriented array

    Returns
    -------
    Q : ndarray
        Returns **Q**\ (h, k, l, e, temp) with shape (N, 5) in a column
        oriented array.

    '''
    return np.vstack((args[i].flatten() for i in
                      ['h', 'k', 'l', 'e', 'temp'])).T


def load_data(files, filetype='auto', tols=1e-4):
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
    if isinstance(files, str):
        files = (files,)

    if isinstance(tols, numbers.Number):
        tols = [tols for i in range(5)]

    for filename in files:
        if filetype == 'auto':
            try:
                filetype = detect_filetype(filename)
            except ValueError:
                raise

        if filetype == 'SPICE':
            data_keys = {'monitor': 'monitor', 'detector': 'detector',
                         'time': 'time'}
            Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'tvti'}
            raw_data = {}

            with open(filename) as f:
                for line in f:
                    if 'col_headers' in line:
                        args = next(f).split()
                        headers = [head for head in args[1:]]

            args = np.genfromtxt(filename, unpack=True, dtype=np.float64)

            _t0 = 60.

        elif filetype == 'ICE':
            data_keys = {'detector': 'Detector', 'monitor': 'Monitor',
                         'time': 'Time'}
            Q_keys = {'h': 'QX', 'k': 'QY', 'l': 'QZ', 'e': 'E',
                      'temp': 'Temp'}
            raw_data = {}
            _t0 = 60.

            with open(filename) as f:
                for line in f:
                    if 'Columns' in line:
                        args = line.split()
                        headers = [head for head in args[1:]]
                        break

            args = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                                 unpack=True, comments="#", dtype=np.float64)

        elif filetype == 'ICP':
            data_keys = {'detector': 'Counts', 'time': 'min'}
            Q_keys = {'h': 'Q(x)', 'k': 'Q(y)', 'l': 'Q(z)', 'e': 'E',
                      'temp': 'T-act'}
            raw_data = {}
            _t0 = 1.

            with open(filename) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        _length = int(re.findall(r"(?='(.*?)')", line)[-2])
                        [_m0, _prf] = [float(i) for i in re.findall(r"(?='(.*?)')", line)[-4].split()]
                    if 'Q(x)' in line:
                        args = line.split()
                        headers = [head for head in args]
                        break

            args = np.genfromtxt(filename, unpack=True, dtype=np.float64, skip_header=12)

            raw_data['monitor'] = np.empty(args[0].shape)
            raw_data['monitor'].fill(_m0 * _prf)

        elif filetype == 'MAD':
            data_keys = {'detector': 'CNTS', 'time': 'TIME', 'monitor': 'M1'}
            Q_keys = {'h': 'QH', 'k': 'QK', 'l': 'QL', 'e': 'EN', 'temp': 'TT'}
            raw_data = {}
            _t0 = 60

            with open(filename) as f:
                for i, line in enumerate(f):
                    if 'DATA_:' in line:
                        args = next(f).split()
                        headers = [head for head in args]
                        skip_lines = i + 2
                        break

            args = np.genfromtxt(filename, unpack=True, dtype=np.float64,
                                 skip_header=skip_lines, skip_footer=1)

        else:
            raise ValueError('Filetype not supported.')

        for key, value in data_keys.items():
            try:
                raw_data[key] = args[headers.index(value)]
            except ValueError:
                print("ValueError: '{0}' is not in list.".format(value))
                raw_data[key] = np.ones(args[0].shape)

        _Q_dict = {}
        for key, value in Q_keys.items():
            try:
                _Q_dict[key] = args[headers.index(value)]
            except ValueError:
                print("ValueError: '{0}' is not in list.".format(value))
                _Q_dict[key] = np.ones(args[0].shape)

        raw_data['time'] /= _t0
        raw_data['Q'] = build_Q(_Q_dict)

        del _Q_dict, args

        try:
            _data_object.combine_data(raw_data, tols=tols)  # @UndefinedVariable
        except (NameError, UnboundLocalError):
            _data_object = Data(**raw_data)

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


def detect_filetype(file):
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
    if file[-3:] == 'nxs':
        return 'nexus'
    elif file[-4:] == 'iexy':
        return 'iexy'
    else:
        with open(file) as f:
            first_line = f.readline()
            second_line = f.readline()
            if '#ICE' in first_line:
                return 'ICE'
            elif '# scan' in first_line:
                return 'SPICE'
            elif 'Filename' in second_line:
                return 'ICP'
            elif 'RRR' in first_line or 'AAA' in first_line or 'VVV' in first_line:
                return 'MAD'
            else:
                raise ValueError('Unknown filetype.')
