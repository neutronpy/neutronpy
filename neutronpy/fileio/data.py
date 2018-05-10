# -*- coding: utf-8 -*-
import copy
import numbers

import numpy as np

from .instrument import save_instrument
from .loaders import DcsMslice, Grasp, Ice, Icp, Mad, Neutronpy, Spice


def load_data(files, filetype='auto', tols=1e-4, build_hkl=True, load_instrument=False):
    r"""Loads one or more files and creates a :class:`Data` object with the
    loaded data.

    Parameters
    ----------
    files : str or tuple of str
        A file or non-keyworded list of files containing data for input.

    filetype : str, optional
        Default: `'auto'`. Specify file type; Currently supported file types
        are `'SPICE'` (HFIR), `'ICE'` and `'ICP'` (NIST), `'MAD'` (ILL),
        `'dcs_mslice'` DAVE exported ascii formats, GRASP exported ascii and
        HDF5 formats, and neutronpy exported formats. By default the function
        will attempt to determine the filetype automatically.

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

    """
    load_filetype = {'dcs_mslice': DcsMslice,
                     'grasp': Grasp,
                     'ice': Ice,
                     'icp': Icp,
                     'mad': Mad,
                     'neutronpy': Neutronpy,
                     'spice': Spice}

    if not isinstance(files, (tuple, list)):
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

        if isinstance(tols, numbers.Number):
            tols = [tols for i in range(len(_data_object_temp._data) - len(_data_object_temp.data_keys))]

        try:
            _data_object.combine_data(_data_object_temp, tols=tols)
        except (NameError, UnboundLocalError):
            _data_object = copy.deepcopy(_data_object_temp)

    return _data_object


def save_data(obj, filename, filetype='ascii', save_instr=False, overwrite=False, **kwargs):
    """Saves a given object to a file in a specified format.

    Parameters
    ----------
    obj : :class:`Data`
        A :class:`Data` object to be saved to disk

    filename : str
        Path to file where data will be saved

    filetype : str, optional
        Default: `'ascii'`. Data can either be saved in human-readable
        `'ascii'` format, binary `'hdf5'` format, or binary `'pickle'`
        format (not recommended).

    save_instr : bool, optional
        Default: False.

    overwrite : bool, optional
        Default: False.

    """
    if filetype == 'ascii':
        from datetime import datetime

        if overwrite:
            mode = 'w+'
        else:
            mode = 'r+'

        header = '### NeutronPy ::: {0} ###\n\n'.format(datetime.now().isoformat())
        header += 'data_keys = {0}\n'.format(str(obj.data_keys))
        if hasattr(obj, 'Q_keys'):
            header += 'Q_keys = {0}\n'.format(str(obj.Q_keys))

        header += '\n\noriginal_header = \n\t'

        if hasattr(obj, 'file_header'):
            old_header = '\n\t'.join(obj.file_header)
        else:
            old_header = ''

        old_header += '\n\n'

        data_columns = obj.data_columns
        data = obj.data
        if hasattr(obj, '_err'):
            data_columns.append('error')
            data['error'] = obj._err

        col_header = '\nnpy_col_headers =\n' + '\t'.join(data_columns)
        header += old_header + col_header

        output = np.vstack((value for value in data.values())).T

        np.savetxt(filename + '.npy', output, header=header, **kwargs)

        if save_instr:
            save_instrument(obj.instrument, filename, filetype='ascii', overwrite=overwrite)

    elif filetype == 'hdf5':
        import h5py

        if overwrite:
            mode = 'w'
        else:
            mode = 'a'

        with h5py.File(filename + '.h5', mode) as f:
            data = f.create_group('data')

            try:
                data.attrs.create('file_header', obj.file_header.encode('utf8'))
            except AttributeError:
                pass

            data_keys = data.create_group('data_keys')
            for key, value in obj.data_keys.items():
                data_keys.attrs.create(key, value.encode('utf8'))

            if hasattr(obj, 'Q_keys'):
                Q_keys = data.create_group('Q_keys')
                for key, value in obj.Q_keys.items():
                    Q_keys.attrs.create(key, value.encode('utf8'))

            for key, value in obj.data.items():
                data.create_dataset(key, data=value)

            if hasattr(obj, '_err'):
                data.create_dataset('error', data=obj._err)

        if save_instr:
            try:
                save_instrument(obj.instrument, filename, filetype='hdf5', overwrite=False)
            except AttributeError:
                pass

    elif filetype == 'pickle':
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError("""Format not supported. Please use 'ascii', 'hdf5', or 'pickle'""")


def detect_filetype(filename):
    r"""Simple method for quickly determining filetype of a given input file.

    Parameters
    ----------
    filename : str
        File path

    Returns
    -------
    filetype : str
        The filetype of the given input file

    """
    if filename[-3:].lower() == 'nxs':
        return 'grasp'
    elif (filename[-4:].lower() == 'iexy') or (filename[-3:].lower() == 'spe') or (filename[-3:].lower() == 'xye') or (filename[-4:] == 'xyie'):
        return 'dcs_mslice'
    elif filename[-2:].lower() == 'h5' or filename[-3:].lower() == 'npy':
        return 'neutronpy'
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
            elif 'NeutronPy' in first_line:
                return 'neutronpy'
            else:
                raise ValueError('Unknown filetype.')
