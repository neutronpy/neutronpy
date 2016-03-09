# -*- coding: utf-8 -*-
r'''File IO
'''
import numbers
import re
import numpy as np
from .data import Data
from .energy import Energy
from .instrument import Instrument
from .sample import Sample


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


def load_instrument(parfile, cfgfile):
    r'''Creates Instrument class using input par and cfg files.

    Parameters
    ----------
    parfile : str
        Path to the .par file

    cfgfile : str
        Path to the .cfg file

    Returns
    -------
    setup : obj
        Returns Instrument class object based on the information in the input
        files.

    Notes
    -----
    The format of the ``parfile`` consists of two tab-separated columns, the first
    column containing the values and the second column containing the value
    names preceded by a '%' character:

    +-------+---------+---------------------------------------------------------------------------------+
    | Type  | Name    | Description                                                                     |
    +=======+=========+=================================================================================+
    | float | %DM     | Monochromater d-spacing (Ang^-1)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %DA     | Analyzer d-spacing (Ang^-1)                                                     |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAM   | Monochromator mosaic (arc min)                                                  |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAA   | Analyzer mosaic (arc min)                                                       |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAS   | Sample mosaic (arc min)                                                         |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SM     | Scattering direction of monochromator (+1 clockwise, -1 counterclockwise)       |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SS     | Scattering direction of sample (+1 clockwise, -1 counterclockwise)              |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SA     | Scattering direction of analyzer (+1 clockwise, -1 counterclockwise)            |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %K      | Fixed wavevector (incident or final) of neutrons                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA1 | Horizontal collimation of in-pile collimator (arc min)                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA2 | Horizontal collimation of collimator between monochromator and sample (arc min) |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA3 | Horizontal collimation of collimator between sample and analyzer (arc min)      |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA4 | Horizontal collimation of collimator between analyzer and detector (arc min)    |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA1  | Vertical collimation of in-pile collimator (arc min)                            |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA2  | Vertical collimation of collimator between monochromator and sample (arc min)   |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA3  | Vertical collimation of collimator between sample and analyzer (arc min)        |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA4  | Vertical collimation of collimator between analyzer and detector (arc min)      |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AS     | Sample lattice constant a (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BS     | Sample lattice constant b (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %CS     | Sample lattice constant c (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AA     | Sample lattice angle alpha (deg)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BB     | Sample lattice angle beta (deg)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %CC     | Sample lattice angle gamma (deg)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AX     | Sample orientation vector u_x (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AY     | Sample orientation vector u_y (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AZ     | Sample orientation vector u_z (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BX     | Sample orientation vector v_x (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BY     | Sample orientation vector v_y (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BZ     | Sample orientation vector v_z (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QX     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QY     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QZ     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %EN     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqx    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqy    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqz    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %de     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gh     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gk     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gl     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gmod   |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+

    The format of the ``cfgfile`` (containing values necessary for Popovici type
    calculations) can consists of a single column of values, or two
    tab-separated columns, the first column containing the values and the
    second column containing the value descriptions preceded by a '%' character.
    The values MUST be in the following order:

    +-------+-------------------------------------------------------+
    | Type  | Description                                           |
    +=======+=======================================================+
    | float | =0 for circular source, =1 for rectangular source     |
    +-------+-------------------------------------------------------+
    | float | width/diameter of the source (cm)                     |
    +-------+-------------------------------------------------------+
    | float | height/diameter of the source (cm)                    |
    +-------+-------------------------------------------------------+
    | float | =0 No Guide, =1 for Guide                             |
    +-------+-------------------------------------------------------+
    | float | horizontal guide divergence (minutes/Angs)            |
    +-------+-------------------------------------------------------+
    | float | vertical guide divergence (minutes/Angs)              |
    +-------+-------------------------------------------------------+
    | float | =0 for cylindrical sample, =1 for cuboid sample       |
    +-------+-------------------------------------------------------+
    | float | sample width/diameter perp. to Q (cm)                 |
    +-------+-------------------------------------------------------+
    | float | sample width/diameter along Q (cm)                    |
    +-------+-------------------------------------------------------+
    | float | sample height (cm)                                    |
    +-------+-------------------------------------------------------+
    | float | =0 for circular detector, =1 for rectangular detector |
    +-------+-------------------------------------------------------+
    | float | width/diameter of the detector (cm)                   |
    +-------+-------------------------------------------------------+
    | float | height/diameter of the detector (cm)                  |
    +-------+-------------------------------------------------------+
    | float | thickness of monochromator (cm)                       |
    +-------+-------------------------------------------------------+
    | float | width of monochromator (cm)                           |
    +-------+-------------------------------------------------------+
    | float | height of monochromator (cm)                          |
    +-------+-------------------------------------------------------+
    | float | thickness of analyser (cm)                            |
    +-------+-------------------------------------------------------+
    | float | width of analyser (cm)                                |
    +-------+-------------------------------------------------------+
    | float | height of analyser (cm)                               |
    +-------+-------------------------------------------------------+
    | float | distance between source and monochromator (cm)        |
    +-------+-------------------------------------------------------+
    | float | distance between monochromator and sample (cm)        |
    +-------+-------------------------------------------------------+
    | float | distance between sample and analyser (cm)             |
    +-------+-------------------------------------------------------+
    | float | distance between analyser and detector (cm)           |
    +-------+-------------------------------------------------------+
    | float | horizontal curvature of monochromator 1/radius (cm-1) |
    +-------+-------------------------------------------------------+
    | float | vertical curvature of monochromator (cm-1) was 0.013  |
    +-------+-------------------------------------------------------+
    | float | horizontal curvature of analyser (cm-1) was 0.078     |
    +-------+-------------------------------------------------------+
    | float | vertical curvature of analyser (cm-1)                 |
    +-------+-------------------------------------------------------+
    | float | distance monochromator-monitor                        |
    +-------+-------------------------------------------------------+
    | float | width monitor (cm)                                    |
    +-------+-------------------------------------------------------+
    | float | height monitor (cm)                                   |
    +-------+-------------------------------------------------------+

    '''
    with open(parfile, "r") as f:
        lines = f.readlines()
        par = {}
        for line in lines:
            rows = line.split()
            par[rows[1][1:].lower()] = float(rows[0])

    with open(cfgfile, "r") as f:
        lines = f.readlines()
        cfg = []
        for line in lines:
            rows = line.split()
            cfg.append(float(rows[0]))

    if par['sm'] == par['ss']:
        dir1 = -1
    else:
        dir1 = 1

    if par['ss'] == par['sa']:
        dir2 = -1
    else:
        dir2 = 1

    if par['kfix'] == 2:
        infin = -1
    else:
        infin = par['kfix']

    hcol = [par['alpha1'], par['alpha2'], par['alpha3'], par['alpha4']]
    vcol = [par['beta1'], par['beta2'], par['beta3'], par['beta4']]

    nsou = cfg[0]  # =0 for circular source, =1 for rectangular source.
    if nsou == 0:
        ysrc = cfg[1] / 4  # width/diameter of the source [cm].
        zsrc = cfg[2] / 4  # height/diameter of the source [cm].
    else:
        ysrc = cfg[1] / np.sqrt(12)  # width/diameter of the source [cm].
        zsrc = cfg[2] / np.sqrt(12)  # height/diameter of the source [cm].

    flag_guide = cfg[3]  # =0 for no guide, =1 for guide.
    guide_h = cfg[4]  # horizontal guide divergence [mins/Angs]
    guide_v = cfg[5]  # vertical guide divergence [mins/Angs]
    if flag_guide == 1:
        alpha_guide = np.pi / 60. / 180. * 2 * np.pi * guide_h / par['k']
        alpha0 = hcol[0] * np.pi / 60. / 180.
        if alpha_guide <= alpha0:
            hcol[0] = 2. * np.pi / par['k'] * guide_h
        beta_guide = np.pi / 60. / 180. * 2 * np.pi * guide_v / par['k']
        beta0 = vcol[0] * np.pi / 60. / 180.
        if beta_guide <= beta0:
            vcol[0] = 2. * np.pi / par['k'] * guide_v

    nsam = cfg[6]  # =0 for cylindrical sample, =1 for cuboid sample.
    if nsam == 0:
        xsam = cfg[7] / 4  # sample width/diameter perp. to Q [cm].
        ysam = cfg[8] / 4  # sample width/diameter along Q [cm].
        zsam = cfg[9] / 4  # sample height [cm].
    else:
        xsam = cfg[7] / np.sqrt(12)  # sample width/diameter perp. to Q [cm].
        ysam = cfg[8] / np.sqrt(12)  # sample width/diameter along Q [cm].
        zsam = cfg[9] / np.sqrt(12)  # sample height [cm].

    ndet = cfg[10]  # =0 for circular detector, =1 for rectangular detector.
    if ndet == 0:
        ydet = cfg[11] / 4  # width/diameter of the detector [cm].
        zdet = cfg[12] / 4  # height/diameter of the detector [cm].
    else:
        ydet = cfg[11] / np.sqrt(12)  # width/diameter of the detector [cm].
        zdet = cfg[12] / np.sqrt(12)  # height/diameter of the detector [cm].

    xmon = cfg[13]  # thickness of monochromator [cm].
    ymon = cfg[14]  # width of monochromator [cm].
    zmon = cfg[15]  # height of monochromator [cm].

    xana = cfg[16]  # thickness of analyser [cm].
    yana = cfg[17]  # width of analyser [cm].
    zana = cfg[18]  # height of analyser [cm].

    L0 = cfg[19]  # distance between source and monochromator [cm].
    L1 = cfg[20]  # distance between monochromator and sample [cm].
    L2 = cfg[21]  # distance between sample and analyser [cm].
    L3 = cfg[22]  # distance between analyser and detector [cm].

    romh = par['sm'] * cfg[23]  # horizontal curvature of monochromator 1/radius [cm-1].
    romv = par['sm'] * cfg[24]  # vertical curvature of monochromator [cm-1].
    roah = par['sa'] * cfg[25]  # horizontal curvature of analyser [cm-1].
    roav = par['sa'] * cfg[26]  # vertical curvature of analyser [cm-1].
    inv_rads = [romh, romv, roah, roav]
    for n, inv_rad in enumerate(inv_rads):
        if inv_rad == 0:
            inv_rads[n] = 1.e6
        else:
            inv_rads[n] = 1. / inv_rad
    [romh, romv, roah, roav] = inv_rads

    L1mon = cfg[27]  # distance monochromator monitor [cm]
    monitorw = cfg[28] / np.sqrt(12)  # monitor width [cm]
    monitorh = cfg[29] / np.sqrt(12)  # monitor height [cm]

    # -------------------------------------------------------------------------

    energy = Energy(wavevector=par['k'])

    sample = Sample(par['as'], par['bs'], par['cs'],
                    par['aa'], par['bb'], par['cc'],
                    par['etas'])
    sample.u = [par['ax'], par['ay'], par['az']]
    sample.v = [par['bx'], par['by'], par['bz']]
    sample.shape = np.diag([xsam, ysam, zsam])

    setup = Instrument(energy.energy, sample, hcol, vcol,
                       2 * np.pi / par['dm'], par['etam'],
                       2 * np.pi / par['da'], par['etaa'])

    setup.method = 1
    setup.dir1 = dir1
    setup.dir2 = dir2
    setup.mondir = par['sm']
    setup.infin = infin
    setup.arms = [L0, L1, L2, L3, L1mon]
    setup.guide.width = ysrc
    setup.guide.height = zsrc

    setup.detector.width = ydet
    setup.detector.height = zdet

    setup.mono.depth = xmon
    setup.mono.width = ymon
    setup.mono.height = zmon
    setup.mono.rv = romv
    setup.mono.rh = romh

    setup.ana.depth = xana
    setup.ana.width = yana
    setup.ana.height = zana
    setup.ana.rv = roav
    setup.ana.rh = roah

    setup.monitor.width = monitorw
    setup.monitor.height = monitorh

    return setup


def save_instrument(instrument, filename):
    r'''Saves an instrument configuration into par and cfg files for loading
    with `load_instrument`

    '''
    pass
