# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag as blkdiag
from .core import Energy


def load(parfile, cfgfile):
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
                       par['dm'], par['etam'],
                       par['da'], par['etaa'],
                       method=1,
                       dir1=dir1, dir2=dir2, mondir=par['sm'], infin=infin,
                       arms=[L0, L1, L2, L3, L1mon])
    
    setup.beam.width = ysrc
    setup.beam.height = zsrc

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


class Sample():
    u'''Private class containing sample information.

    Parameters
    ----------
    a : float
        Unit cell length in angstroms

    b : float
        Unit cell length in angstroms

    c : float
        Unit cell length in angstroms

    alpha : float
        Angle between b and c in degrees

    beta : float
        Angle between a and c in degrees

    gamma : float
        Angle between a and b in degrees

    mosaic : float, optional
        Sample mosaic (FWHM) in arc minutes

    direct : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)
        
    u : array_like
        First orientation vector
        
    v : array_like
        Second orientation vector

    Returns
    -------
    Sample : object

    '''
    def __init__(self, a, b, c, alpha, beta, gamma, mosaic=None, direct=1, u=None, v=None):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if mosaic is not None:
            self.mosaic = mosaic
        self.dir = direct
        if u is not None:
            self.u = np.array(u)
        if v is not None:
            self.v = np.array(v)


class _Monochromator():
    u'''Private class containing monochromator information.

    Parameters
    ----------
    tau : float or string
        Tau value for the monochromator (or analyzer)

    mosaic : int
        Mosaic of the crystal in arc minutes

    dir : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)

    Returns
    -------
    Monochromator : class

    '''
    def __init__(self, tau, mosaic, direct=-1):
        self.tau = tau
        self.mosaic = mosaic
        self.dir = direct


class _dummy():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _scalar(v1, v2, lattice):
    r'''Calculates the _scalar product of two vectors, defined by their
    fractional cell coordinates or Miller indexes.

    Parameters
    ----------
    v1 : array
        First input vector

    v2 : array
        Second input vector

    lattice : Sample class
        Class containing unit cell parameters

    Returns
    -------
    s : _scalar
        The _scalar product of the two input vectors scaled by the lattice
        parameters.

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''

    [x1, y1, z1] = v1
    [x2, y2, z2] = v2

    s = x1 * x2 * lattice.a ** 2 + y1 * y2 * lattice.b ** 2 + z1 * z2 * lattice.c ** 2 + \
        (x1 * y2 + x2 * y1) * lattice.a * lattice.b * np.cos(lattice.gamma) + \
        (x1 * z2 + x2 * z1) * lattice.a * lattice.c * np.cos(lattice.beta) + \
        (z1 * y2 + z2 * y1) * lattice.c * lattice.b * np.cos(lattice.alpha)

    return s


def _star(lattice):
    r'''Given lattice parametrs, calculate unit cell volume V, reciprocal
    volume Vstar, and reciprocal lattice parameters.

    Parameters
    ----------
    lattice : Class
        Sample class with the lattice parameters

    Returns
    -------
    [V, Vstar, latticestar] : [float, float, class]
        Returns the unit cell volume, reciprocal cell volume, and a Sample
        Class with reciprocal lattice parameters

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    V = 2 * lattice.a * lattice.b * lattice.c * \
        np.sqrt(np.sin((lattice.alpha + lattice.beta + lattice.gamma) / 2) * 
                np.sin((-lattice.alpha + lattice.beta + lattice.gamma) / 2) * 
                np.sin((lattice.alpha - lattice.beta + lattice.gamma) / 2) * 
                np.sin((lattice.alpha + lattice.beta - lattice.gamma) / 2))

    Vstar = (2 * np.pi) ** 3 / V

    latticestar = Sample(0, 0, 0, 0, 0, 0)
    latticestar.a = 2 * np.pi * lattice.b * lattice.c * np.sin(lattice.alpha) / V
    latticestar.b = 2 * np.pi * lattice.a * lattice.c * np.sin(lattice.beta) / V
    latticestar.c = 2 * np.pi * lattice.b * lattice.a * np.sin(lattice.gamma) / V
    latticestar.alpha = np.arccos((np.cos(lattice.beta) * np.cos(lattice.gamma) - 
                                   np.cos(lattice.alpha)) / (np.sin(lattice.beta) * np.sin(lattice.gamma)))
    latticestar.beta = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.gamma) - 
                                  np.cos(lattice.beta)) / (np.sin(lattice.alpha) * np.sin(lattice.gamma)))
    latticestar.gamma = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.beta) - 
                                   np.cos(lattice.gamma)) / (np.sin(lattice.alpha) * np.sin(lattice.beta)))

    return [V, Vstar, latticestar]


def _modvec(v, lattice):
    r'''Calculates the modulus of a vector, defined by its fractional cell
    coordinates or Miller indexes.

    Parameters
    ----------
    v : array
        Input vector

    lattice : Sample class
        Class containing unit cell parameters

    Returns
    -------
    v : float
        Modulus of the input vector scaled by the sample lattice

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''

    return np.sqrt(_scalar(v, v, lattice))


def GetTau(x, getlabel=False):
    u'''τ-values for common monochromator and analyzer crystals.

    Parameters
    ----------
    x : float or string
        Either the numerical Tau value, in Å\ :sup:`-1`, or a
        common monochromater / analyzer type. Currently included crystals and their
        corresponding τ values are

            +------------------+--------------+-----------+
            | String           |     τ        |           |
            +==================+==============+===========+
            | Be(002)          | 3.50702      |           |
            +------------------+--------------+-----------+
            | Co0.92Fe0.08(200)| 3.54782      | (Heusler) |
            +------------------+--------------+-----------+
            | Cu(002)          | 3.47714      |           |
            +------------------+--------------+-----------+
            | Cu(111)          | 2.99913      |           |
            +------------------+--------------+-----------+
            | Cu(220)          | 4.91642      |           |
            +------------------+--------------+-----------+
            | Cu2MnAl(111)     | 1.82810      | (Heusler) |
            +------------------+--------------+-----------+
            | Ge(111)          | 1.92366      |           |
            +------------------+--------------+-----------+
            | Ge(220)          | 3.14131      |           |
            +------------------+--------------+-----------+
            | Ge(311)          | 3.68351      |           |
            +------------------+--------------+-----------+
            | Ge(511)          | 5.76968      |           |
            +------------------+--------------+-----------+
            | Ge(533)          | 7.28063      |           |
            +------------------+--------------+-----------+
            | PG(002)          | 1.87325      |           |
            +------------------+--------------+-----------+
            | PG(004)          | 3.74650      |           |
            +------------------+--------------+-----------+
            | PG(110)          | 5.49806      |           |
            +------------------+--------------+-----------+
            | Si(111)          | 2.00421      |           |
            +------------------+--------------+-----------+


    getlabel : boolean
        If True, return the name of the common crystal type that is a
        match to the input τ.

    Returns
    -------
    tau : float or string
        Returns either the numerical τ for a given crystal type or the
        name of a crystal type

    Notes
    -----
    Tau is defined as :math:`\tau = 2\pi/d`, where d is the d-spacing of the crystal in Angstroms.

    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    choices = {'pg(002)'.lower(): 1.87325,
               'pg(004)'.lower(): 3.74650,
               'ge(111)'.lower(): 1.92366,
               'ge(220)'.lower(): 3.14131,
               'ge(311)'.lower(): 3.68351,
               'be(002)'.lower(): 3.50702,
               'pg(110)'.lower(): 5.49806,
               'Cu2MnAl(111)'.lower(): 2 * np.pi / 3.437,
               'Co0.92Fe0.08(200)'.lower(): 2 * np.pi / 1.771,
               'Ge(511)'.lower(): 2 * np.pi / 1.089,
               'Ge(533)'.lower(): 2 * np.pi / 0.863,
               'Si(111)'.lower(): 2 * np.pi / 3.135,
               'Cu(111)'.lower(): 2 * np.pi / 2.087,
               'Cu(002)'.lower(): 2 * np.pi / 1.807,
               'Cu(220)'.lower(): 2 * np.pi / 1.278,
               'Cu(111)'.lower(): 2 * np.pi / 2.095}

    if getlabel:
        # return the index/label of the closest monochromator
        choices_ = dict((key, np.abs(value - x)) for (key, value) in choices.items)
        index = min(choices_, key=choices_.get)
        if choices[index] < 5e-4:
            tau = choices[index]  # the label
        else:
            tau = ''
    elif isinstance(x, (int, float)):
        tau = x
    else:
        try:
            tau = choices[x.lower()]
        except ValueError:
            raise ValueError('Invalid monochromator crystal type.')

    return tau


def _CleanArgs(*varargin):
    r'''Reshapes input arguments to be row-vectors. N is the length of the
    longest input argument. If any input arguments are shorter than N, their
    first values are replicated to produce vectors of length N. In any case,
    output arguments are row-vectors of length N.

    Parameters
    ----------
    varargin : tuple
        Converts arrays into formats appropriate for the calculation and
        extends arrays that are too short

    Returns
    -------
    [length, varargout] : [int, tuple]
        Returns the length of the input vectors and a tuple containing the cleaned vectors

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    varargout = []
    lengths = np.array([], dtype=np.int32)
    for arg in varargin:
        if type(arg) != list and not isinstance(arg, np.ndarray):
            arg = [arg]
        varargout.append(np.array(arg))
        lengths = np.concatenate((lengths, [len(arg)]))

    length = max(lengths)
    bad = np.where(lengths < length)
    if len(bad[0]) > 0:
        for i in bad[0]:
            varargout[i] = np.concatenate((varargout[i], [varargout[i][-1]] * int(length - lengths[i])))
            lengths[i] = len(varargout[i])

    if len(np.where(lengths < length)[0]) > 0:
        raise ValueError('Fatal error: All inputs must have the same lengths.')

    return [length] + varargout


def project_into_plane(index, r0, rm):
    r'''Projects out-of-plane resolution into a specified plane by performing
    a gaussian integral over the third axis.

    Parameters
    ----------
    rm : ndarray
        Resolution array

    index : int
        Index of the axis that should be integrated out

    Returns
    -------
    mp : ndarray
        Resolution matrix in a specified plane

    '''

    r = np.sqrt(2 * np.pi / rm[index, index]) * r0
    mp = rm

    b = rm[:, index] + rm[index, :].T
    b = np.delete(b, index, 0)

    mp = np.delete(mp, index, 0)
    mp = np.delete(mp, index, 1)

    mp -= 1 / (4. * rm[index, index]) * np.outer(b, b.T)

    return [r, mp]


def ellipse(saxis1, saxis2, phi=0, origin=None, npts=31):
    r'''Returns an ellipse.

    Parameters
    ----------
    saxis1 : float
        First semiaxis

    saxis2 : float
        Second semiaxis

    phi : float, optional
        Angle that semiaxes are rotated

    origin : list of floats, optional
        Origin position [x0, y0]

    npts: float, optional
        Number of points in the output arrays.

    Returns
    -------
    [x, y] : list of ndarray
        Two one dimensional arrays representing an ellipse
    '''

    if origin is None:
        origin = [0., 0.]

    theta = np.linspace(0., 2. * np.pi, npts)

    x = np.array(saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi) + origin[0])
    y = np.array(saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi) + origin[1])
    return np.vstack((x, y))


def _voigt(x, a):
    def _approx1(t): 
        return (t * 0.5641896) / (0.5 + t ** 2);
    
    
    def _approx2(t, u):
        return (t * (1.410474 + u * 0.5641896)) / (0.75 + (u * (3. + u)))
    
    
    def _approx3(t):
        return (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t)))) \
            / (16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t)))))
    
         
    def _approx4(t, u):
        return (t * (36183.31 - u * (3321.99 - u * (1540.787 - u * (219.031 - u * (35.7668 - u * (1.320522 - u * 0.56419)))))) \
                / (32066.6 - u * (24322.8 - u * (9022.23 - u * (2186.18 - u * (364.219 - u * (61.5704 - u * (1.84144 - u))))))))

    nx = x.size
    if len(a) == 1:
        a = np.ones(nx, dtype=np.complex64) * a
    y = np.zeros(nx, dtype=np.complex64)

    t = a - 1j * x
    ax = np.abs(x)
    s = ax + a
    u = t ** 2
        
    good = np.where(a == 0)
    y[good] = np.exp(-x[good] ** 2)
    
    good = np.where((a >= 15) | (s >= 15))
    y[good] = _approx1(t[good])
    
    good = np.where((s < 15) & (a < 15) & (a >= 5.5))
    y[good] = _approx2(t[good], u[good])   
    
    good = np.where((s < 15) & (s >= 5.5) & (a < 5.5))
    y[good] = _approx2(t[good], u[good])   
    
    good = np.where((s < 5.5) & (a < 5.5) & (a >= 0.75))
    y[good] = _approx3(t[good])
    
    good = np.where((s < 5.5) & (a >= 0.195 * ax - 0.176) & (a < 0.75))
    y[good] = _approx3(t[good])
    
    good = np.where((~((s < 5.5) & (a >= 0.195 * ax - 0.176))) & (a < 0.75))
    y[good] = np.exp(u[good]) - _approx4(t[good], u[good])
    
    y = np.real(y)
    return y


class Instrument(object):
    u'''An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    efixed : float, optional
        Fixed energy, either ei or ef, depending on the instrument
        configuration. Default: 14.7

    sample : obj, optional
        Sample lattice constants, parameters, mosaic, and orientation
        (reciprocal-space orienting vectors). Default: A crystal with
        a,b,c = 6,7,8 and alpha,beta,gamma = 90,90,90 and orientation
        vectors u=[1 0 0] and v=[0 1 0].

    hcol : list(4)
        Horizontal Soller collimations in minutes of arc starting from the
        neutron guide. Default: [40 40 40 40]

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the
        neutron guide. Default: [120 120 120 120]

    mono_tau : str or float, optional
        The monochromator reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common monochromator types.
        Default: 'PG(002)'

    mono_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    ana_tau : str or float, optional
        The analyzer reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common analyzer types.
        Default: 'PG(002)'

    ana_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    Attributes
    ----------
    method
    moncar
    mono
    ana
    hcol
    vcol
    arms
    efixed
    sample
    orient1
    orient2
    dir1
    dir2
    mondir
    infin
    beam
    detector
    monitor
    Smooth

    Methods
    -------
    calc_resolution
    calc_resolution_in_Q_coords
    calc_projections
    get_lattice
    get_resolution_params
    plot_projections
    resolution_convolution
    resolution_convolution_SMA
    
    '''

    def __init__(self, efixed=14.7, sample=None, hcol=None, vcol=None, mono='PG(002)',
                 mono_mosaic=25, ana='PG(002)', ana_mosaic=25, **kwargs):

        if sample is None:
            sample = Sample(6, 7, 8, 90, 90, 90)
            sample.u = [1, 0, 0]
            sample.v = [0, 1, 0]

        if hcol is None:
            hcol = [40, 40, 40, 40]

        if vcol is None:
            vcol = [120, 120, 120, 120]

        self.mono = _Monochromator(mono, mono_mosaic)
        self.ana = _Monochromator(ana, ana_mosaic)
        self.hcol = np.array(hcol)
        self.vcol = np.array(vcol)
        self.efixed = efixed
        self.sample = sample
        self.orient1 = np.array(sample.u)
        self.orient2 = np.array(sample.v)
        self.beam = _dummy()
        self.detector = _dummy()
        self.monitor = _dummy()

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mono(self):
        u'''A structure that describes the monochromator.
        
        Parameters
        ----------
        tau : str or float 
            The monochromator reciprocal lattice vector in Å\ :sup:`-1`. Instead of
            a numerical input one can use one of the following keyword strings:
            
                +------------------+--------------+-----------+
                | String           |     τ        |           |
                +==================+==============+===========+
                | Be(002)          | 3.50702      |           |
                +------------------+--------------+-----------+
                | Co0.92Fe0.08(200)| 3.54782      | (Heusler) |
                +------------------+--------------+-----------+
                | Cu(002)          | 3.47714      |           |
                +------------------+--------------+-----------+
                | Cu(111)          | 2.99913      |           |
                +------------------+--------------+-----------+
                | Cu(220)          | 4.91642      |           |
                +------------------+--------------+-----------+
                | Cu2MnAl(111)     | 1.82810      | (Heusler) |
                +------------------+--------------+-----------+
                | Ge(111)          | 1.92366      |           |
                +------------------+--------------+-----------+
                | Ge(220)          | 3.14131      |           |
                +------------------+--------------+-----------+
                | Ge(311)          | 3.68351      |           |
                +------------------+--------------+-----------+
                | Ge(511)          | 5.76968      |           |
                +------------------+--------------+-----------+
                | Ge(533)          | 7.28063      |           |
                +------------------+--------------+-----------+
                | PG(002)          | 1.87325      |           |
                +------------------+--------------+-----------+
                | PG(004)          | 3.74650      |           |
                +------------------+--------------+-----------+
                | PG(110)          | 5.49806      |           |
                +------------------+--------------+-----------+
                | Si(111)          | 2.00421      |           |
                +------------------+--------------+-----------+
        
        mosaic : int
            The monochromator mosaic in minutes of arc.
        
        vmosaic : int
            The vertical mosaic of monochromator in minutes of arc. If
            this field is left unassigned, an isotropic mosaic is assumed.
        '''
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value

    @property
    def ana(self):
        u'''A structure that describes the analyzer and contains fields as in :attr:`mono`
        plus optional fields.
        
        Parameters
        ----------
        thickness: float 
            The analyzer thickness in cm for ideal-crystal reflectivity
            corrections (Section II C 3). If no reflectivity corrections are to be made, this field
            should remain unassigned or set to a negative value.
        
        Q : float
            The kinematic reflectivity coefficient for this correction. It is given by 
        
            .. math::    Q = \\frac{4|F|**2}{V_0} \\frac{(2\\pi)**3}{\\tau**3}, 
        
            where V0 is the unit cell volume for the analyzer crystal, F is the structure factor of 
            the analyzer reflection, and τ is the analyzer reciprocal lattice vector. 
            For PG(002) Q = 0.1287. Leave this field unassigned or make it negative if you don’t
            want the correction done.
        
        horifoc : bool 
            A flag that is set to 1 if a horizontally focusing analyzer is used
            (Section II D). In this case ``hcol[2]`` (see below) is the angular size of the
            analyzer, as seen from the sample position. If the field is unassigned or equal to
            -1, a flat analyzer is assumed. Note that this option is only available with the
            Cooper-Nathans method.
        '''
        return self._ana

    @ana.setter
    def ana(self, value):
        self._ana = value

    @property
    def method(self):
        '''Selects the computation method. 
        If ``method=0`` or left undefined, a Cooper-Nathans calculation is
        performed. For a Popovici calculation set ``method=1``.
        '''
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def moncar(self):
        '''Selects the type of normalization used to calculate ``R0``. 
        If ``moncor=1`` or left undefined, ``R0`` is calculated in normalization to monitor counts (Section
        II C 2). 1/k\ :sub:`i` monitor efficiency correction is included automatically. To normalize
        ``R0`` to source flux (Section II C 1), use ``moncor=0``.
        '''
        return self._moncar

    @moncar.setter
    def moncar(self, value):
        self._moncar = value

    @property
    def hcol(self):
        ''' The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. In case of a horizontally-focusing
        analyzer ``hcol[2] is the angular size of the analyzer, as seen from the sample
        position. If the beam divergence is limited by a neutron guide, the corresponding
        element of :attr:`hcol` is the negative of the guide’s *m*-value. For example, for a 58-Ni
        guide (*m*=1.2) before the monochromator, `hcol[0]` should be -1.2.
        '''
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        '''The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. If the beam divergence is limited
        by a neutron guide, the corresponding element of :attr:`vcol` is the negative of the
        guide’s *m*-value. For example, for a 58-Ni guide (*m*=1.2) before the monochromator,
        ``vcol[0]`` should be -1.2.
        '''
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        '''distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and monochromator and
        monitor, respectively. The 5th element is only needed if ``moncor=1``
        '''
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = value

    @property
    def efixed(self):
        '''the fixed incident or final neutron energy, in meV.
        '''
        return self._efixed

    @efixed.setter
    def efixed(self, value):
        self._efixed = value

    @property
    def sample(self):
        '''A structure that describes the sample. 
        
        It contains the following fields:
        
        * EXP.sample.mosaic is the FWHM sample mosaic in the scattering plane in minutes
        of arc. If this field is left unassigned, no sample mosaic corrections (section
        II E) are performed.

        * ``sample.vmosaic`` is the vertical sample mosaic in minutes of arc. If this field
        is left unassigned, isotropic mosaic is assumed.
        '''
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def orient1(self):
        '''Miller indexes of the first reciprocal-space orienting vector for
        the S coordinate system, as explained in Section II G.
        '''
        return self._sample.u

    @orient1.setter
    def orient1(self, value):
        self._sample.u = np.array(value)

    @property
    def orient2(self):
        '''Miller indexes of the second reciprocal-space orienting vector
        for the S coordinate system, as explained in Section II G.
        '''
        return self._sample.v

    @orient2.setter
    def orient2(self, value):
        self._sample.v = np.array(value)

    @property
    def dir1(self):
        '''defines the scattering direction in the monochromator. This field is equal
        to 1 or left unassigned if the scattering direction in the monochromator is opposite to
        that in the sample. Set this field to -1 if the sample and monochromator scattering
        directions are the same.
        '''
        return self._dir1

    @dir1.setter
    def dir1(self, value):
        self._dir1 = value

    @property
    def dir2(self):
        '''defines the scattering direction in the analyzer. This field is equal to 1 or
        left unassigned if the scattering direction in the analyzer is opposite to that in the
        sample. Set this field to -1 if the sample and analyzer scattering directions are the
        same.
        '''
        return self._dir2

    @dir2.setter
    def dir2(self, value):
        self._dir2 = value

    @property
    def mondir(self):
        '''defines the scattering angle in the monochromator which is positive
        (counter-clockwise) if this field is absent or positive, and negative (clockwise) otherwise
        [10].
        '''
        return self._mondir

    @mondir.setter
    def mondir(self, value):
        self._mondir = value

    @property
    def infin(self):
        '''a flag set to -1 or left unassigned if the final energy is fixed, or set to +1
        in a fixed-incident setup.
        '''
        return self._infin

    @infin.setter
    def infin(self, value):
        self._infin = value

    @property
    def beam(self):
        r'''A structure that describes the source
        '''
        return self._beam

    @beam.setter
    def beam(self, value):
        self._beam = value

    @property
    def detector(self):
        '''A structure that describes the detector
        '''
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def monitor(self):
        '''A structure that describes the monitor
        '''
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def Smooth(self):
        u'''Defines the smoothing parameters as explained in Section II H. Leave this
        field unassigned if you don’t want this correction done.
        
        * ``Smooth.E`` is the smoothing FWHM in energy (meV). A small number means
        “no smoothing along this direction”.
        
        * ``Smooth.X`` is the smoothing FWHM along the first orienting vector (x0 axis)
        in Å\ :sup:`-1`.
        
        * ``Smooth.Y`` is the smoothing FWHM along the y axis in Å\ :sup:`-1`.
        
        * ``Smooth.Z`` is the smoothing FWHM along the vertical direction in Å\ :sup:`-1`.
        '''
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value

    def get_lattice(self):
        r'''Extracts lattice parameters from EXP and returns the direct and
        reciprocal lattice parameters in the form used by _scalar.m, _star.m, etc.
    
        Returns
        -------
        [lattice, rlattice] : [class, class]
            Returns the direct and reciprocal lattice sample classes
    
        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory
    
        '''
        s = np.array([self.sample])
        lattice = Sample(np.array([item.a for item in s]),
                          np.array([item.b for item in s]),
                          np.array([item.c for item in s]),
                          np.array([item.alpha for item in s]) * np.pi / 180,
                          np.array([item.beta for item in s]) * np.pi / 180,
                          np.array([item.gamma for item in s]) * np.pi / 180)
        V, Vstar, rlattice = _star(lattice)  # @UnusedVariable
    
        return [lattice, rlattice]

    def _StandardSystem(self):
        r'''Returns rotation matrices to calculate resolution in the sample view
        instead of the instrument view
    
        Parameters
        ----------
        EXP : class
            Instrument class
    
        Returns
        -------
        [x, y, z, lattice, rlattice] : [ndarray, ndarray, ndarray, class, class]
            Returns the rotation matrices and real and reciprocal lattice sample classes
    
        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory
    
        '''
        [lattice, rlattice] = self.get_lattice()
    
        orient1 = self.orient1
        orient2 = self.orient2
    
        modx = _modvec([orient1[0], orient1[1], orient1[2]], rlattice)
    
        x = orient1 / modx
#         x[0] = x[0] / modx  # First unit basis vector
#         x[1] = x[1] / modx
#         x[2] = x[2] / modx
    
        proj = _scalar([orient2[0], orient2[1], orient2[2]], [x[0], x[1], x[2]], rlattice)
    
        y = orient2 - x * proj
#         y[0] = y[0] - x[0, :] * proj
#         y[1] = y[1, :] - x[1, :] * proj
#         y[2] = y[2, :] - x[2, :] * proj
    
        mody = _modvec([y[0], y[1], y[2]], rlattice)
    
        if len(np.where(mody <= 0)[0]) > 0:
            raise ValueError('??? Fatal error: Orienting vectors are colinear!')
    
        y /= mody
#         y[0, :] = y[0, :] / mody  # Second unit basis vector
#         y[1, :] = y[1, :] / mody
#         y[2, :] = y[2, :] / mody
    
        z = np.zeros(3, dtype=np.float64)
        z[0] = x[1] * y[2] - y[1] * x[2]
        z[1] = x[2] * y[0] - y[2] * x[0]
        z[2] = -x[1] * y[0] + y[1] * x[0]
    
        proj = _scalar([z[0], z[1], z[2]], [x[0], x[1], x[2]], rlattice)
        
        z = z - x * proj
#         z[0, :] = z[0, :] - x[0, :] * proj
#         z[1, :] = z[1, :] - x[1, :] * proj
#         z[2, :] = z[2, :] - x[2, :] * proj
    
        proj = _scalar([z[0], z[1], z[2]], [y[0], y[1], y[2]], rlattice)
    
        z = z - y * proj
#         z[0, :] = z[0, :] - y[0, :] * proj
#         z[1, :] = z[1, :] - y[1, :] * proj
#         z[2, :] = z[2, :] - y[2, :] * proj
    
        modz = _modvec([z[0], z[1], z[2]], rlattice)
        
        z /= modz
#         z[0, :] = z[0, :] / modz  # Third unit basis vector
#         z[1, :] = z[1, :] / modz
#         z[2, :] = z[2, :] / modz
    
        return [x, y, z, lattice, rlattice]

    def calc_resolution_in_Q_coords(self, Q, W):
        r'''For a momentum transfer Q and energy transfers W, given experimental
        conditions specified in EXP, calculates the Cooper-Nathans or Popovici
        resolution matrix RM and resolution prefactor R0 in the Q coordinate
        system (defined by the scattering vector and the scattering plane).
    
        Parameters
        ----------
        Q : ndarray or list of ndarray
            The Q vectors in reciprocal space at which resolution should be
            calculated, in inverse angstroms
    
        W : float or list of floats
            The energy transfers at which resolution should be calculated in meV
    
        Returns
        -------
        [R0, RM] : list(float, ndarray)
            Resolution pre-factor (R0) and resolution matrix (RM) at the given
            reciprocal lattice vectors and energy transfers
    
        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
        Oak Ridge National Laboratory
    
        '''
        CONVERT1 = 0.4246609 * np.pi / 60. / 180.
        CONVERT2 = 2.072
                
        [length, Q, W] = _CleanArgs(Q, W)
    
        RM = np.zeros((4, 4, length), dtype=np.float64)
        R0 = np.zeros(length, dtype=np.float64)
        RM_ = np.zeros((4, 4), dtype=np.float64)  # @UnusedVariable
        D = np.matrix(np.zeros((8, 13), dtype=np.float64))
        d = np.matrix(np.zeros((4, 7), dtype=np.float64))
        T = np.matrix(np.zeros((4, 13), dtype=np.float64))
        t = np.matrix(np.zeros((2, 7), dtype=np.float64))
        A = np.matrix(np.zeros((6, 8), dtype=np.float64))
        C = np.matrix(np.zeros((4, 8), dtype=np.float64))
        B = np.matrix(np.zeros((4, 6), dtype=np.float64))

        # the method to use
        method = 0
        if hasattr(self, 'method'):
            method = self.method

        # Assign default values and decode parameters
        moncor = 1
        if hasattr(self, 'moncor'):
            moncor = self.moncor

        alpha = np.array(self.hcol) * CONVERT1
        beta = np.array(self.vcol) * CONVERT1
        mono = self.mono
        etam = np.array(mono.mosaic) * CONVERT1
        etamv = np.copy(etam)
        if hasattr(mono, 'vmosaic') and (method == 1 or method == 'Popovici'):
            etamv = np.array(mono.vmosaic) * CONVERT1

        ana = self.ana
        etaa = np.array(ana.mosaic) * CONVERT1
        etaav = np.copy(etaa)
        if hasattr(ana, 'vmosaic'):
            etaav = np.array(ana.vmosaic) * CONVERT1

        sample = self.sample
        infin = -1
        if hasattr(self, 'infin'):
            infin = self.infin

        efixed = self.efixed
        epm = 1
        if hasattr(self, 'dir1'):
            epm = self.dir1

        ep = 1
        if hasattr(self, 'dir2'):
            ep = self.dir2

        monitorw = 1.
        monitorh = 1.
        beamw = 1.
        beamh = 1.
        monow = 1.
        monoh = 1.
        monod = 1.
        anaw = 1.
        anah = 1.
        anad = 1.
        detectorw = 1.
        detectorh = 1.
        sshapes = [np.identity(3)] * length
        L0 = 1.
        L1 = 1.
        L1mon = 1.
        L2 = 1.
        L3 = 1.
        monorv = 1.e6
        monorh = 1.e6
        anarv = 1.e6
        anarh = 1.e6
        
        if hasattr(self, 'beam'):
            beam = self.beam
            if hasattr(beam, 'width'):
                beamw = beam.width ** 2

            if hasattr(beam, 'height'):
                beamh = beam.height ** 2

        bshape = np.diag([beamw, beamh])
        if hasattr(self, 'monitor'):
            monitor = self.monitor
            if hasattr(monitor, 'width'):
                monitorw = monitor.width ** 2

            monitorh = monitorw
            if hasattr(monitor, 'height'):
                monitorh = monitor.height ** 2

        monitorshape = np.diag([monitorw, monitorh])
        if hasattr(self, 'detector'):
            detector = self.detector
            if hasattr(detector, 'width'):
                detectorw = detector.width ** 2

            if hasattr(detector, 'height'):
                detectorh = detector.height ** 2

        dshape = np.diag([detectorw, detectorh])
        if hasattr(mono, 'width'):
            monow = mono.width ** 2

        if hasattr(mono, 'height'):
            monoh = mono.height ** 2

        if hasattr(mono, 'depth'):
            monod = mono.depth ** 2

        mshape = np.diag([monod, monow, monoh])
        if hasattr(ana, 'width'):
            anaw = ana.width ** 2

        if hasattr(ana, 'height'):
            anah = ana.height ** 2

        if hasattr(ana, 'depth'):
            anad = ana.depth ** 2

        ashape = np.diag([anad, anaw, anah])
        if hasattr(sample, 'width') and hasattr(sample, 'depth') and hasattr(sample, 'height'):
            sshape = np.diag([sample.depth, sample.width, sample.height]) ** 2
        elif hasattr(sample, 'shape'):
            sshapes = sample.shape

        if hasattr(self, 'arms'):
            arms = self.arms
            L0 = arms[0]
            L1 = arms[1]
            L2 = arms[2]
            L3 = arms[3]
            L1mon = np.copy(L1)
            if len(arms) > 4:
                L1mon = np.copy(arms[4])

        if hasattr(mono, 'rv'):
            monorv = mono.rv

        if hasattr(mono, 'rh'):
            monorh = mono.rh

        if hasattr(ana, 'rv'):
            anarv = ana.rv

        if hasattr(ana, 'rh'):
            anarh = ana.rh

        taum = GetTau(mono.tau)
        taua = GetTau(ana.tau)

        horifoc = -1
        if hasattr(self, 'horifoc'):
            horifoc = self.horifoc

        if horifoc == 1:
            alpha[2] = alpha[2] * np.sqrt(8 * np.log(2) / 12.)

        em = 1
        if hasattr(self, 'mondir'):
            em = self.mondir
    
        for ind in range(length):
            sshape = sshapes[ind]
            # Calculate angles and energies
            w = W[ind]
            q = Q[ind]
            ei = efixed
            ef = efixed
            if infin > 0:
                ef = efixed - w
            else:
                ei = efixed + w
            ki = np.sqrt(ei / CONVERT2)
            kf = np.sqrt(ef / CONVERT2)
    
            thetam = np.arcsin(taum / (2. * ki)) * np.sign(epm) * np.sign(em)  # sm  # added sign(em) K.P.
            thetaa = np.arcsin(taua / (2. * kf)) * np.sign(ep) * np.sign(em)  # sa
            s2theta = -np.arccos((ki ** 2 + kf ** 2 - q ** 2) / (2. * ki * kf)) * np.sign(em)  # ss  # 2theta sample @IgnorePep8
            if np.iscomplex(s2theta):
                raise ValueError(': KI,KF,Q triangle will not close (kinematic equations). Change the value of KFIX,FX,QH,QK or QL.')
    
            thetas = s2theta / 2.
            phi = np.arctan2(np.sign(em) * (-kf * np.sin(s2theta)), np.sign(em) * (ki - kf * np.cos(s2theta)))
    
            # Calculate beam divergences defined by neutron guides
            if alpha[0] < 0:
                alpha[0] = -alpha[0] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if alpha[1] < 0:
                alpha[1] = -alpha[1] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if alpha[2] < 0:
                alpha[2] = -alpha[2] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if alpha[3] < 0:
                alpha[3] = -alpha[3] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            
            if beta[0] < 0:
                beta[0] = -beta[0] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if beta[1] < 0:
                beta[1] = -beta[1] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if beta[2] < 0:
                beta[2] = -beta[2] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            if beta[3] < 0:
                beta[3] = -beta[3] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
    
            # Redefine sample geometry
            psi = thetas - phi  # Angle from sample geometry X axis to Q
            rot = np.matrix(np.zeros((3, 3), dtype=np.float64))
            rot[0, 0] = np.cos(psi)
            rot[1, 1] = np.cos(psi)
            rot[0, 1] = np.sin(psi)
            rot[1, 0] = -np.sin(psi)
            rot[2, 2] = 1.
    
            # sshape=rot'*sshape*rot
            sshape = np.matrix(rot) * np.matrix(sshape) * np.matrix(rot).H
    
            # Definition of matrix G
            G = 1. / np.array([alpha[0], alpha[1], beta[0], beta[1], alpha[2], alpha[3], beta[2], beta[3]], dtype=np.float64) ** 2
            G = np.matrix(np.diag(G))
    
            # Definition of matrix F
            F = 1. / np.array([etam, etamv, etaa, etaav], dtype=np.float64) ** 2
            F = np.matrix(np.diag(F))
    
            # Definition of matrix A
            A[0, 0] = ki / 2. / np.tan(thetam)
            A[0, 1] = -A[0, 0]
            A[3, 4] = kf / 2. / np.tan(thetaa)
            A[3, 5] = -A[3, 4]
            A[1, 1] = ki
            A[2, 3] = ki
            A[4, 4] = kf
            A[5, 6] = kf
    
            # Definition of matrix C
            C[0, 0] = 1. / 2.
            C[0, 1] = 1. / 2.
            C[2, 4] = 1. / 2.
            C[2, 5] = 1. / 2.
            C[1, 2] = 1. / (2. * np.sin(thetam))
            C[1, 3] = -C[1, 2]  # mistake in paper
            C[3, 6] = 1. / (2. * np.sin(thetaa))
            C[3, 7] = -C[3, 6]
    
            # Definition of matrix B
            B[0, 0] = np.cos(phi)
            B[0, 1] = np.sin(phi)
            B[0, 3] = -np.cos(phi - s2theta)
            B[0, 4] = -np.sin(phi - s2theta)
            B[1, 0] = -B[0, 1]
            B[1, 1] = B[0, 0]
            B[1, 3] = -B[0, 4]
            B[1, 4] = B[0, 3]
            B[2, 2] = 1.
            B[2, 5] = -1.
            B[3, 0] = 2. * CONVERT2 * ki
            B[3, 3] = -2. * CONVERT2 * kf
    
            # Definition of matrix S
            Sinv = np.matrix(blkdiag(np.array(bshape, dtype=np.float32), mshape, sshape, ashape, dshape))  # S-1 matrix
            S = Sinv.I
    
            # Definition of matrix T
            T[0, 0] = -1. / (2. * L0)  # mistake in paper
            T[0, 2] = np.cos(thetam) * (1. / L1 - 1. / L0) / 2.
            T[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1 - 2. / (monorh * np.sin(thetam))) / 2.
            T[0, 5] = np.sin(thetas) / (2. * L1)
            T[0, 6] = np.cos(thetas) / (2. * L1)
            T[1, 1] = -1. / (2. * L0 * np.sin(thetam))
            T[1, 4] = (1. / L0 + 1. / L1 - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
            T[1, 7] = -1. / (2. * L1 * np.sin(thetam))
            T[2, 5] = np.sin(thetas) / (2. * L2)
            T[2, 6] = -np.cos(thetas) / (2. * L2)
            T[2, 8] = np.cos(thetaa) * (1. / L3 - 1. / L2) / 2.
            T[2, 9] = np.sin(thetaa) * (1. / L2 + 1. / L3 - 2. / (anarh * np.sin(thetaa))) / 2.
            T[2, 11] = 1. / (2. * L3)
            T[3, 7] = -1. / (2. * L2 * np.sin(thetaa))
            T[3, 10] = (1. / L2 + 1. / L3 - 2. * np.sin(thetaa) / anarv) / (2. * np.sin(thetaa))
            T[3, 12] = -1. / (2. * L3 * np.sin(thetaa))
    
            # Definition of matrix D
            # Lots of index mistakes in paper for matrix D
            D[0, 0] = -1. / L0
            D[0, 2] = -np.cos(thetam) / L0
            D[0, 3] = np.sin(thetam) / L0
            D[2, 1] = D[0, 0]
            D[2, 4] = -D[0, 0]
            D[1, 2] = np.cos(thetam) / L1
            D[1, 3] = np.sin(thetam) / L1
            D[1, 5] = np.sin(thetas) / L1
            D[1, 6] = np.cos(thetas) / L1
            D[3, 4] = -1. / L1
            D[3, 7] = -D[3, 4]
            D[4, 5] = np.sin(thetas) / L2
            D[4, 6] = -np.cos(thetas) / L2
            D[4, 8] = -np.cos(thetaa) / L2
            D[4, 9] = np.sin(thetaa) / L2
            D[6, 7] = -1. / L2
            D[6, 10] = -D[6, 7]
            D[5, 8] = np.cos(thetaa) / L3
            D[5, 9] = np.sin(thetaa) / L3
            D[5, 11] = 1. / L3
            D[7, 10] = -D[5, 11]
            D[7, 12] = D[5, 11]
    
            # Definition of resolution matrix M
            if method == 1 or method == 'popovici':
                Minv = np.matrix(B) * np.matrix(A) * np.linalg.inv(np.linalg.inv(np.matrix(D) * np.linalg.inv(np.matrix(S) + np.matrix(T).H * np.matrix(F) * np.matrix(T)) * np.matrix(D).H) + np.matrix(G)) * np.matrix(A).H * np.matrix(B).H
            else:
                HF = np.matrix(A) * np.linalg.inv(np.matrix(G) + np.matrix(C).H * np.matrix(F) * np.matrix(C)) * np.matrix(A).H
                # Horizontally focusing analyzer if needed
                if horifoc > 0:
                    HF = np.linalg.inv(HF)
                    HF[4, 4] = (1 / (kf * alpha[2])) ** 2
                    HF[4, 3] = 0
                    HF[3, 4] = 0 
                    HF[3, 3] = (np.tan(thetaa) / (etaa * kf)) ** 2
                    HF = np.linalg.inv(HF)
                Minv = np.matrix(B) * np.matrix(HF) * np.matrix(B).H
    
            M = np.linalg.inv(Minv)
            # TODO: rows-columns 3-4 swapped for ResPlot to work.
            # Inactivate as we want M=[x,y,z,E]
            RM_[0, 0] = M[0, 0]
            RM_[1, 0] = M[1, 0]
            RM_[0, 1] = M[0, 1]
            RM_[1, 1] = M[1, 1]
    
            RM_[0, 2] = M[0, 3]
            RM_[2, 0] = M[3, 0]
            RM_[2, 2] = M[3, 3]
            RM_[2, 1] = M[3, 1]
            RM_[1, 2] = M[1, 3]
    
            RM_[0, 3] = M[0, 2]
            RM_[3, 0] = M[2, 0]
            RM_[3, 3] = M[2, 2]
            RM_[3, 1] = M[2, 1]
            RM_[1, 3] = M[1, 2]
                
            # Calculation of prefactor, normalized to source
            Rm = ki ** 3 / np.tan(thetam)
            Ra = kf ** 3 / np.tan(thetaa)
    
            if method == 1 or method == 'popovici':
                # Popovici
                R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * np.sin(thetam) * np.sin(thetaa)) * np.sqrt(np.linalg.det(F) / np.linalg.det(np.linalg.inv(np.matrix(D) * np.linalg.inv(np.matrix(S) + np.matrix(T).H * np.matrix(F) * np.matrix(T)) * np.matrix(D).H) + np.matrix(G)))
            else:
                # Cooper-Nathans (popovici Eq 5 and 9)
    #             R0_ = R0_ * np.sqrt(np.linalg.det(np.matrix(F)) / np.linalg.det(np.matrix(H)))
                R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * np.sin(thetam) * np.sin(thetaa)) * np.sqrt(np.linalg.det(np.matrix(F)) / np.linalg.det(np.matrix(G) + np.matrix(C).H * np.matrix(F) * np.matrix(C)))
                
            # Normalization to flux on monitor
            if moncor == 1:
                g = G[0:4, 0:4]
                f = F[0:2, 0:2]
                c = C[0:2, 0:4]
                t[0, 0] = -1. / (2. * L0)  # mistake in paper
                t[0, 2] = np.cos(thetam) * (1. / L1mon - 1. / L0) / 2.
                t[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1mon - 2. / (monorh * np.sin(thetam))) / 2.
                t[0, 6] = 1. / (2. * L1mon)
                t[1, 1] = -1. / (2. * L0 * np.sin(thetam))
                t[1, 4] = (1. / L0 + 1. / L1mon - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
                sinv = blkdiag(np.array(bshape, dtype=np.float32), mshape, monitorshape)  # S-1 matrix
                s = np.linalg.inv(sinv)
                d[0, 0] = -1. / L0
                d[0, 2] = -np.cos(thetam) / L0
                d[0, 3] = np.sin(thetam) / L0
                d[2, 1] = D[0, 0]
                d[2, 4] = -D[0, 0]
                d[1, 2] = np.cos(thetam) / L1mon
                d[1, 3] = np.sin(thetam) / L1mon
                d[1, 5] = 0.
                d[1, 6] = 1. / L1mon
                d[3, 4] = -1. / L1mon
                if method == 1 or method == 'popovici':
                    # Popovici
                    Rmon = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * 
                            np.sqrt(np.linalg.det(f) / np.linalg.det((np.matrix(d) * 
                                                                      (np.matrix(s) + np.matrix(t).H * np.matrix(f) * np.matrix(t)).I * 
                                                                      np.matrix(d).H).I + np.matrix(g))))
                else:
                    # Cooper-Nathans
                    Rmon = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * 
                            np.sqrt(np.linalg.det(f) / np.linalg.det(np.matrix(g) + np.matrix(c).H * np.matrix(f) * np.matrix(c))))
    
                R0_ = R0_ / Rmon
                R0_ = R0_ * ki  # 1/ki monitor efficiency
    
            # Transform prefactor to Chesser-Axe normalization
            R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(RM_))
    
            # Include kf/ki part of cross section
            R0_ = R0_ * kf / ki
    
            # Take care of sample mosaic if needed
            # [S. A. Werner & R. Pynn, J. Appl. Phys. 42, 4736, (1971), eq 19]
            if hasattr(sample, 'mosaic'):
                etas = sample.mosaic * CONVERT1
                etasv = np.copy(etas)
                if hasattr(sample, 'vmosaic'):
                    etasv = sample.vmosaic * CONVERT1
    
                R0_ = R0_ / np.sqrt((1 + (q * etas) ** 2 * RM_[3, 3]) * (1 + (q * etasv) ** 2 * RM_[1, 1]))
                Minv = np.linalg.inv(RM_)
                Minv[1, 1] = Minv[1, 1] + q ** 2 * etas ** 2
                Minv[3, 3] = Minv[3, 3] + q ** 2 * etasv ** 2
                RM_ = np.linalg.inv(Minv)
    
            # Take care of analyzer reflectivity if needed [I. Zaliznyak, BNL]
            if hasattr(ana, 'thickness') and hasattr(ana, 'Q'):
                KQ = ana.Q
                KT = ana.thickness
                toa = (taua / 2.) / np.sqrt(kf ** 2 - (taua / 2.) ** 2)
                smallest = alpha[3]
                if alpha[3] > alpha[2]:
                    smallest = alpha[2]
                Qdsint = KQ * toa
                dth = (np.arange(1, 201) / 200.) * np.sqrt(2. * np.log(2.)) * smallest
                wdth = np.exp(-dth ** 2 / 2. / etaa ** 2)
                sdth = KT * Qdsint * wdth / etaa / np.sqrt(2. * np.pi)
                rdth = 1. / (1 + 1. / sdth)
                reflec = sum(rdth) / sum(wdth)
                R0_ = R0_ * reflec
    
            R0[ind] = R0_
            RM[:, :, ind] = np.copy(RM_[:, :])
    
        return [R0, RM]
    
    def calc_resolution(self, hkle):
        r'''For a scattering vector (H,K,L) and  energy transfers W, given
        experimental conditions specified in EXP, calculates the Cooper-Nathans
        resolution matrix RMS and Cooper-Nathans Resolution prefactor R0 in a
        coordinate system defined by the crystallographic axes of the sample.

        Parameters
        ----------
        hkle : list
            Array of the scattering vector and energy transfer at which the
            calculation should be performed

        npts : int, optional
            Number of points in the ouput curves

        Notes
        -----
            Translated from ResLib, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

        '''
        [H, K, L, W] = hkle

        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        self.H, self.K, self.L, self.W = H, K, L, W

        [x, y, z, sample, rsample] = self._StandardSystem()
        
        Q = _modvec([H, K, L], rsample)
        uq0 = H / Q  # Unit vector along Q
        uq1 = K / Q
        uq2 = L / Q
        uq = np.vstack((uq0, uq1, uq2))

        xq = _scalar([x[0], x[1], x[2]], [uq[0], uq[1], uq[2]], rsample)
        yq = _scalar([y[0], y[1], y[2]], [uq[0], uq[1], uq[2]], rsample)
        zq = 0  # scattering vector assumed to be in (orient1,orient2) plane

        tmat = np.zeros((4, 4, length), dtype=np.float64)  # Coordinate transformation matrix
        tmat[3, 3, :] = 1.
        tmat[2, 2, :] = 1.
        tmat[0, 0, :] = xq
        tmat[0, 1, :] = yq
        tmat[1, 1, :] = xq
        tmat[1, 0, :] = -yq

        RMS = np.zeros((4, 4, length), dtype=np.float64)
        rot = np.zeros((3, 3), dtype=np.float64)

        # Sample shape matrix in coordinate system defined by scattering vector
        
        sample = self.sample
        if hasattr(sample, 'shape'):
            samples = []
            for i in range(length):
                rot[0, 0] = tmat[0, 0, i]
                rot[1, 0] = tmat[1, 0, i]
                rot[0, 1] = tmat[0, 1, i]
                rot[1, 1] = tmat[1, 1, i]
                rot[2, 2] = tmat[2, 2, i]
                samples.append(np.matrix(rot) * np.matrix(sample.shape) * np.matrix(rot.T))
            self.sample.shape = samples

        [R0, RM] = self.calc_resolution_in_Q_coords(Q, W)

        for i in range(length):
            RMS[:, :, i] = np.matrix(tmat[:, :, i]).H * np.matrix(RM[:, :, i]) * np.matrix(tmat[:, :, i])

        mul = np.zeros((4, 4))
        e = np.identity(4)
        for i in range(length):
            if hasattr(self, 'Smooth'):
                if self.Smooth.X:
                    mul[0, 0] = 1 / (self.Smooth.X ** 2 / 8 / np.log(2))
                    mul[1, 1] = 1 / (self.Smooth.Y ** 2 / 8 / np.log(2))
                    mul[2, 2] = 1 / (self.Smooth.E ** 2 / 8 / np.log(2))
                    mul[3, 3] = 1 / (self.Smooth.Z ** 2 / 8 / np.log(2))
                    R0[i] = R0[i] / np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(RMS[:, :, i]))) * np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[:, :, i])))
                    RMS[:, :, i] = np.matrix(e) / (np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[:, :, i]))

        self.R0, self.RMS, self.RM = R0, RMS, RM

    def calc_projections(self, hkle, npts=36):
        r'''Calculates the resolution ellipses for projections and slices from the resolution matrix.

        Parameters
        ----------
        hkle : list
            Positions at which projections should be calculated.

        npts : int, optional
            Number of points in the outputted ellipse curve

        Returns
        -------
        projections : dictionary
            A dictionary containing projections in the planes: QxQy, QxW, and QyW, both projections and slices

        '''
        [H, K, L, W] = hkle
        try:
            if H == self.H and K == self.K and L == self.L and W == self.W:
                NP = np.array(self.RMS)
                R0 = self.R0
            else:
                self.calc_resolution(hkle)
                NP = np.array(self.RMS)
                R0 = self.R0
        except AttributeError:
            self.calc_resolution(hkle)
            NP = np.array(self.RMS)
            R0 = self.R0

        const = 1.17741  # half width factor
        
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        hkle = [H, K, L, W]

        self.projections = {'QxQy': np.zeros((2, npts, NP.shape[-1])),
                            'QxQySlice': np.zeros((2, npts, NP.shape[-1])),
                            'QxW': np.zeros((2, npts, NP.shape[-1])),
                            'QxWSlice': np.zeros((2, npts, NP.shape[-1])),
                            'QyW': np.zeros((2, npts, NP.shape[-1])),
                            'QyWSlice': np.zeros((2, npts, NP.shape[-1]))}
        
        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()
        
        o1 = self.orient1
        o2 = self.orient2
        pr = _scalar([o2[0], o2[1], o2[2]], [yvec[0], yvec[1], yvec[2]], rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr
        
        if np.abs(o2[0]) < 1e-5:
            o2[0] = 0
        if np.abs(o2[1]) < 1e-5:
            o2[1] = 0
        if np.abs(o2[2]) < 1e-5:
            o2[2] = 0
        
        if np.abs(o1[0]) < 1e-5:
            o1[0] = 0
        if np.abs(o1[1]) < 1e-5:
            o1[1] = 0
        if np.abs(o1[2]) < 1e-5:
            o1[2] = 0
        frame = '[Q1,Q2,E]'
        
        A = NP

        for ind in range(A.shape[-1]):
            # Remove the vertical component from the matrix.
            B = A[:3, :3, ind]

            # Projection into Qx, Qy plane
            [R0P, MP] = project_into_plane(2, R0, B)
            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQy'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, npts=npts)

            # Slice through Qx,Qy plane
            MP = A[:2, :2, ind]

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQySlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, npts=npts)

            # Projection into Qx, W plane

            [R0P, MP] = project_into_plane(1, R0, B)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[3][ind]], npts=npts)

            # Slice through Qx,W plane
            MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[3][ind]], npts=npts)

            # Projections into Qy, W plane
            [R0P, MP] = project_into_plane(0, R0, B)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1][ind], hkle[3][ind]], npts=npts)

            # Slice through Qy,W plane
            MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1][ind], hkle[3][ind]], npts=npts)

    def get_resolution_params(self, hkle, plane, mode='project'):
        r'''Returns parameters for the resolution gaussian.

        Parameters
        ----------
        hkle : list of floats
            Position and energy for which parameters should be returned

        plane : 'QxQy' | 'QxQySlice' | 'QxW' | 'QxWSlice' | 'QyW' | 'QyWSlice'
            Two dimensional plane for which parameters should be returned

        mode : 'project' | 'slice'
            Return the projection into or slice through the chosen plane

        Returns
        -------
        tuple : R0, RMxx, RMyy, RMxy
            Parameters for the resolution gaussian

        '''
        A = self.RMS

        ind = np.where((self.H == hkle[0]) & (self.K == hkle[1]) & (self.L == hkle[2]) & (self.W == hkle[3]))
        if len(ind[0]) == 0:
            raise ValueError('Resolution at provided HKLE has not been calculated.')

        ind = ind[0][0]

        # Remove the vertical component from the matrix
        B = np.vstack((np.hstack((A[0, :2:1, ind], A[0, 3, ind])),
                       np.hstack((A[1, :2:1, ind], A[1, 3, ind])),
                       np.hstack((A[3, :2:1, ind], A[3, 3, ind]))))

        if plane == 'QxQy':
            if mode == 'project':
                # Projection into Qx, Qy plane
                R0 = np.sqrt(2 * np.pi / B[2, 2]) * self.R0[ind]
                MP = project_into_plane(B, 2)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,Qy plane
                MP = np.array(A[:2:1, :2:1, ind])
                return (self.R0[ind], MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QxW':
            if mode == 'project':
                # Projection into Qx, W plane
                R0 = np.sqrt(2 * np.pi / B[1, 1]) * self.R0[ind]
                MP = project_into_plane(B, 1)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,W plane
                MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])
                return (self.R0[ind][0], MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QyW':
            if mode == 'project':
                # Projections into Qy, W plane
                R0 = np.sqrt(2 * np.pi / B[0, 0]) * self.R0
                MP = project_into_plane(B, 0)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qy,W plane
                MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])
                return (self.R0[ind][0], MP[0, 0], MP[1, 1], MP[0, 1])

    def resolution_convolution(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None):
        r'''Numerically calculate the convolution of a user-defined cross-section 
        function with the resolution function for a 3-axis neutron scattering 
        experiment.
        
        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.
        
        pref : func
            User-supplied "slow" cross section prefactor and background function.
        
        nargout : int
            Number of arguments returned by the pref function
        
        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e. 
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal lattice 
            units and W in meV.
        
        EXP : obj
            Instrument object containing all information on experimental setup.
        
        METHOD : str
            Specifies which 4D-integration method to use. 'fix' (Default): sample
            the cross section on a fixed grid of points uniformly distributed 
            $\phi$-space. 2*ACCURACY[0]+1 points are sampled along $\phi_1$,
            $\phi_2$, and $\phi_3$, and 2*ACCURACY[1]+1 along $\phi_4$ (vertical
            direction). 'mc': 4D Monte Carlo integration. The cross section is
            sampled in 1000*ACCURACY randomly chosen points, uniformly distributed
            in $\phi$-space.
        
        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.
        
        p : list
            A parameter that is passed on, without change to sqw and pref.
            
        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$
            
        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
        Oak Ridge National Laboratory

        '''        
        self.calc_resolution(hkle)
        [R0, RMS] = [np.copy(self.R0), np.copy(self.RMS)]
        
        H, K, L, W = hkle
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()
    
        Mzz = RMS[3, 3, :]
        Mww = RMS[2, 2, :]
        Mxx = RMS[0, 0, :]
        Mxy = RMS[0, 1, :]
        Mxw = RMS[0, 2, :]
        Myy = RMS[1, 1, :]
        Myw = RMS[1, 2, :]
        
        Mxx = Mxx - Mxw ** 2. / Mww
        Mxy = Mxy - Mxw * Myw / Mww
        Myy = Myy - Myw ** 2. / Mww
        MMxx = Mxx - Mxy ** 2. / Myy
    
        detM = MMxx * Myy * Mzz * Mww
    
        tqz = 1. / np.sqrt(Mzz)
        tqx = 1. / np.sqrt(MMxx)
        tqyy = 1. / np.sqrt(Myy)
        tqyx = -Mxy / Myy / np.sqrt(MMxx)
        tqww = 1. / np.sqrt(Mww)
        tqwy = -Myw / Mww / np.sqrt(Myy)
        tqwx = -(Mxw / Mww - Myw / Mww * Mxy / Myy) / np.sqrt(MMxx)
    
        inte = sqw(H, K, L, W, p)
        [modes, points] = inte.shape
    
        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, self, p)
                bgr = 0
            else: 
                raise ValueError('Invalid number or output arguments in prefactor function, pref')           
    
        found = 0
        if METHOD == 'fix':
            found = 1
            if ACCURACY is None:
                ACCURACY = np.array([7, 0])
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cw, cx, cy] = np.meshgrid(dd1, dd1, dd1, indexing='ij')
            tx = np.tan(cx)
            ty = np.tan(cy)
            tw = np.tan(cw) 
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * np.exp(-0.5 * (tw ** 2)) * (1 + tw ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            
            for iz in range(len(tz)):
                for i in range(length):
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm * normz[iz]
                        convs[j, i] = convs[j, i] + np.sum(add)
                        
                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])
        
            conv = conv * step1 ** 3 * step2 / np.sqrt(detM)
            if M[1] == 0:
                conv = conv * 0.79788
            if M[0] == 0:
                conv = conv * 0.79788 ** 3
        
        if METHOD == 'mc':
            found = 1
            if ACCURACY is None:
                ACCURACY = 10
        
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    r = np.random.randn(4, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    cw = r[3, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    tw = np.tan(cw)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2 + tw ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2) * (1 + tw ** 2)
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm
                        convs[j, i] = convs[j, i] + np.sum(add)
            
                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])
        
            conv = conv / M / 1000 * np.pi ** 4. / np.sqrt(detM) 
        
        if found == 0:
            raise ValueError('Unknown convolution METHOD. Valid options are: "fix",  "mc".')
               
        conv = conv * R0
        conv = conv + bgr

        return conv
    
    def resolution_convolution_SMA(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None):
        r'''Numerically calculate the convolution of a user-defined single-mode
        cross-section function with the resolution function for a 3-axis
        neutron scattering experiment.
        
        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.
        
        pref : func
            User-supplied "slow" cross section prefactor and background function.
        
        nargout : int
            Number of arguments returned by the pref function
        
        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e. 
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal lattice 
            units and W in meV.
        
        EXP : obj
            Instrument object containing all information on experimental setup.
        
        METHOD : str
            Specifies which 3D-integration method to use. 'fix' (Default): sample
            the cross section on a fixed grid of points uniformly distributed 
            $\phi$-space. 2*ACCURACY[0]+1 points are sampled along $\phi_1$,
            and $\phi_2$, and 2*ACCURACY[1]+1 along $\phi_3$ (vertical
            direction). 'mc': 3D Monte Carlo integration. The cross section is
            sampled in 1000*ACCURACY randomly chosen points, uniformly distributed
            in $\phi$-space.
        
        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.
        
        p : list
            A parameter that is passed on, without change to sqw and pref.
            
        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$
            
        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
        Oak Ridge National Laboratory
        
        '''        
        self.calc_resolution(hkle)
        [R0, RMS] = [np.copy(self.R0), np.copy(self.RMS)]
        
        H, K, L, W = hkle
        [length, H, K, L, W] = _CleanArgs(H, K, L, W);
        
        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem();        
        
        Mww = RMS[2, 2, :]
        Mxw = RMS[0, 2, :]
        Myw = RMS[1, 2, :]
        
        GammaFactor = np.sqrt(Mww / 2)
        OmegaFactorx = Mxw / np.sqrt(2 * Mww)
        OmegaFactory = Myw / np.sqrt(2 * Mww)
        
        Mzz = RMS[3, 3, :]
        Mxx = RMS[0, 0, :]
        Mxx = Mxx - Mxw ** 2 / Mww
        Myy = RMS[1, 1, :]
        Myy = Myy - Myw ** 2 / Mww
        Mxy = RMS[0, 1, :]
        Mxy = Mxy - Mxw * Myw / Mww
        
        detxy = np.sqrt(Mxx * Myy - Mxy ** 2)
        detz = np.sqrt(Mzz)
        
        tqz = 1. / detz
        tqy = np.sqrt(Mxx) / detxy
        tqxx = 1. / np.sqrt(Mxx)
        tqxy = Mxy / np.sqrt(Mxx) / detxy

        [disp, inte, WL0] = sqw(H, K, L, p)
        [modes, points] = disp.shape
        
        if pref is None:
            prefactor = np.ones(modes, points)
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, self, p)
                bgr = 0
            else:
                ValueError('Fata error: invalid number or output arguments in prefactor function')         
        
        found = 0
        if METHOD == 'mc': 
            found = 1
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M[0]):
                    r = np.random.randn(3, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2)
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000. * np.pi ** 3

        if METHOD == 'fix':
            found = 1
            if ACCURACY is None:
                ACCURACY = [7, 0]
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cy, cx] = np.meshgrid(dd1, dd1, indexing='ij')
            tx = np.tan(cx.flatten())
            ty = np.tan(cy.flatten())
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            for iz in range(tz.size):
                for i in range(length):
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm * normz[iz] / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 2 * step2
            
            if M[1] == 0:
                conv = conv * 0.79788
            if M[0] == 0:
                conv = conv * 0.79788 ** 2

        if found==0:
            ValueError('??? Error in ConvRes: Unknown convolution method! Valid options are: "fix" or "mc".')
        
        conv=conv*R0
        conv=conv+bgr
        
        return conv

    def plot_projections(self, hkle, npts=36):
        r'''Plots resolution ellipses in the QxQy, QxW, and QyW zones
        
        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer) values at which resolution ellipses are desired to be plotted
        
        npts : int, optional
            Number of points in an individual resolution ellipse. Default: 36
                    
        '''
        try:
            projections = self.projections
        except AttributeError:
            self.calc_projections(hkle, npts)
            projections = self.projections
        
        import matplotlib.pyplot as plt
        plt.rc('font', **{'family': 'serif', 'serif': 'Times New Roman', 'size': 12})
        plt.rc('lines', markersize=3, linewidth=0.5)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, facecolor='w', edgecolor='k')
        fig.subplots_adjust(bottom=0.175, left=0.15, right=0.85, top=0.95, wspace=0.35, hspace=0.25)
        
        for i in range(self.RMS.shape[-1]):
            ax1.plot(projections['QxQy'][0, :, i], projections['QxQy'][1, :, i])
            ax1.plot(projections['QxQySlice'][0, :, i], projections['QxQySlice'][1, :, i])

            ax2.plot(projections['QyW'][0, :, i], projections['QxW'][1, :, i])
            ax2.plot(projections['QyWSlice'][0, :, i], projections['QxWSlice'][1, :, i])
             
            ax3.plot(projections['QxW'][0, :, i], projections['QyW'][1, :, i])
            ax3.plot(projections['QxWSlice'][0, :, i], projections['QyWSlice'][1, :, i])
        
        ax1.set_xlabel('$\mathbf{Q}_x$ (r.l.u.)')
        ax1.set_ylabel('$\mathbf{Q}_y$ (r.l.u.)')
        ax1.locator_params(nbins=4)
        
        ax2.set_xlabel('$\mathbf{Q}_y$ (r.l.u.)')
        ax2.set_ylabel('$E$ (meV)')
        ax2.locator_params(nbins=4)
        
        ax3.set_xlabel('$\mathbf{Q}_x$ (r.l.u.)')
        ax3.set_ylabel('$E$ (meV)')
        ax3.locator_params(nbins=4)
        
        ax4.axis('off')
        
        plt.show()
