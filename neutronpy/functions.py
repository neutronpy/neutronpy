import numpy as np
from scipy import special
from scipy.special import erf


def gaussian(p, q):
    r'''Returns an arbitrary number of Gaussian profiles.

    Parameters
    ----------
    p : ndarray
        Parameters for the Gaussian, in the following format:

        * p[0] : Constant background
        * p[1] : Linear background slope
        * p[2] : Area under the first peak
        * p[3] : Position of the first peak
        * p[4] : FWHM of the first peak
        * p[5] : Area under the second peak
        * p[...] : etc.

    q : ndarray
        One dimensional input array.

    Returns
    -------
    out : ndarray
        One dimensional Gaussian profile.

    Notes
    -----
    A Gaussian profile is defined as:

    .. math::    f(q) = \frac{a}{\sigma \sqrt{2\pi}} e^{-\frac{(q-q_0)^2}{2\sigma^2}},

    where the integral over the whole function is :math:`a`, and

    .. math::    fwhm = 2 \sqrt{2 \ln{2}} \sigma.

    Examples
    --------
    Plot a single gaussian with an integrated intensity of 1, centered at zero, and fwhm of 0.3:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> p = np.array([0., 0., 1., 0., 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = gaussian(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    Plot two gaussians, equidistant from the origin with the same intensity and fwhm as above:

    >>> p = np.array([0., 0., 1., -0.3, 0.3, 1., 0.3, 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = gaussian(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    '''
    funct = p[0] + p[1] * q
    for i in range(int(len(p[2:]) / 3)):
        sigma = p[3 * i + 4] / (2. * np.sqrt(2. * np.log(2.)))

        funct += p[3 * i + 2] / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(q - p[3 * i + 3]) ** 2 / (2 * sigma ** 2))

    return funct


def lorentzian(p, q):
    r'''Returns an arbitrary number of Lorentz profiles.

    Parameters
    ----------
    p : ndarray
        Parameters for the Lorentzian, in the following format:

        * p[0] : Constant background
        * p[1] : Linear background slope
        * p[2] : Integrated intensity of the first peak
        * p[3] : Position of the first peak
        * p[4] : FWHM of the first peak
        * p[5] : Integrated intensity of the second peak
        * p[...] : etc.

    q : ndarray
        One dimensional input array.

    Returns
    -------
    out : ndarray
        One dimensional Lorentzian profile.

    Notes
    -----
    A Lorentzian profile is defined as:

    .. math::    f(q) = \frac{a}{\pi} \frac{\frac{1}{2} \Gamma}{(q-q_0)^2 + (\frac{1}{2} \Gamma)^2},

    where the integral over the whole function is `a`, and :math:`\Gamma` is the full width at half maximum.

    Examples
    --------
    Plot a single lorentzian with an integrated intensity of 1, centered at zero, and fwhm of 0.3:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> p = np.array([0., 0., 1., 0., 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = lorentzian(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    Plot two lorentzians, equidistant from the origin with the same intensity and fwhm as above:

    >>> p = np.array([0., 0., 1., -0.3, 0.3, 1., 0.3, 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = lorentzian(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    '''

    funct = p[0] + p[1] * q

    for i in range(int(len(p[2:]) / 3)):
        funct += p[3 * i + 2] / np.pi * 0.5 * p[3 * i + 4] / ((q - p[3 * i + 3]) ** 2 + (0.5 * p[3 * i + 4]) ** 2)

    return funct


def voigt(p, q):
    r'''Returns an arbitrary number of Voigt profiles, a Lorentz profile convoluted by a Gaussian.

    Parameters
    ----------
    p : ndarray
        Parameters for the Lorentzian, in the following format:

        * p[0] : flat background term
        * p[1] : sloping background
        * p[2] : Area under the first Lorentzian
        * p[3] : Position of first Lorentzian
        * p[4] : FWHM of first Lorentzian
        * p[5] : FWHM of first Gaussian resolution
        * p[6] : Area under the second Lorentzian
        * p[...] : etc.

    q : ndarray
        One dimensional input array.

    Returns
    -------
    out : ndarray
        One dimensional Voigt profile.

    Notes
    -----
    A Voigt profile is defined as a convolution of a Lorentzian profile with a Gaussian Profile:

    .. math::    V(x;\sigma,\gamma)=\int_{-\infty}^\infty G(x';\sigma)L(x-x';\gamma)\, dx'.

    Examples
    --------
    Plot a single Voigt profile with an integrated intensity of 1, centered at zero, and FWHM = 0.2 convoluted with a Gaussian with FWHM = 0.3:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> p = np.array([0., 0., 1., 0., 0.2, 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = voigt(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    Plot two Voigt profiles, equidistant from the origin with the same intensity and fwhm as above:

    >>> p = np.array([0., 0., 1., -0.3, 0.2, 0.3, 1., 0.3, 0.2, 0.3])
    >>> x = np.linspace(-1, 1, 101)
    >>> y = voigt(p, x)
    >>> plt.plot(x, y)
    >>> plt.show()

    '''

    funct = p[0] + p[1] * q
    for i in range(int(len(p[2:]) / 4)):
        sigma = p[4 * i + 5] / (2. * np.sqrt(2. * np.log(2.)))
        gamma = p[4 * i + 4] / 2.

        # Normalization pre-factor
        N = (sigma * np.sqrt(2 * np.pi))

        funct += p[4 * i + 2] / N * np.real(special.wofz(((q - p[4 * i + 3]) + 1j * gamma) /
                                                         (sigma * np.sqrt(2))))

    return funct


def resolution(p, q, mode='gaussian'):
    r'''Returns a gaussian profile using a resolution matrix generated for a Triple Axis Spectrometer.

    Parameters
    ----------
    p : ndarray
        Parameters for the resolution function, in the following format:

        * p[0] : Constant background
        * p[1] : Linear background slope
        * p[2] : Area under first gaussian
        * p[3] : x position of first gaussian
        * p[4] : y position of first gaussian
        * p[5] : R0
        * p[6] : RM_xx (Resolution matrix 1st dimension diagonal element)
        * p[7] : RM_yy (Resolution matrix 2nd dimension diagonal element)
        * p[8] : RM_xy (Resolution matrix off diagonal element)
        * p[9] : Area under the second Gaussian/Voigt profile
        * p[...] : etc.

    q : tuple of ndarray
        Two input arrays of equivalent size and shape.

    Returns
    -------
    out : ndarray
        Two dimensional resolution profile with shape of input arrays.

    Notes
    -----
    A resolution profile is defined as a two dimensional gaussian that is comprised of elements of a
    resolution matrix for a triple axis spectrometer, as produced by neutronpy.resolution.res_calc():

    .. math::    f(q) = R_0 e^{-\frac{1}{2}(RM_{xx}^2 (x-x_0)^2 + RM_{yy}^2 (y-y_0)^2 + 2RM_{xy}(x-x_0)(y-y_0))},

    where RM is the resolution matrix.

    '''
    x, y = q

    funct = p[0] + p[1] * (x + y)

    if mode == 'gaussian':
        for i in range(int(len(p[2:]) / 7)):
            # Normalization pre-factor
            N = (np.sqrt(p[7 * i + 6]) * np.sqrt(p[7 * i + 7] - p[7 * i + 8] ** 2 / p[7 * i + 6])) / (2. * np.pi)

            funct += p[7 * i + 2] * p[7 * i + 5] / N * np.exp(-1. / 2. * (p[7 * i + 6] * (x - p[7 * i + 3]) ** 2 +
                                                                          p[7 * i + 7] * (y - p[7 * i + 4]) ** 2 +
                                                                          2. * p[7 * i + 8] * (x - p[7 * i + 3]) * (y - p[7 * i + 4])))

    return funct


def gaussian_ring(p, q):
    r'''Returns a two dimensional gaussian ellipse profile.

    Parameters
    ----------
    p : ndarray
        Parameters for the gaussian ellipse function, in the following format:

        * p[0] : Constant background
        * p[1] : Linear background slope
        * p[2] : Area under first gaussian ellipse
        * p[3] : x position of first gaussian ellipse
        * p[4] : y position of first gaussian ellipse
        * p[5] : Radius of the first gaussian ellipse
        * p[6] : Eccentricity of the first gaussian ellipse
        * p[7] : FWHM of first gaussian ellipse
        * p[8] : Area under the second gaussian ellipse
        * P[...] : etc.

    q : tuple of ndarray
        Two input arrays of equivalent size and shape, e.g. formed with np.meshgrid().

    Returns
    -------
    out : ndarray
        Two dimensional gaussian ellipse profile.

    Notes
    -----
    A gaussian ellipse profile is defined as:

    .. math::    f(x,y) = \frac{1}{N} e^{-\frac{1}{2}\frac{(\sqrt{(x-x_0)^2 + \alpha^2(y-y_0)^2}-r_0)^2}{2 \sigma}},

    where :math:`FWHM = 2\sqrt{2\log(2)}`, and N is the normalization pre-factor given by:

    .. math::    N =

    '''
    x, y = q

    funct = p[0] + p[1] * (x + y)

    for i in range(int(len(p[2:] / 6))):
        # Normalization pre-factor
        N = 1. / ((2. * np.pi / p[6 * i + 6] ** 2) * (p[6 * i + 7] ** 2 / (8. * np.log(2.))) *
                  np.exp(-4. * np.log(2.) * p[6 * i + 5] ** 2 / p[6 * i + 7] ** 2) +
                  np.sqrt(np.pi / np.log(2.)) * p[6 * i + 5] *
                  (1. + erf(4. * np.sqrt(np.log(2.)) * p[6 * i + 5] / p[6 * i + 7])))

        funct += p[6 * i + 2] * N * np.exp(-4. * np.log(2.) * (np.sqrt((x - p[6 * i + 3]) ** 2 +
                                                                       p[6 * i + 6] ** 2 * (y - p[6 * i + 4]) ** 2) -
                                                               p[6 * i + 5]) ** 2 / p[6 * i + 7])

    return funct
