# -*- coding: utf-8 -*-
r"""A collection of commonly used one- and two-dimensional functions in neutron scattering,

=============== ==========================================================
gaussian        Vector or matrix norm
gaussian2d      Inverse of a square matrix
lorentzian      Solve a linear system of equations
voigt           Determinant of a square matrix
resolution      Logarithm of the determinant of a square matrix
gaussian_ring   Solve linear least-squares problem
=============== ==========================================================


"""
import numpy as np
from scipy import special
from scipy.special import erf


def gaussian(p, q):
    r"""Returns an arbitrary number of Gaussian profiles.

    Parameters
    ----------
    p : ndarray
        Parameters for the Gaussian, in the following format:

            +-------+----------------------------+
            | p[0]  | Constant background        |
            +-------+----------------------------+
            | p[1]  | Linear background slope    |
            +-------+----------------------------+
            | p[2]  | Area under the first peak  |
            +-------+----------------------------+
            | p[3]  | Position of the first peak |
            +-------+----------------------------+
            | p[4]  | FWHM of the first peak     |
            +-------+----------------------------+
            | p[5]  | Area under the second peak |
            +-------+----------------------------+
            | p[...]| etc.                       |
            +-------+----------------------------+

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

    where the integral over the whole function is *a*, and

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

    """
    funct = p[0] + p[1] * q
    for i in range(int(len(p[2:]) / 3)):
        sigma = p[3 * i + 4] / (2. * np.sqrt(2. * np.log(2.)))

        funct += p[3 * i + 2] / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(q - p[3 * i + 3]) ** 2 / (2 * sigma ** 2))

    return funct


def gaussian2d(p, q):
    r"""Returns an arbitrary number of two-dimensional Gaussian profiles.

    Parameters
    ----------
    p : ndarray
        Parameters for the Gaussian, in the following format:

            +-------+------------------------------+
            | p[0]  | Constant background          |
            +-------+------------------------------+
            | p[1]  | Linear background slope      |
            +-------+------------------------------+
            | p[2]  | Volume under the first peak  |
            +-------+------------------------------+
            | p[3]  | X position of the first peak |
            +-------+------------------------------+
            | p[4]  | Y position of the first peak |
            +-------+------------------------------+
            | p[5]  | FWHM_x of the first peak     |
            +-------+------------------------------+
            | p[6]  | FWHM_y of the first peak     |
            +-------+------------------------------+
            | p[7]  | Area under the second peak   |
            +-------+------------------------------+
            | p[...]| etc.                         |
            +-------+------------------------------+

    q : tuple
        Tuple of two one-dimensional input arrays.

    Returns
    -------
    out : ndarray
        One dimensional Gaussian profile.

    Notes
    -----
    A Gaussian profile is defined as:

    .. math::    f(q) = \frac{a}{\sigma \sqrt{2\pi}} e^{-\left(\frac{(q_x-q_x0)^2}{2\sigma_x^2} + \frac{(q_y-q_y0)^2}{2\sigma_y^2}\right)},

    where the integral over the whole function is *a*, and

    .. math::    fwhm = 2 \sqrt{2 \ln{2}} \sigma.

    Examples
    --------
    Plot a single gaussian with an integrated intensity of 1, centered at (0, 0), and fwhm of 0.3:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> p = np.array([0., 0., 1., 0., 0., 0.3, 0.3])
    >>> x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    >>> z = gaussian(p, (x, y))
    >>> plt.pcolormesh(x, y, z)
    >>> plt.show()

    Plot two gaussians, equidistant from the origin with the same intensity and fwhm as above:

    >>> p = np.array([0., 0., 1., -0.3, -0.3, 0.3, 0.3, 1., 0.3, 0.3, 0.3, 0.3])
    >>> x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    >>> z = gaussian(p, x)
    >>> plt.pcolormesh(x, y, z)
    >>> plt.show()

    """
    x, y = q

    funct = p[0] + p[1] * (x + y)

    for i in range(int(len(p[2:]) // 5)):
        sigma_x = p[5 * i + 5] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_y = p[5 * i + 6] / (2. * np.sqrt(2. * np.log(2.)))

        funct += p[5 * i + 2] / (sigma_x * sigma_y * 2. * np.pi) * np.exp(
            -((x - p[5 * i + 3]) ** 2 / (2 * sigma_x ** 2) + (y - p[5 * i + 4]) ** 2 / (2 * sigma_y ** 2)))

    return funct


def lorentzian(p, q):
    u"""Returns an arbitrary number of Lorentz profiles.

    Parameters
    ----------
    p : ndarray
        Parameters for the Lorentzian, in the following format:

            +-------+----------------------------+
            | p[0]  | Constant background        |
            +-------+----------------------------+
            | p[1]  | Linear background slope    |
            +-------+----------------------------+
            | p[2]  | Area under the first peak  |
            +-------+----------------------------+
            | p[3]  | Position of the first peak |
            +-------+----------------------------+
            | p[4]  | FWHM of the first peak     |
            +-------+----------------------------+
            | p[5]  | Area under the second peak |
            +-------+----------------------------+
            | p[...]| etc.                       |
            +-------+----------------------------+

    q : ndarray
        One dimensional input array.

    Returns
    -------
    out : ndarray
        One dimensional Lorentzian profile.

    Notes
    -----
    A Lorentzian profile is defined as:

    .. math::    f(q) = \\frac{a}{\\pi} \\frac{\\frac{1}{2} \\Gamma}{(q-q_0)^2 + (\\frac{1}{2} \\Gamma)^2},

    where the integral over the whole function is *a*, and Γ is the full width at half maximum.

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

    """

    funct = p[0] + p[1] * q

    for i in range(int(len(p[2:]) / 3)):
        funct += p[3 * i + 2] / np.pi * 0.5 * p[3 * i + 4] / ((q - p[3 * i + 3]) ** 2 + (0.5 * p[3 * i + 4]) ** 2)

    return funct


def voigt(p, q):
    r"""Returns an arbitrary number of Voigt profiles, a Lorentz profile convoluted by a Gaussian.

    Parameters
    ----------
    p : ndarray
        Parameters for the Lorentzian, in the following format:

            +-------+------------------------------+
            | p[0]  | Constant background          |
            +-------+------------------------------+
            | p[1]  | Linear background slope      |
            +-------+------------------------------+
            | p[2]  | Area under the first peak    |
            +-------+------------------------------+
            | p[3]  | Position of the first peak   |
            +-------+------------------------------+
            | p[4]  | FWHM of the first Lorentzian |
            +-------+------------------------------+
            | p[5]  | FWHM of the first Gaussian   |
            +-------+------------------------------+
            | p[6]  | Area under the second peak   |
            +-------+------------------------------+
            | p[...]| etc.                         |
            +-------+------------------------------+

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

    """
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
    r"""Returns a gaussian profile using a resolution matrix generated for a Triple Axis Spectrometer.

    Parameters
    ----------
    p : ndarray
        Parameters for the resolution function, in the following format:

            +-------+------------------------------+
            | p[0]  | Constant background          |
            +-------+------------------------------+
            | p[1]  | Linear background slope      |
            +-------+------------------------------+
            | p[2]  | Volume under the first peak  |
            +-------+------------------------------+
            | p[3]  | X position of the first peak |
            +-------+------------------------------+
            | p[4]  | Y position of the first peak |
            +-------+------------------------------+
            | p[5]  | R\ :sub:`0`                  |
            +-------+------------------------------+
            | p[6]  | RM\ :sub:`xx`                |
            +-------+------------------------------+
            | p[7]  | RM\ :sub:`yy`                |
            +-------+------------------------------+
            | p[8]  | RM\ :sub:`xy`                |
            +-------+------------------------------+
            | p[9]  | Area under the second peak   |
            +-------+------------------------------+
            | p[...]| etc.                         |
            +-------+------------------------------+

    q : tuple of ndarray
        Two input arrays of equivalent size and shape.

    Returns
    -------
    out : ndarray
        Two dimensional resolution profile with shape of input arrays.

    Notes
    -----
    A resolution profile is defined as a two dimensional gaussian that is comprised of elements of a
    resolution matrix for a triple axis spectrometer, as produced by :py:meth:`.Instrument.calc_resolution`

    .. math::    f(q) = R_0 e^{-\frac{1}{2}(RM_{xx}^2 (x-x_0)^2 + RM_{yy}^2 (y-y_0)^2 + 2RM_{xy}(x-x_0)(y-y_0))},

    where RM is the resolution matrix.

    """
    funct = p[0] + p[1] * (q[0] + q[1])

    if mode == 'gaussian':
        for i in range(int(len(p[2:]) / 7)):
            # Normalization pre-factor
            N = (np.sqrt(p[7 * i + 6]) * np.sqrt(p[7 * i + 7] - p[7 * i + 8] ** 2 / p[7 * i + 6])) / (
                2. * np.pi * p[7 * i + 5])

            funct += p[7 * i + 2] * p[7 * i + 5] * N * np.exp(-1. / 2. * (p[7 * i + 6] * (q[0] - p[7 * i + 3]) ** 2 +
                                                                          p[7 * i + 7] * (q[1] - p[7 * i + 4]) ** 2 +
                                                                          2. * p[7 * i + 8] * (q[0] - p[7 * i + 3]) * (
                                                                          q[1] - p[7 * i + 4])))

    return funct


def gaussian_ring(p, q):
    r"""Returns a two dimensional gaussian ellipse profile.

    Parameters
    ----------
    p : ndarray
        Parameters for the gaussian ellipse function, in the following format:

            +-------+------------------------------+
            | p[0]  | Constant background          |
            +-------+------------------------------+
            | p[1]  | Linear background slope      |
            +-------+------------------------------+
            | p[2]  | Volume under first ellipse   |
            +-------+------------------------------+
            | p[3]  | X position of first ellipse  |
            +-------+------------------------------+
            | p[4]  | Y position of first ellipse  |
            +-------+------------------------------+
            | p[5]  | Radius of first ellipse      |
            +-------+------------------------------+
            | p[6]  | Eccentricity of first ellipse|
            +-------+------------------------------+
            | p[7]  | FWHM of first ellipse        |
            +-------+------------------------------+
            | p[8]  | Volume under second ellipse  |
            +-------+------------------------------+
            | p[...]| etc.                         |
            +-------+------------------------------+

    q : tuple of ndarray
        Two input arrays of equivalent size and shape, e.g. formed with :py:func:`numpy.meshgrid`.

    Returns
    -------
    out : ndarray
        Two dimensional gaussian ellipse profile.

    Notes
    -----
    A gaussian ellipse profile is defined as

    .. math::    f(x,y) = \frac{1}{N} e^{-\frac{1}{2}\frac{(\sqrt{(x-x_0)^2 + \alpha^2(y-y_0)^2}-r_0)^2}{2 \sigma}},

    where :math:`FWHM = 2\sqrt{2\ln(2)}\sigma`, and N is the normalization pre-factor given by

    .. math::    N = \frac{2\pi}{\alpha} \left(\sigma^2 e^{-\frac{r_0^2}{2\sigma^2}} + \sqrt{\frac{\pi}{2}} r_0 \sigma \left(1 + \mathrm{Erf}\left(\frac{r_0}{\sqrt{2}\sigma}\right)\right)\right).

    """
    x, y = q

    funct = p[0] + p[1] * (x + y)

    for i in range(int(len(p[2:]) / 6)):
        # Normalization pre-factor
        sigma = p[6 * i + 7] / (2. * np.sqrt(2. * np.log(2.)))
        N = 2. * np.pi * (np.exp(-p[6 * i + 5] ** 2 / (2. * sigma ** 2)) *
                          sigma ** 2 + np.sqrt(np.pi / 2) * p[6 * i + 5] *
                          sigma * (1. + erf(p[6 * i + 5] / (np.sqrt(2) * sigma)))) / p[6 * i + 6]

        funct += p[6 * i + 2] / N * np.exp(-4. * np.log(2.) * (np.sqrt((x - p[6 * i + 3]) ** 2 +
                                                                       p[6 * i + 6] ** 2 * (y - p[6 * i + 4]) ** 2) -
                                                               p[6 * i + 5]) ** 2 / p[6 * i + 7] ** 2)

    return funct
