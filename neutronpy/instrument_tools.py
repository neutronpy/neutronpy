import numpy as np


def project_into_plane(index, r0, rm):
    r'''Projects out-of-plane resolution into a specified plane by performing
    a gaussian integral over the third axis.

    Parameters
    ----------
    index : int
        Index of the axis that should be integrated out

    r0 : float
        Resolution prefactor

    rm : ndarray
        Resolution array

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

    x = np.array(saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi)) + origin[0]
    y = np.array(saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi)) + origin[1]
    return np.vstack((x, y))


def get_bragg_widths(RM):
    bragg = np.array([np.sqrt(8 * np.log(2)) / np.sqrt(RM[0, 0]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[1, 1]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[2, 2]),
                      get_phonon_width(0, RM, [0, 0, 0, 1])[1],
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[3, 3])])

    return bragg * 2


def get_phonon_width(r0, M, C):
    T = np.diag(np.ones(4))
    T[3, :] = np.array(C)
    S = np.matrix(np.linalg.inv(T))
    MP = S.H * M * S
    [rp, MP] = project_into_plane(0, r0, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    fwhm = np.sqrt(8 * np.log(2)) / np.sqrt(MP[0, 0])

    return [rp, fwhm]


def fproject(mat, i):
    if i == 0:
        v = 2
        j = 1
    if i == 1:
        v = 0
        j = 2
    if i == 2:
        v = 0
        j = 1
    [a, b, c] = mat.shape
    proj = np.zeros((2, 2, c))
    proj[0, 0, :] = mat[i, i, :] - mat[i, v, :] ** 2 / mat[v, v, :]
    proj[0, 1, :] = mat[i, j, :] - mat[i, v, :] * mat[j, v, :] / mat[v, v, :]
    proj[1, 0, :] = mat[j, i, :] - mat[j, v, :] * mat[i, v, :] / mat[v, v, :]
    proj[1, 1, :] = mat[j, j, :] - mat[j, v, :] ** 2 / mat[v, v, :]
    hwhm = proj[0, 0, :] - proj[0, 1, :] ** 2 / proj[1, 1, :]
    hwhm = np.sqrt(2. * np.log(2.)) / np.sqrt(hwhm)

    return hwhm
