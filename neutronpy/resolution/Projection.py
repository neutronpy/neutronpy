'''
Created on Jun 13, 2014

@author: davidfobes
'''
import numpy as np


def project_into_plane(rm, index):
    '''Projects out-of-plane resolution into a specified plane by performing
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

    out = rm
    gauss = rm[:, index] + rm[index, :].T
    gauss = np.delete(gauss, index)
    out = np.delete(out, index, 0)
    out = np.delete(out, index, 1)
    out = out - (1. / (4. * rm[index, index]) * gauss * gauss.T)

    return out


def projections(R0, RMS, hkle=[0, 0, 0, 0], mode='QxQy', npts=31):
    '''
    %
    % MATLAB function to plot the projections of the resolution ellipse
    % of a triple axis
    %
    % Input:
    %  out:  EXP ResLib structure
    %  mode: can be set to 'rlu' so that the plot is in lattice RLU frame
    %
    % DFM 10.11.95
    '''
    NP = RMS

    A = np.array(NP)
    const = 1.17741  # half width factor

    #----- Remove the vertical component from the matrix.
    B = np.vstack((np.hstack((A[0, :2:1], A[0, 3])),
                   np.hstack((A[1, :2:1], A[1, 3])),
                   np.hstack((A[3, :2:1], A[3, 3]))))

    # Projection into Qx, Qy plane

    if mode == 'QxQy':
        MP = project_into_plane(B, 2)  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0], hkle[1]], npts)

    # Slice through Qx,Qy plane

    if mode == 'QxQySlice':
        MP = A[:2:1, :2:1]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0], hkle[1]], npts)

    # Projection into Qx, W plane

    if mode == 'QxW':
        MP = project_into_plane(B, 1)  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0], hkle[3]], npts)

    # Slice through Qx,W plane

    if mode == 'QxWSlice':
        MP = [[A[0, 0], A[0, 3]], [A[3, 0], A[3, 3]]]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0], hkle[3]], npts)

    # Projections into Qy, W plane

    if slice == 'QyW':
        MP = project_into_plane(B, 0)  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1], hkle[3]], npts)

    # Slice through Qy,W plane

    if slice == 'QyWSlice':
        MP = [[A[1, 1], A[1, 3]], [A[3, 1], A[3, 3]]]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1], hkle[3]], npts)


def ellipse(saxis1, saxis2, phi=0, origin=[0, 0], npts=31):
    '''Returns an ellipse.

    Parameters
    ----------
    saxis1 : float
        First semiaxis

    saxis2 : float
        Second semiaxis

    phi : float, optional
        Angle that semiaxes are rotated from +x

    origin : list of floats, optional
        Origin position [x0, y0]

    npts: float, optional
        Number of points in the output arrays.

    Returns
    -------
    [x, y] : list of ndarray
        Two one dimensional arrays representing an ellipse
    '''

    theta = np.linspace(0., 2. * np.pi, npts + 1)

    x = saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi) + origin[0]
    y = saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi) + origin[1]

    return [x, y]
