'''
Created on Jun 13, 2014

@author: davidfobes
'''
import numpy as np


def rc_int(index, r0, m):
    '''
    function [r,mp]=rc_int(index,r0,m)
    %
    % MATLAB function that takes a matrix and performs a Gaussian integral
    % over the row and column specified by index and returns
    % a new matrix. Tested against maple integration.
    %
    % ResCal5/A.T.
    '''
    r = np.sqrt(2 * np.pi / m[int(index), int(index)]) * r0

    # remove columns and rows from m
    # that contain the subscript "index".

    mp = m
    b = m[:, index] + m[index, :].T
    b = np.delete(b, index)
    mp = np.delete(mp, index, 0)
    mp = np.delete(mp, index, 1)
    mp = mp - 1 / (4 * m[index, index]) * b * b.T

    return [r, mp]


def res_projs(R0, RMS, hkle=[0, 0, 0, 0], mode='QxQy', n=36):
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

    #----- Work out projections for different cuts through the ellipse
    #----- S is the rotation matrix that diagonalises the projected ellipse

    #----- 1. Qx, Qy plane

    if mode == 'QxQy':
        [R0P, MP] = rc_int(2, R0, np.array(B))  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[0], hkle[1], n)

    #---------------- Add slice through Qx,Qy plane ----------------------

    if mode == 'QxQySlice':
        MP = A[:2:1, :2:1]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[0], hkle[1], n)

    #----- 2. Qx, W plane

    if mode == 'QxW':
        [R0P, MP] = rc_int(1, R0, B)  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[0], hkle[3], n)

    #---------------- Add slice through Qx,W plane ----------------------

    if mode == 'QxWSlice':
        MP = [[A[0, 0], A[0, 3]], [A[3, 0], A[3, 3]]]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[0], hkle[3], n)

    #----- 3. Qy, W plane

    if slice == 'QyW':
        [R0P, MP] = rc_int(0, R0, B)  # @UnusedVariable

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[1], hkle[3], n)

    #---------------- Add slice through Qy,W plane ----------------------

    if slice == 'QyWSlice':
        MP = [[A[1, 1], A[1, 3]], [A[3, 1], A[3, 3]]]

        theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
        S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).T

        hwhm_xp = const / np.sqrt(MP[0, 0])
        hwhm_yp = const / np.sqrt(MP[1, 1])

        return rc_ellip(hwhm_xp, hwhm_yp, theta, hkle[1], hkle[3], n)


def rc_ellip(a, b, phi=0, x0=0, y0=0, n=36):
    '''
    function  [x,y] = rc_ellip(a,b,phi,x0,y0,n)
    # ELLIPSE  Plotting ellipse.
    #    ELLIPSE(A,B,PHI,X0,Y0,N)  Plots ellipse with
    #    semiaxes A, B, rotated by the angle PHI,
    #    with origin at X0, Y0 and consisting of N points
    #    (default 100).
    #    [X,Y] = ELLIPSE(...) Instead of plotting returns
    #    coordinates of the ellipse.

    #  Kirill K. Pankratov, kirill@plume.mit.edu
    #  03/21/95
    '''

    th = np.linspace(0, 2 * np.pi, n + 1)
    x = a * np.cos(th)
    y = b * np.sin(th)

    c = np.cos(phi)
    s = np.sin(phi)

    th = x * c - y * s + x0
    y = x * s + y * c + y0
    x = th

    return [x, y]
