# -*- coding: utf-8 -*-
import datetime as dt

import numpy as np

from ..energy import Energy
from .tools import _CleanArgs, fproject, get_bragg_widths


class PlotTasInstrument(object):
    r"""Class containing resolution plotting methods

    Methods
    -------
    plot_projections
    plot_ellipsoid
    plot_instrument
    plot_slice
    description_string

    """

    def plot_projections(self, hkle, npts=36, dpi=100):
        r"""Plots resolution ellipses in the QxQy, QxW, and QyW zones

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer)
            values at which resolution ellipses are desired to be plotted

        npts : int, optional
            Number of points in an individual resolution ellipse. Default: 36

        dpi : int, optional
            Specify DPI of figure. Default: 100

        """
        [H, K, L, W] = hkle

        if hasattr(self, "projections"):
            if np.all(H == self.H) and np.all(K == self.K) and np.all(L == self.L) and np.all(W == self.W):
                projections = self.projections
            else:
                self.calc_projections(hkle, npts)
                projections = self.projections
        else:
            self.calc_projections(hkle, npts)
            projections = self.projections

        import matplotlib.pyplot as plt

        plt.rc('font', **{'family': 'Bitstream Vera Sans', 'serif': 'cm10', 'size': 6})
        plt.rc('lines', markersize=3, linewidth=1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, facecolor='w', edgecolor='k', dpi=dpi)
        fig.subplots_adjust(bottom=0.175, left=0.15, right=0.85, top=0.95, wspace=0.35, hspace=0.25)

        ax1_lims, ax2_lims, ax3_lims = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]

        if len(self.RMS.shape) == 3:
            length = self.RMS.shape[-1]
        else:
            length = 1

        for i in range(length):
            for ax, slice_str, lims in zip([ax1, ax2, ax3], ['QxQy', 'QxW', 'QyW'], [ax1_lims, ax2_lims, ax3_lims]):
                self.plot_slice(ax, slice_str, projections, self.sample.u, self.sample.v, num=i)

                xlim = ax.get_xlim()
                if xlim[0] < lims[0][0] or i == 0:
                    lims[0][0] = xlim[0]
                if xlim[1] > lims[0][1] or i == 0:
                    lims[0][1] = xlim[1]

                ylim = ax.get_ylim()
                if ylim[0] < lims[1][0] or i == 0:
                    lims[1][0] = ylim[0]
                if ylim[1] > lims[1][1] or i == 0:
                    lims[1][1] = ylim[1]

                ax.set_xlim(lims[0])
                ax.set_ylim(lims[1])

        ax4.axis('off')
        ax4.text(0, 1, self.description_string(), transform=ax4.transAxes, horizontalalignment='left',
                 verticalalignment='top')

        plt.show()

    def plot_ellipsoid(self, hkle, dpi=100):
        r"""Plots the resolution ellipsoid in the $Q_x$, $Q_y$, $W$ zone

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer)
            values at which resolution ellipsoid are desired to be plotted

        dpi : int, optional
            Number of points in the plot

        """
        try:
            from skimage import measure
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise

        # clean arguments
        [length, H, K, L, W] = _CleanArgs(*hkle)

        # check if resolution already calculated at given hkle
        try:
            if np.all(H == self.H) and np.all(K == self.K) and np.all(L == self.L) and np.all(W == self.W):
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

        if NP.shape == (4, 4):
            NP = NP[np.newaxis]
            R0 = [R0]
        else:
            NP = NP.T

        # Create a canvas with a 3D viewport
        fig = plt.figure(facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, projection='3d')

        for ind, (r0, rms) in enumerate(zip(R0, NP)):
            qx0, qy0 = [np.dot([H[ind], K[ind], L[ind]], self.orient1 / np.linalg.norm(self.orient1) ** 2),
                        np.dot([H[ind], K[ind], L[ind]], self.orient2 / np.linalg.norm(self.orient2) ** 2)]
            qw0 = W[ind]

            def resolution_ellipsoid(prefac, resmat, x, y, w, x0, y0, w0):
                r"""Resolution ellipsoid equation for qx, qy, and w dimensions

                Parameters
                ----------
                prefac : float
                resmat : ndarray(4,4)
                x : ndarray
                y : ndarray
                w : ndarray
                x0 : float
                y0 : float
                w0 : float

                Returns
                -------
                    res_func : ndarray
                """
                ee = (resmat[0, 0] * (x - x0) ** 2 +
                      resmat[1, 1] * (y - qy0) ** 2 +
                      resmat[3, 3] * (w - w0) ** 2 +
                      2 * resmat[0, 1] * (x - x0) * (y - y0) +
                      2 * resmat[0, 2] * (x - x0) * (w - w0) +
                      2 * resmat[1, 2] * (y - y0) * (w - w0))

                return prefac * np.exp(-1 / 2 * ee)

            # get fwhm to generate grid for plotting
            w = []
            for i in range(3):
                _rms = np.delete(np.delete(rms, 2, axis=0), 2, axis=1)
                w.append(fproject(_rms.reshape((3, 3, 1)), i)[0] * 1.01)
            wx, wy, ww = w

            # build grid
            xg, yg, zg = np.mgrid[qx0 - wx:qx0 + wx:(dpi + 1) * 1j,
                                  qy0 - wy:qy0 + wy:(dpi + 1) * 1j,
                                  qw0 - ww:qw0 + ww:(dpi + 1) * 1j]

            # resolution function (ellipsoidal gaussian)
            vol = resolution_ellipsoid(r0, rms, xg, yg, zg, qx0, qy0, qw0)

            # isosurface of resolution function at half-max
            vertices, faces, normals, values = measure.marching_cubes(vol, vol.max() / 2.0, spacing=(wx / dpi, wy / dpi, ww / dpi))

            # properly center ellipsoid
            x_verts = vertices[:, 0] + qx0 - wx / 2
            y_verts = vertices[:, 1] + qy0 - wy / 2
            z_verts = vertices[:, 2] + qw0 - ww / 2

            # plot ellipsoid
            ax.plot_trisurf(x_verts, y_verts, faces, z_verts, lw=0, cmap='Spectral')

        # figure labels
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.set_xlabel(r'$q_x$ (along {0}) (r.l.u.)'.format(self.orient1), fontsize=12)
        ax.set_ylabel(r'$q_y$ (along {0}) (r.l.u.)'.format(self.orient2), fontsize=12)
        ax.set_zlabel(r'$\hbar \omega$ (meV)', fontsize=12)

        plt.show()

    def plot_instrument(self, hkle):
        r"""Plots the instrument configuration using angles for a given position
        in Q and energy transfer

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer)
            values at which the instrument setup should be plotted

        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise

        fig = plt.figure(edgecolor='k', facecolor='w', figsize=plt.figaspect(0.4) * 1.25)
        ax = fig.gca(projection='3d')

        measurements = {'guide.width': 1, 'guide.height': 1, 'mono.width': 1, 'mono.height': 1, 'sample.width': 1,
                        'sample.height': 1, 'sample.depth': 1, 'ana.width': 1, 'ana.height': 1, 'detector.width': 1,
                        'detector.height': 1, 'arms': [10, 10, 10, 10]}

        for key, value in measurements.items():
            if '.' in key:
                if hasattr(getattr(self, key.split('.')[0]), key.split('.')[1]):
                    measurements[key] = getattr(getattr(self, key.split('.')[0]), key.split('.')[1])
            else:
                if hasattr(self, key):
                    measurements[key] = getattr(self, key)

        angles = self.get_angles_and_Q(hkle)[0]
        distances = measurements['arms']

        angles = -np.deg2rad(angles)
        x, y, direction = np.zeros(3)

        # plot the Source -----------------------------------------------------
        translate = 0
        rotate = 0 * (np.pi / 180)
        direction += rotate
        x += translate * np.sin(direction)
        y += translate * np.cos(direction)

        # create a square source
        X = np.array(
            [-measurements['guide.width'] / 2, -measurements['guide.width'] / 2, measurements['guide.width'] / 2,
             measurements['guide.width'] / 2, -measurements['guide.width'] / 2])
        Z = np.array(
            [measurements['guide.height'] / 2, -measurements['guide.height'] / 2, -measurements['guide.height'] / 2,
             measurements['guide.height'] / 2, measurements['guide.height'] / 2])
        Y = np.zeros(5)
        l = ax.plot(X + x, Y + y, zs=Z, color='b')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Beam/Source', color='b')

        x0, y0 = x, y

        # plot the Monochromator ----------------------------------------------
        translate = distances[0]
        rotate = 0
        direction += rotate
        x += translate * np.sin(direction)
        y += translate * np.cos(direction)

        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square Monochromator
        X = np.array([-measurements['mono.width'] / 2, -measurements['mono.width'] / 2, measurements['mono.width'] / 2,
                      measurements['mono.width'] / 2, -measurements['mono.width'] / 2]) * np.sin(angles[0])
        Z = np.array(
            [measurements['mono.height'] / 2, -measurements['mono.height'] / 2, -measurements['mono.height'] / 2,
             measurements['mono.height'] / 2, measurements['mono.height'] / 2])
        Y = X * np.cos(angles[0])
        l = ax.plot(X + x, Y + y, zs=Z, color='r')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Monochromator', color='r')

        x0, y0 = x, y

        # plot the Sample -----------------------------------------------------
        translate = distances[1]

        rotate = angles[1]
        direction += rotate
        x += translate * np.sin(direction)
        y += translate * np.cos(direction)

        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a rotated square Sample
        X = np.array(
            [-measurements['sample.width'] / 2, -measurements['sample.width'] / 2, measurements['sample.width'] / 2,
             measurements['sample.width'] / 2, -measurements['sample.width'] / 2]) * np.sin(angles[2])
        Z = np.array(
            [measurements['sample.height'] / 2, -measurements['sample.height'] / 2, -measurements['sample.height'] / 2,
             measurements['sample.height'] / 2, measurements['sample.height'] / 2])
        Y = X * np.cos(angles[2])
        l1 = ax.plot(X + x, Y + y, zs=Z, color='g')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Sample', color='g')
        X = np.array(
            [-measurements['sample.depth'] / 2, -measurements['sample.depth'] / 2, measurements['sample.depth'] / 2,
             measurements['sample.depth'] / 2, -measurements['sample.depth'] / 2]) * np.sin(
            angles[2] + np.pi / 2)
        Z = np.array(
            [measurements['sample.height'] / 2, -measurements['sample.height'] / 2, -measurements['sample.height'] / 2,
             measurements['sample.height'] / 2, measurements['sample.height'] / 2])
        Y = X * np.cos(angles[2] + np.pi / 2)
        l2 = ax.plot(X + x, Y + y, zs=Z, color='g')

        x0, y0 = x, y

        # plot the Analyzer ---------------------------------------------------
        translate = distances[2]

        rotate = angles[3]
        direction += rotate

        x += translate * np.sin(direction)
        y += translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square
        X = np.array([-measurements['ana.width'] / 2, -measurements['ana.width'] / 2, measurements['ana.width'] / 2,
                      measurements['ana.width'] / 2, -measurements['ana.width'] / 2]) * np.sin(angles[4])
        Z = np.array([measurements['ana.height'] / 2, -measurements['ana.height'] / 2, -measurements['ana.height'] / 2,
                      measurements['ana.height'] / 2, measurements['ana.height'] / 2])
        Y = X * np.cos(angles[4])
        l = ax.plot(X + x, Y + y, zs=Z, color='magenta')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Analyzer', color='magenta')

        x0, y0 = x, y

        # plot the Detector ---------------------------------------------------
        translate = distances[3]
        rotate = angles[5]
        direction += rotate

        x += translate * np.sin(direction)
        y += translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square
        X = np.array(
            [-measurements['detector.width'] / 2, -measurements['detector.width'] / 2,
             measurements['detector.width'] / 2, measurements['detector.width'] / 2,
             -measurements['detector.width'] / 2])
        Z = np.array(
            [measurements['detector.height'] / 2, -measurements['detector.height'] / 2,
             -measurements['detector.height'] / 2, measurements['detector.height'] / 2,
             measurements['detector.height'] / 2])
        Y = np.zeros(5)
        l = ax.plot(X + x, Y + y, zs=Z, color='k')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Detector', color='k')

        ax.set_zlim3d(getattr(ax, 'get_zlim')()[0], getattr(ax, 'get_zlim')()[1] * 10)

        plt.show()

    @staticmethod
    def plot_slice(axis, qslice, projections, u, v, num=0):

        r"""Class method for plotting individual projections. Plots both
        slices and projections.

        Parameters
        ----------
        axis : matplotlib_axis
            Axis to which to plot the selected projection

        qslice : str
            The projection to plot. 'QxQy', 'QxW' or 'QyW'.

        projections : dict
            Projections as generated by :py:meth:`.Instrument.calc_projections`

        u : ndarray(3)
            First orientation vector.

        v : ndarray(3)
            Second orientation vector.

        num : int, optional
            If multiple projections are present, choose which one to plot

        """
        dQ1, dQ2 = [], []

        axis.fill(projections[qslice][0, :][:, num], projections[qslice][1, :][:, num], zorder=0, alpha=0.5,
                  edgecolor='none')
        axis.plot(projections[qslice + 'Slice'][0, :][:, num], projections[qslice + 'Slice'][1, :][:, num], zorder=1)

        dQ1.append(np.max(projections[qslice][0, :][:, num]) - np.min(projections[qslice][0, :][:, num]))
        dQ2.append(np.max(projections[qslice][1, :][:, num]) - np.min(projections[qslice][1, :][:, num]))

        axis.set_xlim(np.min(projections[qslice][0, :][:, num]), np.max(projections[qslice][0, :][:, num]))
        axis.set_ylim(np.min(projections[qslice][1, :][:, num]), np.max(projections[qslice][1, :][:, num]))

        dQ1, dQ2 = [np.max(item) for item in [dQ1, dQ2]]

        axis.set_xlabel(r'$\mathbf{Q}_1$ (along ' + str(u) + r') (r.l.u.)' + r', $\delta Q_1={0:.3f}$'.format(dQ1),
                        fontsize=7)
        if 'W' in qslice:
            axis.set_ylabel(r'$\hbar \omega$ (meV)' + r', $\delta \hbar \omega={0:.3f}$'.format(dQ2), fontsize=7)
        else:
            axis.set_ylabel(r'$\mathbf{Q}_2$ (along ' + str(v) + r') (r.l.u.)' + r', $\delta Q_2={0:.3f}$'.format(dQ2),
                            fontsize=7)

    def description_string(self):
        r"""Generates text string describing most recent resolution calculation

        """
        try:
            method = ['Cooper-Nathans', 'Popovici'][self.method]
        except AttributeError:
            method = 'Cooper-Nathans'
        frame = '[Q1,Q2,Qz,E]'

        if hasattr(self, 'infin'):
            FX = 2 * int(self.infin == -1) + int(self.infin == 1)
        else:
            FX = 2

        if self.RMS.shape == (4, 4):
            NP = self.RMS
            R0 = float(self.R0)
            hkle = self.HKLE
        else:
            NP = self.RMS[:, :, 0]
            R0 = self.R0[0]
            hkle = [self.H[0], self.K[0], self.L[0], self.W[0]]

        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP))
        bragg_widths = get_bragg_widths(NP)
        angles = self.get_angles_and_Q(hkle)[0]

        text_format = ['Method: {0}'.format(method),
                       'Position HKLE [{0}]\n'.format(dt.datetime.now().strftime('%d-%b-%Y %H:%M:%S')),
                       ' [Q_H, Q_K, Q_L, E] = {0} \n'.format(self.HKLE),
                       'Resolution Matrix M in {0} (M/10^4):'.format(frame),
                       '[[{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 0] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 1] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 2] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]]\n'.format(*NP[:, 3] / 1.0e4),
                       'Resolution volume:   V_0={0:.6f} meV/A^3'.format(2 * ResVol),
                       'Intensity prefactor: R_0={0:.3f}'.format(R0),
                       'Bragg width in [Q_1,Q_2,E] (FWHM):',
                       ' dQ_1={0:.3f} dQ_2={1:.3f} [A-1] dE={2:.3f} [meV]'.format(bragg_widths[0], bragg_widths[1],
                                                                                  bragg_widths[4]),
                       ' dQ_z={0:.3f} Vanadium width V={1:.3f} [meV]'.format(*bragg_widths[2:4]),
                       'Instrument parameters:',
                       ' DM  =  {0:.3f} ETAM= {1:.3f} SM={2}'.format(self.mono.d, self.mono.mosaic, self.mono.dir),
                       ' KFIX=  {0:.3f} FX  = {1} SS={2}'.format(Energy(energy=self.efixed).wavevector, FX,
                                                                 self.sample.dir),
                       ' DA  =  {0:.3f} ETAA= {1:.3f} SA={2}'.format(self.ana.d, self.ana.mosaic, self.ana.dir),
                       ' A1= {0:.2f} A2={1:.2f} A3={2:.2f} A4={3:.2f} A5={4:.2f} A6={5:.2f} [deg]'.format(*angles),
                       'Collimation [arcmin]:',
                       ' Horizontal: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.hcol),
                       ' Vertical: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.vcol),
                       'Sample:',
                       ' a, b, c  =  [{0}, {1}, {2}] [Angs]'.format(self.sample.a, self.sample.b, self.sample.c),
                       ' Alpha, Beta, Gamma  =  [{0}, {1}, {2}] [deg]'.format(self.sample.alpha, self.sample.beta,
                                                                              self.sample.gamma),
                       ' U  =  {0} [rlu]\tV  =  {1} [rlu]'.format(self.orient1, self.orient2)]

        return '\n'.join(text_format)


class PlotTofInstrument(object):
    def __init__(self):
        pass
