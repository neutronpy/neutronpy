import datetime as dt
import numpy as np
from .tools import get_bragg_widths, fproject
from ..energy import Energy


class PlotInstrument(object):
    r'''Class containing resolution plotting methods

    Methods
    -------
    plot_projections
    plot_ellipsoid
    plot_instrument
    plot_slice
    description_string

    '''
    def plot_projections(self, hkle, npts=36, dpi=100):
        r'''Plots resolution ellipses in the QxQy, QxW, and QyW zones

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer)
            values at which resolution ellipses are desired to be plotted

        npts : int, optional
            Number of points in an individual resolution ellipse. Default: 36

        '''
        try:
            projections = self.projections
        except AttributeError:
            self.calc_projections(hkle, npts)
            projections = self.projections

        import matplotlib.pyplot as plt

        plt.rc('font', **{'family': 'Bitstream Vera Sans', 'serif': 'cm10', 'size': 6})
        plt.rc('lines', markersize=3, linewidth=1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, facecolor='w', edgecolor='k', dpi=dpi)
        fig.subplots_adjust(bottom=0.175, left=0.15, right=0.85, top=0.95, wspace=0.35, hspace=0.25)

        ax1_lims, ax2_lims, ax3_lims = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        for i in range(self.RMS.shape[-1]):
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
        ax4.text(0, 1, self.description_string(), transform=ax4.transAxes, horizontalalignment='left', verticalalignment='top')

        plt.show()

    def plot_ellipsoid(self, hkle, dpi=100):
        r'''Plots the resolution ellipsoid in the $Q_x$, $Q_y$, $W$ zone

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer) values at which resolution ellipsoid are desired to be plotted

        '''
        from vispy import app, scene, visuals
        import sys

        [H, K, L, W] = hkle
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
            NP = NP[np.newaxis].reshape((4, 4, 1))
            R0 = [R0]

        # Create a canvas with a 3D viewport
        canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        view = canvas.central_widget.add_view()

        surface = []

        for ind in range(NP.shape[-1]):
            # for this plot to work, we need to remove row-column 3 of RMS
            A = np.copy(NP)
            RMS = np.delete(np.delete(A, 2, axis=0), 2, axis=1)[:, :, ind]

#             [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()
            qx = [0]  # _scalar([xvec[0], xvec[1], xvec[2]], [self.H[ind], self.K[ind], self.L[ind]], rsample)
            qy = [0]  # _scalar([yvec[0], yvec[1], yvec[2]], [self.H[ind], self.K[ind], self.L[ind]], rsample)
            qw = [0]  # [self.W[ind]]

            # Q vectors on figure axes
#             o1 = np.copy(self.orient1)
#             o2 = np.copy(self.orient2)
#             pr = _scalar([o2[0], o2[1], o2[2]], [yvec[0], yvec[1], yvec[2]], rsample)

#             o2[0] = yvec[0] * pr
#             o2[1] = yvec[1] * pr
#             o2[2] = yvec[2] * pr
#
#             if np.abs(o2[0]) < 1e-5:
#                 o2[0] = 0
#             if np.abs(o2[1]) < 1e-5:
#                 o2[1] = 0
#             if np.abs(o2[2]) < 1e-5:
#                 o2[2] = 0
#
#             if np.abs(o1[0]) < 1e-5:
#                 o1[0] = 0
#             if np.abs(o1[1]) < 1e-5:
#                 o1[1] = 0
#             if np.abs(o1[2]) < 1e-5:
#                 o1[2] = 0

#             frame = '[Q1,Q2,E]'

#             SMAGridPoints = 40
            EllipsoidGridPoints = 100

            def fn(r0, rms, q1, q2, q3, qx0, qy0, qw0):
                ee = rms[0, 0] * (q1 - qx0[0]) ** 2 + rms[1, 1] * (q2 - qy0[0]) ** 2 + rms[2, 2] * (q3 - qw0[0]) ** 2 + \
                   2 * rms[0, 1] * (q1 - qx0[0]) * (q2 - qy0[0]) + \
                   2 * rms[0, 2] * (q1 - qx0[0]) * (q3 - qw0[0]) + \
                   2 * rms[2, 1] * (q3 - qw0[0]) * (q2 - qy0[0])
                return ee

            # plot ellipsoids
            wx = fproject(RMS.reshape((3, 3, 1)), 0)
            wy = fproject(RMS.reshape((3, 3, 1)), 1)
            ww = fproject(RMS.reshape((3, 3, 1)), 2)

            surface = []
            x = np.linspace(-wx[0] * 1.5, wx[0] * 1.5, EllipsoidGridPoints) + qx[0]
            y = np.linspace(-wy[0] * 1.5, wy[0] * 1.5, EllipsoidGridPoints) + qy[0]
            z = np.linspace(-ww[0] * 1.5, ww[0] * 1.5, EllipsoidGridPoints) + qw[0]
            [xg, yg, zg] = np.meshgrid(x, y, z)

            data = fn(R0[ind], RMS, xg, yg, zg, qx, qy, qw)

            # Create isosurface visual
            surface.append(scene.visuals.Isosurface(data, level=2. * np.log(2.), color=(0.5, 0.6, 1, 1), shading='smooth', parent=view.scene))  # @UndefinedVariable

        for surf in surface:
            [nx, ny, nz] = data.shape
            center = scene.transforms.STTransform(translate=(-nx / 2., -ny / 2., -nz / 2.))
            surf.transform = center

        frame = scene.visuals.Cube(size=(EllipsoidGridPoints * 5, EllipsoidGridPoints * 5, EllipsoidGridPoints * 5), color='white', edge_color=(0., 0., 0., 1.), parent=view.scene)  # @UndefinedVariable
        grid = scene.visuals.GridLines(parent=view.scene)  # @UndefinedVariable
        grid.set_gl_state('translucent')

        # Add a 3D axis to keep us oriented
        axis = scene.visuals.XYZAxis(parent=view.scene)  # @UndefinedVariable

        # Use a 3D camera
        # Manual bounds; Mesh visual does not provide bounds yet
        # Note how you can set bounds before assigning the camera to the viewbox
        cam = scene.TurntableCamera()
        cam.azimuth = 135
        cam.elevation = 30
        cam.fov = 60
        cam.distance = 1.2 * EllipsoidGridPoints
        cam.center = (0, 0, 0)
        view.camera = cam

        canvas.show()
        if sys.flags.interactive == 0:
            app.run()

    def plot_instrument(self, hkle):
        '''Plots the instrument configuration using angles for a given position
        in Q and energy transfer

        Parameters
        ----------
        hkle : tup
            A tuple of intergers or arrays of H, K, L, and W (energy transfer)
            values at which the instrument setup should be plotted

        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport

        fig = plt.figure(edgecolor='k', facecolor='w', figsize=plt.figaspect(0.4) * 1.25)
        ax = fig.gca(projection='3d')

        if hasattr(self.guide, 'width'):
            beam_width = self.guide.width
        else:
            beam_width = 1
        if hasattr(self.guide, 'height'):
            beam_height = self.guide.height
        else:
            beam_height = 1

        if hasattr(self.mono, 'width'):
            mono_width = self.mono.width
        else:
            mono_width = 1
        if hasattr(self.mono, 'height'):
            mono_height = self.mono.height
        else:
            mono_height = 1

        if hasattr(self.sample, 'width'):
            sample_width = self.sample.width
        else:
            sample_width = 1
        if hasattr(self.sample, 'height'):
            sample_height = self.sample.height
        else:
            sample_height = 1
        if hasattr(self.sample, 'depth'):
            sample_depth = self.sample.depth
        else:
            sample_depth = 1

        if hasattr(self.ana, 'width'):
            ana_width = self.ana.width
        else:
            ana_width = 1
        if hasattr(self.ana, 'height'):
            ana_height = self.ana.height
        else:
            ana_height = 1

        if hasattr(self.detector, 'width'):
            detector_width = self.detector.width
        else:
            detector_width = 1
        if hasattr(self.detector, 'height'):
            detector_height = self.detector.height
        else:
            detector_height = 1

        if hasattr(self, 'arms'):
            arms = self.arms
        else:
            arms = [10, 10, 10, 10]

        angles, q = self.get_angles_and_Q(hkle)
        distances = arms

        angles = np.deg2rad(angles)
        A1, A2, A3, A4, A5, A6 = -angles
        x, y, direction = 0, 0, 0

        x0, y0 = x, y
        # plot the Source -----------------------------------------------------
        translate = 0
        rotate = 0 * (np.pi / 180)
        direction = direction + rotate
        x = x + translate * np.sin(direction)
        y = y + translate * np.cos(direction)

        # create a square source
        X = np.array([-beam_width / 2, -beam_width / 2, beam_width / 2, beam_width / 2, -beam_width / 2])
        Z = np.array([beam_height / 2, -beam_height / 2, -beam_height / 2, beam_height / 2, beam_height / 2])
        Y = np.zeros(5)
        l = ax.plot(X + x, Y + y, zs=Z, color='b')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Beam/Source', color='b')

        x0 = x
        y0 = y
        # plot the Monochromator ----------------------------------------------
        translate = distances[0]
        rotate = 0
        direction = direction + rotate
        x = x + translate * np.sin(direction)
        y = y + translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square Monochromator
        X = np.array([-mono_width / 2, -mono_width / 2, mono_width / 2, mono_width / 2, -mono_width / 2]) * np.sin(A1)
        Z = np.array([mono_height / 2, -mono_height / 2, -mono_height / 2, mono_height / 2, mono_height / 2])
        Y = X * np.cos(A1)
        l = ax.plot(X + x, Y + y, zs=Z, color='r')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Monochromator', color='r')

        x0 = x
        y0 = y
        # plot the Sample -----------------------------------------------------
        translate = distances[1]
        rotate = A2
        direction = direction + rotate
        x = x + translate * np.sin(direction)
        y = y + translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a rotated square Sample
        X = np.array([-sample_width / 2, -sample_width / 2, sample_width / 2, sample_width / 2, -sample_width / 2]) * np.sin(A3)
        Z = np.array([sample_height / 2, -sample_height / 2, -sample_height / 2, sample_height / 2, sample_height / 2])
        Y = X * np.cos(A3)
        l1 = ax.plot(X + x, Y + y, zs=Z, color='g')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Sample', color='g')
        X = np.array([-sample_depth / 2, -sample_depth / 2, sample_depth / 2, sample_depth / 2, -sample_depth / 2]) * np.sin(A3 + np.pi / 2)
        Z = np.array([sample_height / 2, -sample_height / 2, -sample_height / 2, sample_height / 2, sample_height / 2])
        Y = X * np.cos(A3 + np.pi / 2)
        l2 = ax.plot(X + x, Y + y, zs=Z, color='g')

        x0 = x
        y0 = y
        # plot the Analyzer ---------------------------------------------------
        translate = distances[2]
        rotate = A4
        direction = direction + rotate
        x = x + translate * np.sin(direction)
        y = y + translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square
        X = np.array([-ana_width / 2, -ana_width / 2, ana_width / 2, ana_width / 2, -ana_width / 2]) * np.sin(A5)
        Z = np.array([ana_height / 2, -ana_height / 2, -ana_height / 2, ana_height / 2, ana_height / 2])
        Y = X * np.cos(A5)
        l = ax.plot(X + x, Y + y, zs=Z, color='magenta')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Analyzer', color='magenta')

        x0 = x
        y0 = y
        # plot the Detector ---------------------------------------------------
        translate = distances[3]
        rotate = A6
        direction = direction + rotate
        x = x + translate * np.sin(direction)
        y = y + translate * np.cos(direction)
        l = ax.plot([x, x0], [y, y0], zs=[0, 0], color='cyan', linestyle='--')

        # create a square
        X = np.array([-detector_width / 2, -detector_width / 2, detector_width / 2, detector_width / 2, -detector_width / 2])
        Z = np.array([detector_height / 2, -detector_height / 2, -detector_height / 2, detector_height / 2, detector_height / 2])
        Y = np.zeros(5)
        l = ax.plot(X + x, Y + y, zs=Z, color='k')
        t = ax.text(X[0] + x, Y[0] + y, Z[0], 'Detector', color='k')

        ax.set_zlim3d(getattr(ax, 'get_zlim')()[0], getattr(ax, 'get_zlim')()[1] * 10)
        plt.show()

    def plot_slice(self, axis, qslice, projections, u, v, num=0):
        r'''Class method for plotting individual projections. Plots both
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

        '''
        dQ1, dQ2 = [], []

        if qslice == 'QxQy':
            axis.fill(projections['QxQy'][0, :][:, num], projections['QxQy'][1, :][:, num], zorder=0, alpha=0.5, edgecolor='none')
            axis.plot(projections['QxQySlice'][0, :][:, num], projections['QxQySlice'][1, :][:, num], zorder=1)
            dQ1.append(np.max(projections['QxQy'][0, :][:, num]) - np.min(projections['QxQy'][0, :][:, num]))
            dQ2.append(np.max(projections['QxQy'][1, :][:, num]) - np.min(projections['QxQy'][1, :][:, num]))
            axis.set_xlim(np.min(projections['QxQy'][0, :][:, num]), np.max(projections['QxQy'][0, :][:, num]))
            axis.set_ylim(np.min(projections['QxQy'][1, :][:, num]), np.max(projections['QxQy'][1, :][:, num]))
        elif qslice == 'QxW':
            axis.fill(projections['QxW'][0, :][:, num], projections['QxW'][1, :][:, num], zorder=0, alpha=0.5, edgecolor='none')
            axis.plot(projections['QxWSlice'][0, :][:, num], projections['QxWSlice'][1, :][:, num], zorder=1)
            dQ1.append(np.max(projections['QxW'][0, :][:, num]) - np.min(projections['QxW'][0, :][:, num]))
            dQ2.append(np.max(projections['QxW'][1, :][:, num]) - np.min(projections['QxW'][1, :][:, num]))
            axis.set_xlim(np.min(projections['QxW'][0, :][:, num]), np.max(projections['QxW'][0, :][:, num]))
            axis.set_ylim(np.min(projections['QxW'][1, :][:, num]), np.max(projections['QxW'][1, :][:, num]))
        elif qslice == 'QyW':
            axis.fill(projections['QyW'][0, :][:, num], projections['QyW'][1, :][:, num], zorder=0, alpha=0.5, edgecolor='none')
            axis.plot(projections['QyWSlice'][0, :][:, num], projections['QyWSlice'][1, :][:, num], zorder=1)
            dQ1.append(np.max(projections['QyW'][0, :][:, num]) - np.min(projections['QyW'][0, :][:, num]))
            dQ2.append(np.max(projections['QyW'][1, :][:, num]) - np.min(projections['QyW'][1, :][:, num]))
            axis.set_xlim(np.min(projections['QyW'][0, :][:, num]), np.max(projections['QyW'][0, :][:, num]))
            axis.set_ylim(np.min(projections['QyW'][1, :][:, num]), np.max(projections['QyW'][1, :][:, num]))

        dQ1, dQ2 = [np.max(item) for item in [dQ1, dQ2]]

        axis.set_xlabel(r'$\mathbf{Q}_1$ (along ' + str(u) + r') (r.l.u.)' + r', $\delta Q_1={0:.3f}$'.format(dQ1), fontsize=7)
        if 'W' in qslice:
            axis.set_ylabel(r'$\hbar \omega$ (meV)' + r', $\delta \hbar \omega={0:.3f}$'.format(dQ2), fontsize=7)
        else:
            axis.set_ylabel(r'$\mathbf{Q}_2$ (along ' + str(v) + r') (r.l.u.)' + r', $\delta Q_2={0:.3f}$'.format(dQ2), fontsize=7)

    def description_string(self):
        r'''Generates text string describing most recent resolution calculation

        '''
        try:
            method = ['Cooper-Nathans', 'Popovici'][self.method]
        except AttributeError:
            method = 'Cooper-Nathans'
        frame = '[Q1,Q2,Qz,E]'

        try:
            FX = 2 * int(self.infin == -1) + int(self.infin == 1)
        except AttributeError:
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
                       'Position HKLE [{0}]'.format(dt.datetime.now().strftime('%d-%b-%Y %H:%M:%S')),
                       '',
                       ' [Q_H, Q_K, Q_L, E] = {0} '.format(self.HKLE),
                       '',
                       'Resolution Matrix M in {0} (M/10^4):'.format(frame),
                       '[[{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 0] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 1] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]'.format(*NP[:, 2] / 1.0e4),
                       ' [{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}]]'.format(*NP[:, 3] / 1.0e4),
                       '',
                       'Resolution volume:   V_0={0:.6f} meV/A^3'.format(2 * ResVol),
                       'Intensity prefactor: R_0={0:.3f}'.format(R0),
                       'Bragg width in [Q_1,Q_2,E] (FWHM):',
                       ' dQ_1={0:.3f} dQ_2={1:.3f} [A-1] dE={2:.3f} [meV]'.format(bragg_widths[0], bragg_widths[1], bragg_widths[4]),
                       ' dQ_z={0:.3f} Vanadium width V={1:.3f} [meV]'.format(*bragg_widths[2:4]),
                       'Instrument parameters:',
                       ' DM  =  {0:.3f} ETAM= {1:.3f} SM={2}'.format(self.mono.d, self.mono.mosaic, self.mono.dir),
                       ' KFIX=  {0:.3f} FX  = {1} SS={2}'.format(Energy(energy=self.efixed).wavevector, FX, self.sample.dir),
                       ' DA  =  {0:.3f} ETAA= {1:.3f} SA={2}'.format(self.ana.d, self.ana.mosaic, self.ana.dir),
                       ' A1= {0:.2f} A2={1:.2f} A3={2:.2f} A4={3:.2f} A5={4:.2f} A6={5:.2f} [deg]'.format(*angles),
                       'Collimation [arcmin]:',
                       ' Horizontal: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.hcol),
                       ' Vertical: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.vcol),
                       'Sample:',
                       ' a, b, c  =  [{0}, {1}, {2}] [Angs]'.format(self.sample.a, self.sample.b, self.sample.c),
                       ' Alpha, Beta, Gamma  =  [{0}, {1}, {2}] [deg]'.format(self.sample.alpha, self.sample.beta, self.sample.gamma),
                       ' U  =  {0} [rlu]\tV  =  {1} [rlu]'.format(self.orient1, self.orient2)]

        return '\n'.join(text_format)