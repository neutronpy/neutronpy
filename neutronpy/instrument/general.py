import numpy as np

from .exceptions import InstrumentError
from .tools import (_CleanArgs, _scalar, calculate_projection_hwhm, ellipse,
                    project_into_plane)


class GeneralInstrument(object):
    r"""Class containing methods general to both Triple Axis and Time of Flight
    instruments.

    Methods
    -------
    calc_projections
    get_resolution_params
    get_resolution

    """

    def calc_projections(self, hkle, npts=36):
        r"""Calculates the resolution ellipses for projections and slices from
        the resolution matrix.

        Parameters
        ----------
        hkle : list
            Positions at which projections should be calculated.

        npts : int, optional
            Number of points in the outputted ellipse curve

        Returns
        -------
        projections : dictionary
            A dictionary containing projections in the planes: QxQy, QxW, and
            QyW, both projections and slices

        """
        R0, NP = self.get_resolution(hkle)

        [H, K, L, W] = _CleanArgs(*hkle)[1:]
        hkle = [H, K, L, W]

        if len(NP.shape) == 2:
            length = 1
        else:
            length = NP.shape[0]

        self.projections = {'QxQy': np.zeros((length, 2, npts)),
                            'QxQySlice': np.zeros((length, 2, npts)),
                            'QxW': np.zeros((length, 2, npts)),
                            'QxWSlice': np.zeros((length, 2, npts)),
                            'QyW': np.zeros((length, 2, npts)),
                            'QyWSlice': np.zeros((length, 2, npts)),
                            'QxQy_fwhm': np.zeros((length, 2)),
                            'QxQySlice_fwhm': np.zeros((length, 2)),
                            'QxW_fwhm': np.zeros((length, 2)),
                            'QxWSlice_fwhm': np.zeros((length, 2)),
                            'QyW_fwhm': np.zeros((length, 2)),
                            'QyWSlice_fwhm': np.zeros((length, 2))}

        A = NP.copy()

        if A.shape == (4, 4):
            A = A.reshape((1, 4, 4))
            R0 = R0[np.newaxis]

        for ind in range(length):
            # Remove the vertical component from the matrix.
            Bmatrix = np.matrix([np.concatenate((A[ind, 0, 0:2], [A[ind, 0, 3]])),
                                 np.concatenate((A[ind, 1, 0:2], [A[ind, 1, 3]])),
                                 np.concatenate((A[ind, 3, 0:2], [A[ind, 3, 3]]))])
            # Projection into Qx, Qy plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(project_into_plane(2, R0[ind], Bmatrix)[-1])

            self.projections['QxQy_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QxQy_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QxQy'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                    [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                             self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                     np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                             self.orient2 / np.linalg.norm(self.orient2) ** 2)],
                                                    npts=npts)

            # Slice through Qx,Qy plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(A[ind, :2, :2])

            self.projections['QxQySlice_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QxQySlice_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QxQySlice'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                         [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                                 self.orient1 /
                                                                 np.linalg.norm(self.orient1) ** 2),
                                                          np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                                 self.orient2 /
                                                                 np.linalg.norm(self.orient2) ** 2)],
                                                               npts=npts)

            # Projection into Qx, W plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(project_into_plane(1, R0, Bmatrix)[-1])

            self.projections['QxW_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QxW_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QxW'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                   [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                           self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                    self.W[ind]],
                                                   npts=npts)

            # Slice through Qx,W plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(
                np.array([[A[ind, 0, 0], A[ind, 0, 3]], [A[ind, 3, 0], A[ind, 3, 3]]]))

            self.projections['QxWSlice_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QxWSlice_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QxWSlice'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                        [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                                self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                         self.W[ind]],
                                                        npts=npts)

            # Projections into Qy, W plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(project_into_plane(0, R0, Bmatrix)[-1])

            self.projections['QyW_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QyW_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QyW'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                   [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                           self.orient2 / np.linalg.norm(self.orient2) ** 2),
                                                    self.W[ind]],
                                                   npts=npts)

            # Slice through Qy,W plane
            hwhm_xp, hwhm_yp, theta = calculate_projection_hwhm(
                np.array([[A[ind, 1, 1], A[ind, 1, 3]], [A[ind, 3, 1], A[ind, 3, 3]]]))

            self.projections['QyWSlice_fwhm'][ind, 0] = 2 * hwhm_xp
            self.projections['QyWSlice_fwhm'][ind, 1] = 2 * hwhm_yp

            self.projections['QyWSlice'][ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                        [np.dot([self.H[ind], self.K[ind], self.L[ind]],
                                                                self.orient2 / np.linalg.norm(self.orient2) ** 2),
                                                         self.W[ind]],
                                                        npts=npts)

    def get_resolution_params(self, hkle, plane, mode='project'):
        r"""Returns parameters for the resolution gaussian.

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

        """

        try:
            A = self.RMS
        except:
            self.calc_resolution(hkle)
            A = self.RMS

        ind = np.where((self.H == hkle[0]) & (self.K == hkle[1]) & (self.L == hkle[2]) & (self.W == hkle[3]))
        print(ind)
        if len(ind[0]) == 0:
            raise InstrumentError('Resolution at provided HKLE has not been calculated.')

        ind = ind[0][0]

        if len(A.shape) != 3:
            A = A.reshape(1, 4, 4)
            selfR0 = self.R0[np.newaxis]

        # Remove the vertical component from the matrix
        Bmatrix = np.vstack((np.hstack((A[ind, 0, :2:1], A[ind, 0, 3])),
                             np.hstack((A[ind, 1, :2:1], A[ind, 1, 3])),
                             np.hstack((A[ind, 3, :2:1], A[ind, 3, 3]))))

        if plane == 'QxQy':
            R0 = np.sqrt(2 * np.pi / Bmatrix[2, 2]) * selfR0[ind]
            if mode == 'project':
                # Projection into Qx, Qy plane
                R0, MP = project_into_plane(2, R0, Bmatrix)
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]
            if mode == 'slice':
                # Slice through Qx,Qy plane
                MP = np.array(A[ind, :2:1, :2:1])
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]

        if plane == 'QxW':
            R0 = np.sqrt(2 * np.pi / Bmatrix[1, 1]) * selfR0[ind]
            if mode == 'project':
                # Projection into Qx, W plane
                R0, MP = project_into_plane(1, R0, Bmatrix)
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]
            if mode == 'slice':
                # Slice through Qx,W plane
                MP = np.array([[A[ind, 0, 0], A[ind, 0, 3]], [A[ind, 3, 0], A[ind, 3, 3]]])
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]

        if plane == 'QyW':
            R0 = np.sqrt(2 * np.pi / Bmatrix[0, 0]) * selfR0[ind]
            if mode == 'project':
                # Projections into Qy, W plane
                R0, MP = project_into_plane(0, R0, Bmatrix)
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]
            if mode == 'slice':
                # Slice through Qy,W plane
                MP = np.array([[A[ind, 1, 1], A[ind, 1, 3]], [A[ind, 3, 1], A[ind, 3, 3]]])
                return R0, MP[0, 0], MP[1, 1], MP[0, 1]

    def get_resolution(self, hkle):
        r"""Returns the resolution matrix and r0 correction at a given Q in rlu

        Parameters
        ----------
        hkle

        Returns
        -------

        """
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

        return R0, NP
