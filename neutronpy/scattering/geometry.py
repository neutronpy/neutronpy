# -*- coding: utf-8 -*-
r"""Tools for visualizing the scattering geometry

"""
import warnings
import numpy as np
from ..energy import Energy
from .tools import get_coords_in_reciprocal_space, get_vector_in_reciprocal_units, vector_gcd


class ScatteringPlane(object):
    r"""Scattering plane definition

    """

    def brillouin_zone(self):
        pass

    @property
    def w(self):
        r"""

        Returns
        -------
        vector normal to the scattering plane

        """
        return vector_gcd(np.cross(self.u, self.v))

    @property
    def reciprocal_u(self):
        r"""

        Returns
        -------

        """
        return get_vector_in_reciprocal_units(self.u, self.abc)

    @property
    def reciprocal_v(self):
        r"""

        Returns
        -------

        """
        return get_vector_in_reciprocal_units(self.v, self.abc)

    @property
    def reciprocal_w(self):
        r"""

        Returns
        -------

        """
        return get_vector_in_reciprocal_units(self.w, self.abc)

    @property
    def reciprocal_transformation(self):
        r"""

        Returns
        -------

        """
        return np.array([np.linalg.norm(vec) for vec in [self.reciprocal_u, self.reciprocal_v, self.reciprocal_w]])

    def get_points_in_scattering_plane(self, zones):
        r"""

        Parameters
        ----------
        zones

        Returns
        -------
        vectors

        """
        vectors = []
        for n in range(-zones, zones + 1):
            for m in range(-zones, zones + 1):
                if n == 0 and m == 0:
                    pass
                else:
                    vectors.append(n * np.array(self.u) + m * np.array(self.v))

        return vectors

    def plot_reciprocal_vectors(self, ax, zones=5):
        r"""

        Parameters
        ----------
        ax
        zones

        Returns
        -------

        """
        reciprocal_vectors = self.get_points_in_scattering_plane(zones)
        reciprocal_points = np.array(
            [get_coords_in_reciprocal_space(vec, self.u, self.v) for vec in reciprocal_vectors]).T

        ax.scatter(reciprocal_points[0], reciprocal_points[1], facecolor='r')

        for vector, coords in zip(reciprocal_vectors, reciprocal_points[:2, :].T):
            vector = str(vector).replace('  ', ',').replace(' ', ',').replace('[', '(').replace(']', ')')
            ax.annotate('{0}'.format(vector), xy=coords, textcoords='data', color='r')

        ax.scatter(0, 0, facecolor='g')
        ax.annotate('(0,0,0)', xy=(0, 0), textcoords='data', color='g')

    def plot_scattering_triangle(self, ax, hkle, eief, efixed='ei'):
        r"""

        Parameters
        ----------
        ax
        hkle
        eief
        efixed

        Returns
        -------

        """
        from matplotlib.pyplot import Circle
        if efixed == 'ei':
            ki = Energy(energy=eief).wavevector
            kf = Energy(energy=eief - hkle[-1]).wavevector
        elif efixed == 'ef':
            kf = Energy(energy=eief).wavevector
            ki = Energy(energy=eief + hkle[-1]).wavevector

        q_vector_Ainv = get_vector_in_reciprocal_units(hkle[:-1], self.abc)
        q_coords_rlu = np.array(get_coords_in_reciprocal_space(q_vector_Ainv, self.reciprocal_u, self.reciprocal_v))
        q_coords_Ainv = q_coords_rlu * self.reciprocal_transformation

        q = self.get_q(hkle[:-1])
        ki_q_angle = np.arccos((ki ** 2 - kf ** 2 + q ** 2) / (2 * ki * q))
        kf_q_angle = np.arccos((ki ** 2 - kf ** 2 - q ** 2) / (2 * kf * q))

        ki_x = np.array([0, np.cos(ki_q_angle) * ki]) / np.array([self.reciprocal_transformation[0]] * 2)
        ki_y = np.array([0, np.sin(ki_q_angle) * ki]) / np.array([self.reciprocal_transformation[1]] * 2)

        kf_x = np.array([q_coords_Ainv[0], q_coords_Ainv[0] - (np.cos(np.pi - kf_q_angle) * kf)]) / np.array(
            [self.reciprocal_transformation[0]] * 2)
        kf_y = np.array([q_coords_Ainv[1], q_coords_Ainv[1] + (np.sin(np.pi - kf_q_angle) * kf)]) / np.array(
            [self.reciprocal_transformation[1]] * 2)

        ax.plot(ki_x, ki_y, color='k')
        ax.plot(kf_x, kf_y, color='k')
        ax.plot([0, q_coords_rlu[0]], [0, q_coords_rlu[1]], color='red')
        ax.scatter(ki_x, ki_y, facecolor='none')

        center = np.array([ki_x[1], ki_y[1], 0])
        radius = np.linalg.norm(center - q_coords_rlu)
        ewald_circle = Circle(center, radius, facecolor='none', edgecolor='b')
        ax.add_artist(ewald_circle)

    def plot(self, hkle, eief, efixed='ei', zones=5):
        r"""

        Parameters
        ----------
        hkle
        eief
        efixed
        zones

        Returns
        -------

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        self.plot_reciprocal_vectors(ax, zones)
        self.plot_scattering_triangle(ax, hkle, eief, efixed)

        plt.axis('equal')
        plt.show()
