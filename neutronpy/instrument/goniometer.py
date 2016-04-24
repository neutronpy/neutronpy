# -*- coding: utf-8 -*-
r'''Define Triple Axis goniometer

'''
import numpy as np


class Goniometer(object):
    r'''Defines a goniometer

    '''

    def __init__(self, u, theta_u, v, theta_v, sgu, sgl, omega=0):
        self.u = u
        self.theta_u = theta_u

        self.v = v
        self.theta_v = theta_v

        self.sgu = sgu
        self.sgl = sgl

        self.omega = 0

    @property
    def omega_rad(self):
        return self.omega_rad

    @property
    def sgu_rad(self):
        return np.deg2rad(self.sgu)

    @property
    def sgl_rad(self):
        return np.deg2rad(self.sgl)

    @property
    def theta_rad(self):
        return np.arctan((self.ki - self.kf * np.cos(self.phi)) / (self.kf * np.sin(self.phi)))

    @property
    def theta(self):
        pass

    @property
    def N(self):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(self.sgu_rad), -np.sin(self.sgu_rad)],
                          [0, np.sin(self.sgu_rad), np.cos(self.sgu_rad)]])

    @property
    def M(self):
        return np.matrix([[np.cos(self.sgl_rad), 0, np.sin(self.sgl_rad)],
                          [0, 1, 0],
                          [-np.sin(self.sgl_rad), 0, np.cos(self.sgl_rad)]])

    @property
    def Omega(self):
        return np.matrix([[np.cos(self.omega_rad), -np.sin(self.omega_rad), 0],
                          [np.sin(self.omega_rad), np.cos(self.omega_rad), 0],
                          [0, 0, 1]])

    @property
    def Theta(self):
        return np.matrix([[np.cos(self.theta_rad), -np.sin(self.theta_rad), 0],
                          [np.sin(self.theta_rad), np.cos(self.theta_rad), 0],
                          [0, 0, 1]])

    @property
    def T_c(self):
        return np.matrix([self.u, self.v, np.cross(self.u, self.v)]).T

    @property
    def T_phi(self):
        return np.matrix([self.u_phi(np.deg2rad(self.theta_u), self.sgl_rad, self.sgu_rad),
                          self.u_phi(np.deg2rad(self.theta_v), self.sgl_rad, self.sgu_rad),
                          self.u_phi(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))]).T

    @property
    def R(self):
        return self.Omega * self.M * self.N

    @property
    def U(self):
        r'''Defines an orientation matrix based on supplied goniometer angles

        '''
        return self.T_phi * np.linalg.inv(self.T_c)

    def u_phi(self, omega, chi, phi):
        return [np.cos(omega) * np.cos(chi) * np.cos(phi) - np.sin(omega) * np.sin(phi),
                np.cos(omega) * np.cos(chi) * np.sin(phi) + np.sin(omega) * np.cos(phi),
                np.cos(omega) * np.sin(chi)]
