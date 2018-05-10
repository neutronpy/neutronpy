import numpy as np

class Guide(object):
    r"""
    """

    def __init__(self, m, length, width, height):
        self.m = m
        self.length = length
        self.width = width
        self.height = height

    @property
    def sigma_l(self):
        return np.sqrt(self.length ** 2 + self.width ** 2 + self.height ** 2) - self.length

    @property
    def sigma_theta(self):
        return np.arcsin(np.sqrt(self.m * np.sin(2.03e-3) ** 2))

    @property
    def sigma_phi(self):
        return self.sigma_theta
