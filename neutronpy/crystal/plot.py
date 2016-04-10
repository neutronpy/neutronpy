import numpy as np


class PlotMaterial(object):
    r'''Class containing plotting methods for Material object

    Methods
    -------
    plot_unit_cell

    '''
    def plot_unit_cell(self):
        r'''Plots the unit cell and atoms of the material.

        '''

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport
        from itertools import product, combinations

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # draw unit cell
        for s, e in combinations(np.array(list(product([0, self.abc[0]], [0, self.abc[1]], [0, self.abc[2]]))), 2):
            if np.sum(np.abs(s - e)) in self.abc:
                ax.plot3D(*zip(s, e), color="b")

        # plot atoms
        x, y, z, m = [], [], [], []
        for item in self.atoms:
            x.append(item.pos[0] * self.abc[0])
            y.append(item.pos[1] * self.abc[1])
            z.append(item.pos[2] * self.abc[2])
            m.append(item.mass)

        ax.scatter(x, y, z, s=m)

        plt.axis('scaled')
        plt.axis('off')

        plt.show()
