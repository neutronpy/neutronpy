import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from neutronpy.resolution import Instrument, Sample
from neutronpy.functions import resolution

FeTe = Sample(3.81, 3.81, 6.25, 90, 90, 90, 70)
FeTe.u = [1, 0, 0]
FeTe.v = [0, 1, 0]

EXP = Instrument(5., FeTe, [32, 80, 120, 120], ana='PG(002)', mono='PG(002)', infin=1)
EXP.arms = [1560, 600, 260, 300]
EXP.method = 1

hkle = [1., 1., 0., 0.]
EXP.calc_resolution(hkle)

x, y = np.meshgrid(np.linspace(hkle[0] - 0.05, hkle[0] + 0.05, 501), np.linspace(hkle[1] - 0.05, hkle[1] + 0.05, 501), sparse=True)

R0, RMxx, RMyy, RMxy = EXP.get_resolution_params(hkle, 'QxQy', mode='slice')
p = np.array([0., 0., 1., hkle[0], hkle[1], R0, RMxx, RMyy, RMxy])
z = resolution(p, (x, y))

fig = plt.figure(facecolor='w', edgecolor='k')

plt.pcolormesh(x, y, z, cmap=cm.jet)  # @UndefinedVariable

[x1, y1] = EXP.projections['QxQy'][:, :, 0]
plt.fill(x1, y1, 'r', alpha=0.25)
[x1, y1] = EXP.projections['QxQySlice'][:, :, 0]
plt.plot(x1, y1, 'w--')

plt.xlim(hkle[0] - 0.05, hkle[0] + 0.05)
plt.ylim(hkle[1] - 0.05, hkle[1] + 0.05)

plt.show()
