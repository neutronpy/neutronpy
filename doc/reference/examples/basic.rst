Basic introduction to using NeutronPy
=====================================

In this tutorial the basic features and layout of NeutronPy are described, covering primarily their most typical usage. The more in-depth tutorials are linked within their respective sections. Installation is addressed `on the main page <http://neutronpy.github.com>`_ and will not be covered here. I am assuming the use of a python shell, like ``ipython`` or or ``idle``.

Importing and using NeutronPy
-----------------------------
My recommendation, and personal use mode, is to ``import neutronpy`` under the namespace ``npy``, `i.e.`

>>> import neutronpy as npy

The modules and essential classes can then be accessed as follows

* ``npy.Data`` - :py:class:`.Data` class for handling neutron data
* ``npy.Energy`` - :py:class:`.Energy` class for converting neutron energy to various units and vise versa
* ``npy.Fitter`` - :py:class:`.Fitter` class for performing least-squares fits
* ``npy.Instrument`` - :py:class:`.Instrument` class for calculating Triple-Axis resolution
* ``npy.Lattice`` - :py:class:`.Lattice` class for performing geometry calculations
* ``npy.Material`` - :py:class:`.Material` class for performing structure factor calculations
* ``npy.Sample`` - :py:class:`.Sample` class for use in the :py:class:`.Instrument` class
* ``npy.constants`` - :py:mod:`.constants` module containing the constants used in NeutronPy
* ``npy.functions`` - :py:mod:`.functions` module containing various commonly used functions
* ``npy.gui`` - :py:mod:`.gui` module containing the GUI for resolution calculations :py:func:`.gui.launch`
* ``npy.io`` - :py:mod:`.io` module containing file handling functions
* ``npy.models`` - :py:mod:`.models` module containing commonly used dispersions
* ``npy.spurion`` - :py:mod:`.spurion` module containing spurion calculator for common spurions
* ``npy.symmetry`` - :py:mod:`.symmetry` module containing the :py:class:`.SpaceGroup` class

Other modules, classes, etc. are not imported into the ``npy`` namespace by default, since they are used only internally, but can be accessed `e.g.`

>>> from neutronpy import data

Handling Data
-------------
To load a known filetype, like ``SPICE``, ``ICE`` or ``ICP``

>>> data = npy.io.load_data('path_to_file')

Multiple files can be loaded simultaneously (data will be combined into a single object). Otherwise, if the filetype is unknown data must be loaded separately, e.g. if you have an ``ascii`` format file

>>> import numpy as np
>>> h, k, l, e, counts, monitor, temperature = np.loadtxt('path_to_file', unpack=True)

and a ``Data`` object can be then created by

>>> data = npy.Data(h=h, k=k, l=l, e=e, detector=counts, monitor=monitor, temp=temperature)

Data objects can be added together, quickly plotted, and basic analysis can be performed. For more details see :doc:`data` and :py:class:`.Data`.

Converting Neutron Energy
-------------------------
To easily convert neutron energy in e.g. meV to wavelength in Angstrom

>>> wavelength = npy.Energy(energy=10).wavelength

To print all of the available conversions for one value

>>> print(npy.Energy(energy=10).values)

For complete documentation see :py:class:`.Energy`.

Fitting Data
------------
To perform least-square fitting you must construct a fitting function, and a residuals function, e.g.

.. code-block:: python

    def function(p, x):
        return p[0] + p[1] * x

    def residuals(p, data):
        x, y, err = data
        return (function(p, x) - y) / err

Note that the residuals return is automatically squared. You then construct the ``Fitter`` object, and perform a fit using initial parameters

>>> fit_obj = npy.Fitter(residuals, data=(x, y, err))
>>> fit_obj.fit(params0 = [0, 0])

The fitted parameters can be accessed by the :py:attr:`.Fitter.params` attribute, i.e.

>>> fitted_params = fit_obj.params

More details can be found in :doc:`fitting` and :py:class:`.Fitter`.

Calculating Triple-Axis Resolution
----------------------------------
To calculate the resolution of a Triple-Axis instrument, you must create an instrument and a sample. A basic default instrument and sample and can be constructed by

>>> instr = npy.Instrument()

Resolution ellipses can be calculated for a single or multiple positions in reciprocal space by

>>> q = [1,1,0,0]
>>> instr.plot_projections(q)

More details can be found in :doc:`instrument` and :py:class:`.Instrument` and :py:class:`.Sample`.

Using the Triple-Axis Resolution GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To launch the gui using a python command line

>>> import neutronpy as npy
>>> npy.gui.launch()

Currently, to set new values, press TAB after entering value.

More details are available in :doc:`gui`.


Performing Lattice Geometry Calculations
----------------------------------------
It is possible to calculate several values for a given crystal lattice, defined e.g.

>>> lattice = npy.Lattice(4, 5, 6, 90, 90, 90)

From this ``Lattice`` object the lattice type, volume, and reciprocal values can be found, e.g.

>>> lattice.lattice_type
>>> lattice.volume
>>> lattice.reciprocal_abc

The d-spacing, q and 2theta values of a given reciprocal lattice point can be obtained by

>>> q = [1,0,0]
>>> lattice.get_d_spacing(q)
>>> lattice.get_q(q)
>>> lattice.get_2theta(q)

Finally, you can obtain the angle between two planes defined by their normal vectors by

>>> a = [1,0,0]
>>> b = [0,1,0]
>>> lattice.get_angle_between_planes(a, b)

More details are available in :py:class:`.Lattice`.

Performing Structure Factor Calculations
----------------------------------------
To perform structure factor calculations you must first construct a :py:class:`.Material` object. In this example we build a simple Aluminum crystal. We first define the crystal with a dictionary:

... code-block:: python

    def_Al = {'name': 'Al',
              'composition': [dict(ion='Al', pos=[0, 0, 0])],
              'lattice': dict(abc=[4.0495, 4.0495, 4.0495], abg=[90, 90, 90]),
              'space_group': 'Fm-3m'}

Then we create the ``Material`` object
>>> Al = npy.Material(def_Al)

We can then obtain the nuclear structure factor at a single point by

>>> q = [1,1,1]
>>> Al.calc_nuc_str_fac(q)

For more information see :doc:`material` and :py:class:`.Material`.

Finding Aluminum Spurion Positions
----------------------------------
One can easily find the positions in 2theta of spurious Aluminum peaks which have scattered from the holder or sample environment, given an incident energy by

>>> energy = 14.7
>>> npy.spurion.aluminum(energy)

More details can be seen in :py:func:`.aluminum`.
