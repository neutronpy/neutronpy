=========
Changelog
=========

* :release:`1.1.1 <2018-06-25>` 77, 74, 111
* :feature:`77` Added Violini resolution calculation to :py:class:`.Instrument`
* :feature:`111` Added calculation of UB matrix to :py:class:`.Sample` via :py:meth:`.Sample.UBmatrix`
* :feature:`111` Added custom exceptions to be more descriptive
* :support:`111` Refactored :py:class:`.Instrument` to support creation of ToF and TAS type instruments
* :support:`111` Added example jupyter notebooks to documentation
* :support:`111` Added factors of 2pi in :py:class`.Lattice`
* :support:`74` Rewrote method for generating 3D plot of resolution ellipsoid

* :release:`1.0.4 <2017-05-18>` 102, 104, 105, 106, 107, 108, 109, 110
* :support:`110` Added jupyter notebook examples to documentation
* :bug:`109` Fixed Anaconda and PYPI automatic deployment
* :bug:`108` Fixed continuous integration builds
* :feature:`107` Exposed figure handles in :py:class:`.Scans`
* :bug:`106` Fixed typo in calculation of Magnetic Form Factor in :py:class:`.MagneticFormFactor`
* :feature:`105` Added methods to calculate statistics of data in :py:class:`.Scans`
* :support:`104` Updated documentation for :py:class:`.Data` and :py:class:`.Material`
* :bug:`102` Fixes problem with building of Q in :py:class:`.Data` when some data not available

* :release:`1.0.3 <2017-02-17>` 93, 94, 95, 96, 98, 99, 100, 101, 103
* :support:`103` Updated examples and docstring for :py:class:`.Material` to reflect proper usage of ``massNorm`` key
* :bug:`101` Fixed ``KeyError`` when using :py:class:`.Data` attributes where one or more of the data columns is not defined
* :support:`100` Updated examples documentation for the :py:class:`.Data`
* :bug:`99` Changed input requirements of :py:class:`.Scans` to be an ``collections.OrderedDict``
* :bug:`98` Fixed bug concerning loading files with unicode characters in the file name
* :feature:`96` Added ``__repr__`` methods to :py:class:`.Data`, :py:class:`.Material`, :py:class:`.Atom`, :py:class:`.Lattice`, etc.
* :support:`95` Refactor setup.py and automated package building by CI
* :support:`94` Removed default matplotlib formatting, also fixes a import problem
* :feature:`93` Added Ce3+ magnetic form factor

* :release:`1.0.2 <2016-12-15>` 80, 81, 82, 83, 84, 86, 89, 90, 91, 92
* :feature:`92` Updated documentation and made bug fixes to :py:class:`.Fitter` and :py:meth:`.load_data`
* :feature:`91` Added :py:class:`.Scans` class to build collections of Data objects for easier plotting
* :feature:`90` Added automatic conda builds for tagged releases
* :feature:`89` Depreciated support for py26 and py33, and changed to py.test from nose. Added ``nan_policy`` kwarg to :py:class:`.Fitter`
* :feature:`86` Enhanced :py:meth:`.Data.integrate`, :py:meth:`.Data.position`, :py:meth:`.Data.width`
* :bug:`84` Fixed typo in ``resolution.ui`` to select ``'Co0.92Fe0.08(200)'`` as monochromator/analyzer in :py:mod:`neutronpy.gui`
* :feature:`80` Rewrite of fitting routines into pure python implementation using LMFIT package
* :bug:`81` Fixed error in :py:func:`.functions.gaussian2d`
* :feature:`82` Added ability to compare between :py:class:`.Data` and :py:class:`.Instrument` objects
* :support:`83` Added developer guide and changed to semantic version numbering

* :release:`1.0.0 <2016-05-10>`
* :feature:`68` Background subtraction added to :py:class:`.Data`
* :feature:`73` Added support for saving :py:class:`.Data` and :py:class:`.Instrument` objects to disk
* :bug:`72 major` Fixed problem with ``io`` namespace by renaming it to ``fileio``

* :release:`1.0.0-b3 <2016-04-24>`
* :feature:`67` All data is now imported from files when using :py:func:`.load_data`. Binning is now more generalized.
* :feature:`71` Added Currat-Axe spurion method :py:meth:`.spurion.currat_axe_peaks`

* :release:`1.0.0-b1 <2016-04-07>`
* :feature:`62` Major refactoring, breaks backwards compatibility. See documentation for new usage.
* :feature:`64` Added basic physical models to :py:mod:`.models`
* :feature:`65` Added :py:class:`.SpaceGroup` to generate all symmetry operations given space group symbol, and added :py:class:`.SpaceGroup` to :py:class:`.Material` to symmetrize crystal structures
* :feature:`63` Added :py:mod:`.gui` for resolution calculations, invoked by :py:meth:`.gui.launch`
* :bug:`61 major` Fixed error handling in :py:meth:`.Instrument.calc_resolution_in_Q_coords` for scattering triangle not closing
* :support:`60` Added documentation to :py:class:`.Monochromator` and :py:class:`.Analyzer` concerning focusing
* :bug:`58 major` Fixed error in :py:meth:`.Instrument.resolution_convolution_SMA` and :py:meth:`.Instrument.resolution_convolution` giving incorrect lineshapes
* :bug:`57 major` Fixed ``R0`` prefactors calculated by :py:meth:`.Instrument.calc_resolution` to be consistent with ResLib
* :bug:`56 major` Fixed handling of ``ACCURACY`` input argument in :py:meth:`.Instrument.resolution_convolution` and :py:meth:`.Instrument.resolution_convolution_SMA`
* :bug:`55 major` Fixed call of prefactor function ``pref`` in :py:meth:`.Instrument.resolution_convolution` and :py:meth:`.Instrument.resolution_convolution_SMA` to include ``W``
* :bug:`54 major` Fixed documentation to reflect correct usage of ``mono.dir``, ``ana.dir`` and ``sample.dir`` to define handedness of spectrometer
* :bug:`53 major` Added ``xlabel`` and ``ylabel`` to data plotting method :py:meth:`.Data.plot`

* :release:`0.3.5 <2016-02-26>` 48, 49, 50, 51, 52
* :support:`52` Updated License from BSD 3-Clause to MIT License
* :bug:`51` Fixed default behavior of :py:attr:`.Instrument.moncor` variable in :py:class:`.Instrument` to coincide with documentation
* :bug:`50` Explicitly defined vertical mosaic ``vmosaic`` in :py:class:`.Sample`
* :bug:`49` Fixed incorrect usage of ``strftime`` in :py:meth:`.Instrument.plot_projections`
* :bug:`48` Fixed error when ``u`` and ``v`` were defined at ``list`` types instead of ``ndarray`` in :py:class:`.Sample`

* :release:`0.3.4 <2016-01-21>` 40, 41, 42, 43, 44, 45, 46, 47
* :support:`40` Added unittests for all libraries to increase code coverage
* :support:`41` Added documentation for spurion library, corrected docs for resolution and core libraries
* :bug:`42` Fixed variable name ``moncar`` to correct name ``moncor`` in :py:class:`.Instrument`
* :bug:`43` Fixed :py:func:`.GetTau` handling of ``getlabel`` option
* :bug:`44` Fixed calculation of :py:meth:`.Data.scattering_function` to use detector counts as expected
* :bug:`45` Fixed :py:func:`.save` to form output array correctly, removed ``'nexus'`` and ``'binary'`` as output formats, and added ``'hdf5'`` and ``'pickle'`` as output formats
* :bug:`46` Fixed behavior of division operations on :py:class:`.Data` objects for Python 3
* :bug:`47` Resolved errors and warnings generated by sphinx autodoc, documentation now should build without error

* :release:`0.3.3 <2016-01-15>` 37, 38, 39
* :support:`37` Updated example documentation for :py:class:`.Material`
* :bug:`38` Fixed problem with Sample Shape matrix being the wrong shape upon initialization generating error in :py:meth:`.Instrument.calc_resolution_in_Q_coords`
* :support:`39` Updated TravisCI test environments to include Python 3.5 and latest versions of numpy and scipy

* :release:`0.3.2 <2015-09-02>` 24, 33, 34
* :feature:`34` Added method :py:meth:`.Instrument.plot_ellipsoid`
* :bug:`33` Fixed error in :py:meth:`.Instrument.calc_projections` where only giving one point would generate an error
* :feature:`24` Added plotting of Instrument setup diagram with :py:meth`.Instrument.plot_instrument`

* :release:`0.3.1 <2015-08-14>` 27, 29, 30, 31, 32
* :feature:`32` Added a Aluminum spurion calculator
* :feature:`31` Beginnings of a :py:class:`.Goniometer` class added for future ability for crystal alignment
* :bug:`30` Added ability to specify seed for Monte Carlo technique in :py:meth:`.Instrument.resolution_convolution`
* :bug:`29` Fixed problem with forked processes not closing after completing
* :feature:`27` Added proper error handling to :py:meth:`.Data.bin`

* :release:`0.3.0 <2015-03-31>`
* :feature:`26` Added :py:meth:`.resolution.load` to load experimental setup from files
* :bug:`25 major` Fixed ComplexWarning in the structural form factor calculation which was casting complex values to only real
* :feature:`23` Added :py:meth:`.plot_projections` to give simple plots of resolution ellipses in three different views
* :feature:`22` Convolution algorithm methods added to :py:class:`.Instrument`: :py:meth:`.resolution_convolution` and :py:meth:`.resolution_convolution_SMA`
* :feature:`17` :py:class:`.Instrument` has been refactored to be more self contained and pythonic

* :release:`0.2.0 <2015-03-11>`
* :bug:`20 major` Prefactor now taken into account when loading ICP files
* :bug:`19 major` Files loaded even if some default data headers are not found in file
* :bug:`18 major` Structure factor calculation now can use sparse arrays as generated by meshgrid
* :support:`16` Documentation for Material and Instrument classes updated
* :feature:`15` Data class rewrite. Data is now loaded with :py:meth:`.load` method, not using :py:class:`.Data` class.

* :release:`0.1.3 <2014-12-30>` 18
* :bug:`18` Update :py:meth:`.Material.calc_str_fac` to include better checking of input hkl tuple

* :release:`0.1.2 <2014-09-22>` 11, 12
* :support:`12` Major overhaul of documentation, including new theme based on ReadTheDocs style
* :feature:`11` tools package has been changed to core package and package contents are accessible from root level ``neutronpy.``

* :release:`0.1.1 <2014-09-12>` 5, 6, 7
* :bug:`7` Added tolerances to :py:meth:`.Data.combine_data` so that small differences in Q will be ignored
* :feature:`6` Added time to :py:class:`.Data` for normalization purposes
* :bug:`5` Update :py:meth:`.Data.bin` to use binary search algorithm for speed increase

* :release:`0.1.0 <2014-09-09>`
* :support:`4` Added examples for :py:class:`.Data` to documentation
* :support:`3` Added examples for :py:class:`.Fitter` to documentation
* :feature:`2` Added fitting to :py:meth:`.Data.plot` with ``fit_options`` argument
* :feature:`1` Added error plots to :py:class:`.Data` using :py:meth:`.Data.plot`
