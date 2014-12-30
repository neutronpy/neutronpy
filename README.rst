NeutronPy
=========

.. warning::

    v0.x.x releases may not be backwards compatibile. This software is in a fluid state and undergoing rapid changes. The v1.0.0 release will indicate the start of backwards compatibile updates. Major number releases (x.0) may break backwards compatibility, but users will be notified in the changelog.

**master** |master| :: **develop** |develop|

.. |master| image:: https://travis-ci.org/neutronpy/neutronpy.svg?branch=master
      :target: https://travis-ci.org/neutronpy/neutronpy

.. |develop| image:: https://travis-ci.org/neutronpy/neutronpy.svg?branch=develop
      :target: https://travis-ci.org/neutronpy/neutronpy

NeutronPy is a python library with commonly used tools for neutron scattering measurements, primarily for Triple Axis Spectrometer data, but easily applied to other types of data, including Time of Flight.

* Triple Axis Spectrometer resolution function calculation (Translated from ResLib), including:
	* Resolution ellipses
	* Instrument visualization (planned)
* Form factor calculation, including:
	* Structure factors with support for
		* Mass Normalization
		* Debye-Waller factor
		* Unit cell visualization (future)
	* Single-ion magnetic form factor calculation
* Least-Squares fitting (KMPFIT, based on the C-implementation of MPFIT, from the Kapteyn package)
* Basic data operations
   * Binning
   * Monitor Normalization
   * Calculated peak integrated intensity, position, and width
   * Loading from known filetypes (SPICE, ICE, and ICP supported. More planned)
   * Plotting (incomplete)
   * Slicing (future)
* And More...

NeutronPy is developed by David M Fobes in the `Neutron Scattering Group <http://neutrons.phy.bnl.gov/>`_, part of the Condensed Matter Physics & Materials Science Department (CMPMSD) at `Brookhaven National Laboratory <http://www.bnl.gov/>`_, a `US Department of Energy, Office of Basic Energy Sciences <http://science.energy.gov/bes/>`_ funded laboratory.

NeutronPy is a work-in-progress (see the roadmap in the wiki for indications of new upcoming features) and as such, still has many bugs, so use at your own risk. See the Disclaimer below. To report bugs or suggest features see the Contributions section below.

Requirements
------------
The following packages are required to install this library:

* ``Python >= 2.6 (incl. python 3)``
* ``numpy >= 1.8.0``
* ``scipy >= 0.13.0``
* ``Cython >= 0.20``
* ``matplotlib >= 1.3.0``
* ``nose >= 1.3.0`` (optional, tests)

Installation
------------
It is recommended that you use ``pip`` to install NeutronPy::

    pip install neutronpy

Documentation
-------------
Documentation is available at `neutronpy.github.io <https://neutronpy.github.io/>`_, or can be built using sphinx by navigating to the doc/ folder and executing ``make html``; results will be in the ``doc/_build/`` folder.

Contributions
-------------
Contributions may be made by submitting a pull-request for review using the fork-and-pull method on GitHub. Feature requests and bug reports can be made using the GitHub issues interface.

Copyright & Licensing
---------------------
Copyright (c) 2014, David M. Fobes, Released under terms in LICENSE.

KMPFIT and MPFIT are currently used in part from the `Kapteyn <https://www.astro.rug.nl/software/kapteyn/>`_ package and a custom implementation of the `MINPACK-1 <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_ Least Squares Fitting Library in C, released under the terms in LICENSE.KAPTEYN and LICENSE.MPFIT, respectively.

The source for the Triple Axis Spectrometer resolution calculations was translated in part from the `ResLib <http://www.neutron.ethz.ch/research/resources/reslib>`_ 3.4c (2009) library released under the terms in LICENSE.RESLIB, originally developed by Andrey Zheludev at Brookhaven National Laboratory, Oak Ridge National Laboratory and ETH Zuerich. email: zhelud@ethz.ch.

Disclaimer
----------
THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDERS, THEIR THIRD PARTY LICENSORS, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.
