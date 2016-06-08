NeutronPy
=========

.. warning::

    New releases may not be backwards compatibile. This software is in a fluid state and undergoes rapid changes. Bug fix and maintanence releases will generally be  backwards compatibile updates. Major number releases (x.0) could potentially break backwards compatibility, but users will be notified in the changelog.

**TravisCI** |mastertravis| :: **Appveyor** |masterappveyor|

**Code Climate** |climate| :: **Codecov** |coverage|

.. |mastertravis| image:: https://travis-ci.org/neutronpy/neutronpy.svg?branch=master
        :target: https://travis-ci.org/neutronpy/neutronpy

.. |masterappveyor| image:: https://ci.appveyor.com/api/projects/status/github/neutronpy/neutronpy?branch=master&svg=true
        :target: https://ci.appveyor.com/project/pseudocubic/neutronpy

.. |coverage| image:: https://codecov.io/gh/neutronpy/neutronpy/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/neutronpy/neutronpy

.. |climate| image:: https://codeclimate.com/github/neutronpy/neutronpy/badges/gpa.svg
        :target: https://codeclimate.com/github/neutronpy/neutronpy
        :alt: Code Climate


NeutronPy is a python library with commonly used tools for neutron scattering measurements, primarily for Triple Axis Spectrometer data, but easily applied to other types of data, including some *reduced* Time of Flight data. Below is a non-exhaustive list of Neutronpy's features:

    * Triple Axis Spectrometer resolution function calculation (Translated from ResLib), including
        * Resolution ellipses
        * Instrument visualization
    * Structure factor calculation, including
        * Structure factors with support for
            * Mass Normalization
            * Debye-Waller factor
            * Unit cell visualization
        * Single-ion magnetic form factor calculation
    * Least-Squares fitting (custom interface for scipy.optimize.leastsq using lmfit features)
    * Basic data operations
        * Binning
        * Normalization (time/monitor)
        * Calculated peak integrated intensity, position, and width
        * Plotting
        * Slicing
    * Loading from common TAS file types, including
        * SPICE
        * ICE
        * ICP
        * MAD


See `Roadmap <https://github.com/neutronpy/neutronpy/wiki/Roadmap>`_ for panned future features.

NeutronPy began to be developed by David M Fobes in the `Neutron Scattering Group <http://neutrons.phy.bnl.gov/>`_, part of the Condensed Matter Physics & Materials Science Department (CMPMSD) at `Brookhaven National Laboratory <http://www.bnl.gov/>`_. It is currently being developed in the `MPA-CMMS <http://www.lanl.gov/org/padste/adeps/materials-physics-applications/condensed-matter-magnet-science/index.php>`_ division of `Los Alamos National Laboratory <http://www.lanl.gov/>`_. Both are `US Department of Energy, Office of Basic Energy Sciences <http://science.energy.gov/bes/>`_ funded laboratories.

NeutronPy is a work-in-progress (see the `Roadmap <https://github.com/neutronpy/neutronpy/wiki/Roadmap>`_ in the wiki for indications of new upcoming features) and as such, still has many bugs, so use at your own risk. See the Disclaimer below. To report bugs or suggest features see the Contributions section below.

Requirements
------------
The following packages are required to install this library:

* ``Python >= 2.6 (incl. Python 3.3-3.5)``
* ``numpy >= 1.8.0``
* ``scipy >= 0.13.0``
* ``lmfit >= 0.9.1`` (``==0.9.1`` for Python 2.6)
* ``h5py`` (optional, file IO)
* ``matplotlib >= 1.3.0`` (optional, plotting)
* ``nose >= 1.3.0`` (optional, tests)

Installation
------------
It is recommended that you use `anaconda <>`_ and ``pip`` to install NeutronPy::

    conda config --add channels mmcauliffe
    conda install python nomkl numpy scipy h5py matplotlib pyqt5
    pip install neutronpy

Note: the added channel is for pyqt5.

The resolution calculator gui can be used after installation by executing ``neutronpy`` from the command-line.

Documentation
-------------
Documentation is available at `neutronpy.github.io <https://neutronpy.github.io/>`_, or can be built using sphinx by navigating to the doc/ folder and executing ``make html``; results will be in the ``doc/_build/`` folder.

To ask questions you may either `request access <http://goo.gl/forms/odTeCYQQEc>`_ to the `Neutronpy Slack Team <http://neutronpy.slack.com>`_, or create a `Github issue <https://github.com/neutronpy/neutronpy/issues/new>`_.

Contributions
-------------
Contributions may be made by submitting a pull-request for review using the fork-and-pull method on GitHub. Feature requests and bug reports can be made using the GitHub issues interface. Please read the `development guide <https://neutronpy.github.io/development.html>`_ for details on how to best contribute.

To discuss development you may `request access <http://goo.gl/forms/odTeCYQQEc>`_ to the `Neutronpy Slack Team <http://neutronpy.slack.com>`_.

Copyright & Licensing
---------------------
Copyright (c) 2014-2016, David M. Fobes, Released under terms in LICENSE.

The source for the Triple Axis Spectrometer resolution calculations was translated in part from the `ResLib <http://www.neutron.ethz.ch/research/resources/reslib>`_ 3.4c (2009) library released under the terms in LICENSE.RESLIB, originally developed by Andrey Zheludev at Brookhaven National Laboratory, Oak Ridge National Laboratory and ETH Zuerich. email: zhelud@ethz.ch.

Disclaimer
----------
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
