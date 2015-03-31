NeutronPy Roadmap
=================
This document outlines the future of the NeutronPy package, including planned milestones and features. Detail decreases the further in the future we go. The roadmap will be continually updated.

Milestones
----------
There is currently one milestone on Github:

* `0.3 ResLib <https://github.com/neutronpy/neutronpy/milestones/0.3%20ResLib>`_ - contains issues which are ready to be worked on

The Reslib milestone will focus on making the Instrument class more pythonic, and implementing additional ResLib features, such as Monte Carlo convolution with the resolution fuctions.

Next Release
------------
In the next release there are several planned features:

* Complete translation of ResLib library features
    * Complete Documentation
    * Support both *Q* and *S* coordinate systems
    * Metric tensor calculation
    * Convolution with resolution function
    * Least-Squares fitting of convoluted cross sections
    * Plotting of resolution ellipses (support 3D plots)
    * Potential spurion detector
    * Instrument visualization (w/ shaft angles)
* Expanded support for input file types
    * NeXus
    * SPE
* Add support for function input (in addition to residuals)

Future Features
---------------
This section is for planned features that may require significant development time.

* 3D visualization of volumes (using OpenGL)
    * Support for slicing and cutting
