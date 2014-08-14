Example: Resolution calculation with Instrument Class
=====================================================

*Note: This module is still a work-in-progress and the usage of these classes and/or functions will likely change in the near future.*

The following are examples on the usage of the ``neutronpy.resolution.Instrument()`` class, used to define a triple-axis spectrometer instrument configuration and calculate the resolution in reciprocal lattice units for a given sample at a given wave-vector :math:`q`. This tutorial will cover the utilization of both Cooper-Nathans and Popovici calculation methods.

Instrument Configuration
------------------------
First, we will begin by defining an generic triple-axis spectrometer instrument configuration that we will use for this example.

.. plot:: examples/plots/neutronpy_instrument.py
    :height: 200px
    :width: 800px

For the Cooper-Nathans calculation only a rudamentary set of information is required to estimate the resolution at a given :math:`q`. Namely, the fixed energy (incident or final), the relevant collimations (Col\ :math:`_n`), and the monochromator and analyzer crystal types (or :math:`\tau`, if the crystal type is not included in this software).

The rest of the necessary information are all dependent on the sample configuration.

Sample Configuration
--------------------
Defining the sample using the ``neutronpy.resolution.Sample`` class is simple. In this example we define Fe\ :sub:`1.1`\ Te, a high-temperature tetragonal sample, with a sample mosiac of :math:`1.1^{\circ}` or 70\'.

.. code-block:: python

    from neutronpy.resolution import Sample

    FeTe = Sample(3.81, 3.81, 6.25, 90, 90, 90, 70)
    FeTe.u = [1, 0, 0]
    FeTe.v = [0, 1, 0]

where the inputs for ``Sample`` are ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``, and ``mosaic``, respectively, and ``u`` and ``v`` are the orientation vectors in reciprocal lattice units. In this case the sample is oriented in the (*h*, *k*, 0)-plane

Initializing the Instrument
---------------------------
Once the sample is defined and information about the instrument collected we can formally define the instrument using ``neutronpy.resolution.Instrument``.

.. code-block:: python

    from neutronpy.resolution import Instrument

    efixed = 5.
    samp_abc = [3.81, 3.81, 6.25]
    samp_abg = [90., 90., 90.]
    samp_mosaic = 70.
    orient1 = [1, 0, 0]
    orient2 = [0, 1, 0]
    hcol = [32, 80, 120, 120]
    ana_tau = 'PG(002)'
    mono_tau = 'PG(002)'


    EXP = Instrument(efixed, samp_abc, samp_abg, samp_mosaic,
                     orient1, orient2, hcol=hcol, ana_tau=ana_tau,
                     mono_tau=mono_tau)

    EXP.calc_resolution([1, 1, 0, 0])
