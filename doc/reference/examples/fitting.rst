Least-squares fitting with the Fitter Class
===========================================

*Note: This module is still a work-in-progress and the usage of these classes and/or functions may change in the future.*

The follow is an example of how to use the :py:class:`.Fitter` class, which is a non-linear least-squares minimization routine based on the `C implementation of MPFIT <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_. :py:class:`.Fitter` in its current form is identical to the Fitter class in the `kmpfit <https://www.astro.rug.nl/software/kapteyn/kmpfit.html>`_ module of the `Kapteyn Package <https://www.astro.rug.nl/software/kapteyn/index.html>`_.

Defining the problem
--------------------
In its current state, :py:class:`.Fitter` requires you to pass a ``residuals`` function, rather than the function to which you desire to fit your data. On one hand, this gives more flexibility because the form of the residuals can be defined by the user, rather than being confined to the standard formula for residuals:

.. math::   S = \sum_{i}\frac{y_i - f(x_i)}{\sigma_i}.

Note: In the future there will be a change that will allow the user to pass a function and necessary data, without having to write a residuals function, but the current implementation will be preserved, *i.e.* a residual function will always be able to be defined if the user desires.

Assuming you already have your ``function`` prepared, defining the residuals function is trivial. For the sake of this example we will assume that ``function`` is the one dimensional :py:func:`.functions.gaussian`:

.. code-block:: python

    from neutronpy.functions import gaussian
    def residuals(params, data):
        x, y, err = data
        return (y - gaussian(params, x)) / err

Notice that I did not square the residuals function. This is because :py:class:`.Fitter` will square the residuals as part of the *Least-squares* fitting.

Initializing
------------
Once the problem is defined, we can initialize our object using :py:class:`.Fitter`:

>>> from neutronpy import Fitter
>>> fitobj = Fitter(residuals=residuals, data=(x, y, err))

Initial parameters and Constraints
--------------------------
The parameters and constraints will obviously depend on the specific problem. Initial parameters can be defined as a list; in this example we will define a single gaussian peak with no background:

>>> params = [0, 0, 1, 0, 0.1]

Defining constraints is slightly more complicated however. It is possible to define whether the parameter is fixed, its limits, its step size, and its sidedness, for each parameter. Like the parameters, constraints, *i.e.* ``parinfo`` is defined as a list, but with either dictionaries or ``None`` for each parameter. For example, let us fix the background terms:

>>> fitobj.parinfo = [{'fixed': True}, {'fixed': 1}, {'fixed': 0}, {'fixed': False}, None]

The above illustrates equivalent ways to either fix (the first two list members), or leave unfixed (the last three list members) the parameters.

Fitting
-------
To fit the function we simply call :py:meth:`.Fitter.fit` with our initial parameters:

>>> fitobj.fit(params0=params)

Results
-------
Results of the fit are stored in ``fitobj.params``. :math:`\chi^2` is stored in ``fitobj.chi2_min``, and the parameter errors are stored in ``fitobj.xerror``.
