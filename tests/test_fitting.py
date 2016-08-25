# -*- coding: utf-8 -*-
r"""Tests least-squares fitting

"""
import pytest
import numpy as np
from neutronpy import Fitter
from neutronpy import functions


def residuals(params, data):
    """Residuals function
    """
    funct, x, y, err = data
    return (y - funct(params, x)) ** 2 / err ** 2


def test_gauss_fit_of_gauss_rand():
    """Test gaussian fit
    """
    p = np.array([0., 0., 3., 0., 0.3])
    x = np.linspace(-1, 1, 201)
    y = functions.gaussian(p, x) + np.random.normal(loc=0., scale=0.05, size=201)
    err = np.sqrt(np.abs(y))

    fitobj = Fitter(residuals=residuals, data=(functions.gaussian, x, y, err))
    fitobj.parinfo = [{'fixed': fix} for fix in np.asarray([0, 0, 0, 0, 0]).astype(bool)]
    fitobj.fit(params0=p)

    assert ((fitobj.chi2_min < 5.))


def test_voigt_fit_of_gauss_rand():
    """Test voigt fit of gaussian
    """
    p = np.array([0., 0., 3., 0., 0.3])
    x = np.linspace(-1, 1, 201)
    y = functions.gaussian(p, x) + np.random.normal(loc=0., scale=0.1, size=201)
    err = np.sqrt(np.abs(y))

    fitobj = Fitter(residuals=residuals, data=(functions.voigt, x, y, err))
    fitobj.parinfo = [{'fixed': fix} for fix in np.asarray([0, 0, 0, 0, 0, 0]).astype(bool)]
    fitobj.fit(params0=np.concatenate((p, np.array([0.2]))))

    assert ((fitobj.chi2_min < 5.))


def test_voigt_fit_of_voigt_rand():
    """Test voigt fit of voigt function
    """
    p = np.array([0., 0., 3., 0., 0.3, 0.2])
    x = np.linspace(-1, 1, 201)
    y = functions.voigt(p, x) + np.random.normal(loc=0., scale=0.1, size=201)
    err = np.sqrt(np.abs(y))

    fitobj = Fitter(residuals=residuals, data=(functions.voigt, x, y, err))
    fitobj.parinfo = [{'fixed': fix} for fix in np.asarray([0, 0, 0, 0, 0, 0]).astype(bool)]
    fitobj.fit(params0=p)

    assert ((fitobj.chi2_min < 5.))


if __name__ == '__main__':
    pytest.main()
