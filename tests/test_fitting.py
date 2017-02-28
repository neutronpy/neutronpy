# -*- coding: utf-8 -*-
r"""Tests least-squares fitting

"""
import pytest
import numpy as np
from neutronpy import Fitter
from neutronpy import functions
import neutronpy.fileio as npyio
from mock import patch
import os


def residuals(params, data):
    """Residuals function
    """
    funct, x, y, err = data
    return (y - funct(params, x)) ** 2 / err ** 2


def residuals2(params, data):
    """Residuals function
    """
    funct, x, y, err = data
    return (y - funct(params, x))/ err

def test_gauss_fit_of_gauss_rand():
    """Test gaussian fit
    """
    p = np.array([0., 0., 3., 0., 0.3])
    x = np.linspace(-1, 1, 201)
    np.random.seed(0)
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
    np.random.seed(0)
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
    np.random.seed(0)
    y = functions.voigt(p, x) + np.random.normal(loc=0., scale=0.1, size=201)
    err = np.sqrt(np.abs(y))

    fitobj = Fitter(residuals=residuals, data=(functions.voigt, x, y, err))
    fitobj.parinfo = [{'fixed': fix} for fix in np.asarray([0, 0, 0, 0, 0, 0]).astype(bool)]
    fitobj.fit(params0=p)

    assert ((fitobj.chi2_min < 5.))

@patch('matplotlib.pyplot.show')
def test_fit_plot(mock_show):
    """Test that plot method works
    """
    t_scan=npyio.load_data(os.path.join(os.path.dirname(__file__),'filetypes/HB1A/HB1A_exp0718_scan0222.dat'))
    x=t_scan.data['l']
    y=t_scan.data['detector']
    err=np.sqrt(y)
    params_in=[0.,0.,18.,0.0,0.01,400,0.22,0.01,8,0.25,0.01]
    fitobj=Fitter(residuals=residuals2, data=(functions.gaussian,x,y,err))
    fitobj.fit(params0=params_in)
    fitobj.plot(functions.gaussian)
    fitobj.plot(functions.gaussian,plot_residuals=False)
    fitobj.plot(functions.gaussian,function_str="$p_0+p_1x+2\sqrt{\frac{\ln{2}}{\pi}}\sum_{i=0}^{1} \frac{p_{3i+2}}{p_{3i+4}}e^{-\frac{\ln{2}(x-p_{3i+3})^2}{\pi p_{3i+4}^2}}$")




if __name__ == '__main__':
    pytest.main()
