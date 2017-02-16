# -*- coding: utf-8 -*-
r""" testing of a collection of scans

"""

import neutronpy.data as npysc
import neutronpy.fileio as npyio
import os
import pytest
import collections
from mock import patch




def load_scans(start,stop):
    """
    Creates a dictionary of scans for use in other test functions
    """
    scansin=collections.OrderedDict()
    try:
      for idx in range(start,stop+1):
         scansin[idx]=npyio.load_data(os.path.join(os.path.dirname(__file__),'filetypes/HB1A/HB1A_exp0718_scan0%d.dat'%idx))
      return scansin
    except:
      pytest.failed('scan load failed in scan collection test')


def test_scans_init():
  #load file
  scansin=load_scans(222,243)
  # check expecteed execution
  try:
      tst_obj=npysc.Scans(scans_dict=scansin)
  except:
      pytest.failed('could not build scan collection')
  # check if appropriate error is raise if not an ordered dict.
  dict_scans=dict(scansin)
  with pytest.raises(RuntimeError) as excinfo:
      npysc.Scans(scans_dict=dict_scans)
  assert 'type OrderedDict' in str(excinfo.value)
  # check that update raises the proper error
  with pytest.raises(RuntimeError) as excinfo:
      tst_obj.update(dict_scans)
  assert 'type OrderedDict' in str(excinfo.value)


@patch('matplotlib.pyplot.show')
def test_pcolor(mock_show):
    """
    test pcolor plotting
    """
    scansin=load_scans(222,243)
    s_obj=npysc.Scans(scans_dict=scansin)
    s_obj.pcolor(x='l',y='coldtip')

@patch('matplotlib.pyplot.show')

def test_waterfall(mock_show):
    """
    test waterfall plotting
    """
    scansin=load_scans(222,243)
    s_obj=npysc.Scans(scans_dict=scansin)
    s_obj.waterfall(x='l',label_column='coldtip',offset=5000)
