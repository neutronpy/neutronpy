'''
Created on May 19, 2014

@author: davidfobes
'''
import os
import json
from scipy import constants

with open(os.path.join(os.path.dirname(__file__),
                       "database/mag_ion_j.json"), 'r') as infile:
    magIonJ = json.load(infile)

with open(os.path.join(os.path.dirname(__file__),
                       "database/periodic_table.json"), 'r') as infile:
    periodicTable = json.load(infile)

with open(os.path.join(os.path.dirname(__file__),
                       "database/scat_len.json"), 'r') as infile:
    scatLen = json.load(infile)

joules2meV = 1. / constants.physical_constants['electron volt-joule relationship'][0] * 1.e3  # Joules to meV @IgnorePep8
