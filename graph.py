import numpy as np
from matplotlib import pyplot as plt
import csv
import argparse
import pickle
from itertools import product
import json
import os

"""
'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid',
'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',
'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette',
'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper',
'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
'seaborn-whitegrid', 'tableau-colorblind10'
"""

# plt.style.use('ggplot')
colormap = plt.cm.nipy_spectral
parser = argparse.ArgumentParser(description='Graphing Utils')

parser.add_argument(
    '--directory',
    type=str,
    default=None,
    help='choose file')

parser.add_argument(
    '--plot-type',
    default=None,
    help='what to plot')

args = parser.parse_args()

keys = ['N1','N2','dt','size_tab','lr_tab','use_bias','no_reset','no_rhop','plain_data','update_rule','trace_decay','spiking','step','no_clamp','beta']
