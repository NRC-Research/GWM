# -*- coding: utf-8 -*-
""" Test using RG 1.60 DRS
Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
from pathlib import Path
import numpy as np
import random

import gwm.greedy_wavelet_method as wm
import gwm.eqio as eqio
import gwm.eqmodel as em

# target RS
freq0 = np.array([0.1, 0.25, 2.5, 9.0, 33.0, 100.0])
trs0 = np.array([0.0226, 0.14, 0.94, 0.78, 0.3, 0.3]) 
damping = 0.05
freq_cutoff = 100.00

tol = 0.05 # default 5% tolerance RspMatch09 paper uses 5%
# seed
seed_file = 'RSN9_BORREGO_B-ELC090.AT2' # change this for new seed

# read in seed
dt, acc = eqio.read_PEER_NGA_AT2(seed_file)
seed = np.asarray(acc)

# interpolate to 301 frequencies and apply cutoff frequency
freq301 = em.freq_SRP371_Option1_Approach2(301)
id_cutoff = freq301.searchsorted(freq_cutoff)
freq = freq301[:id_cutoff]
trs = em.loglog_interp(freq, freq0, trs0)

# run GWM
m = wm.WaveletMatch(dt, seed, 
                    freq, trs,
                    accname=seed_file,
                    scaling='SA', # scalling to PSA does not converge quickly
                    for_design=False,
                    zpa_clipping=True,
                    maxiter=300,
                    match_on_select=True,
                    tol=tol,
                   )
