# -*- coding: utf-8 -*-
""" Test using RG 1.60 DRS
Modified on Jul 09, 2026

@author: JS Nie @ US NRC
"""
import numpy as np
import gwm.greedy_wavelet_method as wm
import gwm.eqio as eqio
import gwm.eqmodel as em

# Define target RS
freq0 = np.array([0.1, 0.25, 2.5, 9.0, 33.0, 100.0])
trs0 = np.array([0.0226, 0.14, 0.94, 0.78, 0.3, 0.3]) 
damping = 0.05
freq_cutoff = 100.00

# Set convergence criterion
# RspMatch09 paper uses 5%
tol = 0.05 

# Read in or create the seed in any appropriate ways
seed_file = 'RSN9_BORREGO_B-ELC090.AT2' # change this for new seed
dt, acc = eqio.read_PEER_NGA_AT2(seed_file)
seed = np.asarray(acc)

# Interpolate to 301 frequencies and apply cutoff frequency
# Choose or define a frequency density to meet your need
freq301 = em.freq_SRP371_Option1_Approach2(301)
id_cutoff = freq301.searchsorted(freq_cutoff)
freq = freq301[:id_cutoff]
trs = em.loglog_interp(freq, freq0, trs0)

# run GWM
m = wm.WaveletMatch(dt, seed, 
                    freq, trs,
                    accname=seed_file,
                    scaling='SA', # one-time scaling to SA usually helps converge quickly
                    for_design=False,
                    zpa_clipping=True,
                    maxiter=300,
                    match_on_select=True,
                    tol=tol,
                   )
