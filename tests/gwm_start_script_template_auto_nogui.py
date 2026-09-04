# -*- coding: utf-8 -*-
""" Test using RG 1.60 DRS
Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
from pathlib import Path
import numpy as np

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
seed_files = ['RSN9_BORREGO_B-ELC090.AT2', # change this for new seed
              'RSN9_BORREGO_B-ELC000.AT2',
              'RSN9_BORREGO_B-ELC-UP.AT2',
              ]

# interpolate to 301 frequencies and apply cutoff frequency
freq301 = em.freq_SRP371_Option1_Approach2(301)
id_cutoff = freq301.searchsorted(freq_cutoff)
freq = freq301[:id_cutoff]
trs = em.loglog_interp(freq, freq0, trs0)

# create result directory
results_dir = 'Results_Auto'
Path(results_dir).mkdir(exist_ok=True)

def match(i, seed_file):
    print('\n*', i, seed_file)
    dt, seed = eqio.read_PEER_NGA_AT2(seed_file)
    # m = wavelet_match.WaveletMatch(
            # dt, seed, freq, trs,
            # accname=at2file.name,
            # zpa=1,
            # tol=0.05,
            # scaling='SA',
            # auto=True,
            # use_mpl_iter=False,
            # zpa_clipping=True,
            # results_dir='Results_PEER_301') 
    # run GWM
    m = wm.WaveletMatch(
            dt, seed, freq, trs,
            accname=seed_file,
            drsname="RG1.60_HOR",
            # zpa=0.3,
            tol=tol,
            scaling='SA', # scalling to PSA does not converge quickly
            auto=True,
            use_mpl_iter=False,
            for_design=True,
            zpa_clipping=True,
            results_dir=results_dir,
            # maxiter=300,
            # match_on_select=True,
            )
    m.match(2000) # match with a maximum of 2000 iterations/wavelets
    m.on_close(None)
    

if __name__ == '__main__':
    for i, seed_file in enumerate(seed_files):
        match(i, seed_file)
