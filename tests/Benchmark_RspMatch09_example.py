# -*- coding: utf-8 -*-
""" Benchmark Example using RspMatch09_v2 data
Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
#%%
import sys
from pathlib import Path

import numpy as np
import gwm.greedy_wavelet_method as wm

#%%
def test_rspmatch09_v2_data():
    trs = np.load('cms_T0.2_horiz.npy')
    seed = np.load('set3_h1_cmsT0.2.npy')
    freq = trs[:, 0]
    period = 1/freq
    id35 = freq.searchsorted(35.)
    dt1 = 0.0050
    beta = damping = 0.05
    m = wm.WaveletMatch(dt1, seed, freq[:id35], trs[:id35, 1],
                          accname='set3_h1_cmsT0.2.acc',
                          scaling='SA', # scalling to PSA does not converge quickly
                          for_design=False,
                          zpa_clipping=False,
                          maxiter=300,
                          match_on_select=True,
                          tol=0.05, # default 5% tolerance RspMatch09 paper uses 5%
                     )
    return m

#%%
if __name__ == '__main__':
    mr = test_rspmatch09_v2_data()
