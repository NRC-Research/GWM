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

#%%
HERE = Path(__file__).parent    

#%%
def test_rg_1_60(s=None):
    freq160 = np.array([0.1, 0.25, 2.5, 9.0, 33.0, 100.0])
    rs160 = np.array([0.0226, 0.14, 0.94, 0.78, 0.3, 0.3]) /0.3 # pga = 1 g
    if s is None:
        s = random.choice(list(HERE.glob('*.AT2')))
        name = s.name
        s = str(s)
        print('Using randomly selected seed', name)
    else:
        name = s
        s = str(HERE / s)
        print('Using default seed', name)
        
    dt, acc = eqio.read_PEER_NGA_AT2(s)
    seed = np.asarray(acc)
    damping = 0.05

    freq11 = em.freq_SRP371_Option1_Approach2(301)
    id33 = freq11.searchsorted(33.0)
    freq = freq11[:id33]
    rs = em.loglog_interp(freq, freq160, rs160) * 0.3
    
    # get target PSD
    nyquist = 1.0 / (2 * dt)
    psd_nyquist = 74.2/100**2 * (16/nyquist)**8 # m**2/s**3
    Nt = len(seed)
    T = Nt * dt
    Nf = Nt // 2 + 1
    df = 1.0 / T
    fo = np.arange(Nf) * df
    minfreq = min(0.01, df)
    psd_minfreq = 0.419 * (minfreq/2.5)**0.2 # m**2/s**3
    maxfreq = fo[-1]
    psd_maxfreq = 74.2e-4 * (16.0 / maxfreq)**8 # m**2/s**3
    # table_freq = np.array([0.1, 2.5, 9.0, 16.0, 100.0])
    # psd_appa = np.array([0.220, 0.419, 0.0418, 0.00742, 3.19e-09]) # m**2/s**3
    table_freq = np.array([minfreq, 2.5, 9.0, 16.0, maxfreq]) #100.0])
    psd_appa = np.array([psd_minfreq, 0.419, 0.0418, 0.00742, psd_maxfreq]) #3.19e-09]) # m**2/s**3
    psd_appa /= 96.17038422249999 # in ggs
    # ggs = 149064.39367366233 inch**2/s**3
    # psd_appa *= 149064.39367366233 # inch**2/s**3
    table_psd = psd_appa * 0.3 * 0.3 # make it compatible with PGA=0.3 g
    m = wm.WaveletMatch(dt, seed, freq, rs,
                     accname=name,
                     tol=0.05,
                     scaling='SA',
                     for_design=True,
                     zpa_clipping=True,
                     maxiter=300,
                     # match_on_select=False,
                     use_mpl_iter=True,
                     psdfreq=table_freq, 
                     targetpsd=table_psd,
                     minpsd_ratio=0.8
                     )
    return m

#%%
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 2:
        m = test_rg_1_60() # randomly pick a seed from the PEER folder
    else:
        m = test_rg_1_60('RSN9_BORREGO_B-ELC090.AT2')
        