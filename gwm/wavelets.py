# -*- coding: utf-8 -*-
""" Wavelets defined for GWM - Greedy Wavelet Method for response spectrum matching

All wavelets must have the same signature like the default wavelet

    wavelet_gaussian(f, damping, t, tj, gamma=2, Dtj=None)

TODO:
    Make gamma as an array or function call so a wavelet does not need to 
    use if statement
    
Jan 08, 2024

@author: JS Nie @ US NRC
"""

#%%
from functools import partial
import numpy as np

_GWM_wavelets_dict = {} 

def get_wavelet_names():
    return list(_GWM_wavelets_dict)

def get_wavelet(wavelet_name=''):
    # print('Set wavelet:', wavelet_name)
    # defaults to the current fastest exponential + gamma=3
    return _GWM_wavelets_dict.get(wavelet_name, wavelet_exponential)

def register(name, wl):
    assert name not in _GWM_wavelets_dict, f'{name} is already registered!'
    _GWM_wavelets_dict[name] = wl
    
    
def wavelet_gaussian(f, damping, t, tj, gamma=1, Dtj=None):
    '''return a wavelet described in Eq. (16) in Atik and Abrahamson (2010)

    Atik and Abrahamson (2010), "An Improved Method for Nonstationary Spectral
    Matching," Earthquake Spectra, 26(3), pp. 601-617, August 2010

    The Gaussian wavelet, as described in the paper, does not introduce drifts
    in the velocity and displacement time hisotries.

    f - freuqnecy (Hz) of the oscillator for computing RS 
    damping - damping of the oscillator
    t - array for the entire time duration
    tj - time for the peak response of oscillator j subjected to the acceleration 
    '''
    w = 2 * np.pi * f
    R1B2 = np.sqrt(1.0 - damping*damping)
    wpj = w * R1B2
    fp = f * R1B2
    # Dtj - the difference between the time of peak response tj and the reference origin of the wavelet
    # Eq. (14)
    # Dtj is critically important. 
    if Dtj is None:
        Dtj = np.arctan(R1B2 / damping) / wpj
    
    # Eq (17)
    if gamma == 1:
        gamma = 1.178 / f**0.93
    elif gamma == 2:
        gamma = 1.0 / fp
    else:
        gamma = 1.0 / f
    ttDt = t - tj + Dtj
    # Eq (16)
    ff = np.cos(wpj * ttDt) * np.exp(-(ttDt / gamma)**2)
    return ff, tj-Dtj, gamma

register('Gaussian', wavelet_gaussian) 
register('Gaussian1', partial(wavelet_gaussian, gamma=1))
register('Gaussian2', partial(wavelet_gaussian, gamma=2))
register('Gaussian3', partial(wavelet_gaussian, gamma=3))


def wavelet_exponential(f, damping, t, tj, gamma=3, Dtj=None):
    '''return a wavelet described in Eq. (13) in Atik and Abrahamson (2010), with alpha
    replaced by 1/gamma. This factor should also consider beta*wp, as proposed by
    Hancock et al. 2006 and Suarez and Montejo [2003,2005]

    f - freuqnecy (Hz) of the oscillator for computing RS 
    damping - damping of the oscillator
    t - array for the entire time duration
    tj - time for the peak response of oscillator j subjected to the acceleration 
   
    The tapered cosine wavelet was introdued by Abrahamson (1992). Hancock et al. (2006)
    provided a corrected tappered cosine wavelet, and Al Atik and Abrahamson (2010) proposed 
    the improved tapered cosine wavelet.
    
    REFERENCES:
    
    Abrabamson, N. A. [I9921 "Non-stationary spectral matching," Seisnaological Research
    Letters 63(1), 30. 
    
    Atik and Abrahamson (2010), "An Improved Method for Nonstationary Spectral
    Matching," Earthquake Spectra, 26(3), pp. 601-617, August 2010
     
    Hancock, J., Watson-Lamprey, J., Abrahamson, N. A., Bommer, J. J., Markatis, A., McCoy, E.,
    and Mendis, R., 2006. An improved method of matching response spectra of recorded 
    earthquake ground motion using wavelets, J. Earthquake Eng. 10, 67-89.
    
    Suarez, L. E. and Montejo, L. A. [2003] "Generacion de registros artificiales compatibles
    con un espectro de respuesta mediante la transformada wavelet," Proceedings of II 
    Congreso Nacional de Ingenieria Sismica, Medellin. 
    
    Suarez, L. E.' and Montejo, L. A. [2005] "Generation of artificial earthquakes via 
    the wavelet transform," International Journal of So1zd.s and Stmctures, 42(21-22), 
    5905-5919. 

    '''
    w = 2 * np.pi * f
    R1B2 = np.sqrt(1.0 - damping*damping)
    wpj = w * R1B2
    fp = f * R1B2
    # Dtj - the difference between the time of peak response tj and the reference origin of the wavelet
    # Eq. (14)
    # Dtj is critically important. 
    if Dtj is None:
        Dtj = np.arctan(R1B2 / damping) / wpj
    
    # Eq (17)
    if gamma == 1:
        gamma = 1.178 / f**0.93
    elif gamma == 2:
        gamma = 1.0 / fp
    elif gamma == 3:
        # beta * wp
        gamma = 1.0 / (damping * wpj)
    elif gamma == 4:
        gamma = 1.474 * f ** -0.833
    else:
        gamma = 1.0 / f
    ttDt = t - tj + Dtj
    # Eq (16)
    ff = np.cos(wpj * ttDt) * np.exp(-np.abs(ttDt / gamma))
    return ff, tj-Dtj, gamma

register('Exponential', wavelet_exponential) 
register('Exponential1', partial(wavelet_exponential, gamma=1))
register('Exponential2', partial(wavelet_exponential, gamma=2))
register('Exponential3', partial(wavelet_exponential, gamma=3))
register('Exponential4', partial(wavelet_exponential, gamma=4))
register('Exponential5', partial(wavelet_exponential, gamma=5))

def wavelet_tseng():
    pass


#%%
def test():
    import sys
    from pathlib import Path
    _PATH = Path('../..').resolve()
    if _PATH not in sys.path:
        sys.path.insert(0, _PATH)
        
    import matplotlib.pyplot as plt
    import GWM.eqplot as ep
    fig, (ax1, ax2) = plt.subplots(2, 1)
    damping = 0.05 # 5%
    dt = 0.005 # sec
    num = 4096
    T = num * dt
    t = np.arange(0.0, T, dt)
    # w01 = wavelet_gaussian(0.5, damping,  t, 10.0)
    w05, *_ = wavelet_gaussian(0.5, damping,  t, 10.0, gamma=1)
    w05p, *_ = wavelet_gaussian(0.5, damping,  t, 10.0, gamma=2)
    w05_3, *_ = wavelet_gaussian(0.5, damping,  t, 10.0, gamma=3)
    
    w5, *_ = wavelet_gaussian(5, damping,  t, 12.0, gamma=1)
    w5p, *_ = wavelet_gaussian(5, damping,  t, 12.0, gamma=2)
    w5_3, *_ = wavelet_gaussian(5, damping,  t, 12.0, gamma=3)
    
    w30, *_ = wavelet_gaussian(30, damping,  t, 15.0, gamma=1)
    w30p, *_ = wavelet_gaussian(30, damping,  t, 15.0, gamma=2)
    w30_3, *_ = wavelet_gaussian(30, damping,  t, 15.0, gamma=3)
    # w03 = wavelet_gaussian(50, damping,  t, 10.0)
    # ax1.plot(t, w01)
    ax1.plot(t, w05)
    ax1.plot(t, w05p)
    ax1.plot(t, w05_3)
    ax1.plot(t, w5)
    ax1.plot(t, w5p)
    ax1.plot(t, w5_3)
    ax1.plot(t, w30)
    ax1.plot(t, w30p)
    ax1.plot(t, w30_3)    
    # ax1.plot(t, w03)
    # print(ax1.get_ylim())
    ax1.vlines([10, 12, 15], *ax1.get_ylim(), color='gray')
    ax1.set_title(f'Wavelet_Atik, {damping=:.2f}')
    ax1.hlines(0.0, 0, t[-1], lw=0.5, color='gray')
    freq, rslist = ep.ax_plot_rs(ax2, dt, (w05, w05p, w05_3, w5, w5p, w5_3, w30, w30p, w30_3,),
                                 ('0.5 HZ-1.178 / f**0.93', 'D0.5 HZ-1.0 / fp', 'D0.5 HZ-1.0 / f', 
                                 '5 HZ-1.178 / f**0.93', 'D5 HZ-1.0 / fp', 'D5 HZ-1.0 / f', 
                                 '30 HZ-1.178 / f**0.93', 'D30 HZ-1.0 / fp', 'D30 HZ-1.0 / f', ),
                                 dmp_list=[damping * 100], # in percent
                                 g_unit=1.0)
    # freq, rslist = ep.ax_plot_rs(ax2, dt, (w01, w02, w03),
    #                              ('0.3 Hz', '5 Hz', '50 Hz'),
    #                              dmp_list=[damping * 100], # in percent
    #                              g_unit=1.0)
    ax2.vlines([0.5, 5, 30], 0, 5, color='gray')
    ax2.set_ylim(0, 5)

    plt.show()


if __name__ == '__main__':
    test()