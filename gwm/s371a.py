# -*- coding: utf-8 -*-
""" A RS matching checker based on SRP Section 3.7.1 Option 1, Approach 2

Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
import numpy as np
import matplotlib.pyplot as plt

from .draggables import DraggableText
from . import _rs_time_openmp as rst
from . import eqmodel as em

#%%
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    This function was from Internet.

    """
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


class SRP371_Option1_Approach2_Checker:
    '''check whether an acceleration time history satisfy the SRP 3.7.1
    Option 1 Approach 2 critera.

    Enveloping the 5%-damped design response spectra (DRS):
        (a) dt is small enought to have a Nyquest frequency of at least 50 Hz (
            dt <= 0.01 s).  A tottal duration of 20 s.
        (b) 5% damped RS should be calculated with a minimum of 100 pts per
            frequency decade, uniformly spaced on the log scale, from 0.1 Hz to
            50 Hz or the Nyquist frequency.
        (c) The 5% damped RS should have no more than 10% below the DRS at any
            frequency.  No larger than +-10% frequency window fall below the
            DRS (e.g., 9 points below with 100 points in per frequency decade).
        (d) The 5% damped RS should not exceed DRS by 30%.

    The other criterion in (d), PSD check, should be checked separately.

    RG 1.60 RS --> SRP 3.7.1 Appendix A for target PSD function.

    Other RS --> SRP 3.7.1 Appendix B for guidliens and procedures for
    generation of target PSD compatible with the RS. "Procedures used to
    generate the target PSD will be reviwed on a case-by-case basis."

    The PSD requirement is secondary and minimum requirements to prevent
    potential deficiency of power over the frequency range of interest.

    '''

    def __init__(self, tfreq, tsa, acc, damping=0.05,
                title = 'SRP 3.7.1 Option 1, Approach 2, '\
                'Response Spectra Check',
                start_freq=None,
                end_freq=None,
                show=False):

        self.cutoff_freq = max(50.0, 0.5 / acc.dt)
        tfreq01_100 = em.freq_SRP371_Option1_Approach2()
        if start_freq is not None and end_freq is not None:
            ifs, ife = tfreq01_100.searchsorted([start_freq, end_freq])
            self.tfreq = tfreq01_100[ifs:ife]
        else:
            self.tfreq = tfreq01_100
        self.periods = 1.0 / self.tfreq
        self.tsa = em.loglog_interp(self.tfreq, tfreq, tsa)
        self.acc = acc
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.damping = damping
        self.title = title + ': ' + self.acc.name
        self.show = show
        self.calc_response_spectra()
        # setup plot
        self.__init_plots()


    def __print_summary_statistics(self, event):
        '[p]: Print a summary of the ground motion statistics'
        
        print(self.results)

    # --
    def calc_response_spectra(self):
        # response spectrum
        # ret = rst.rst_exactmethod(dampings, pd, acc, dt)
        # ret: sd, sv, sa, ta, pa, fs
        sd, sv, sa, ta, pa, fs = rst.rst_exactmethod((self.damping, ),
                self.periods, self.acc, self.acc.dt)
        self.sa = sa[0]
        # Fourier spectrum and power spectrum

    def __init_plots(self):
        def check_op1_app2(tsa, sa):
            ratio = sa / tsa
            below = ratio < 1.0
            below09 = ratio < 0.9
            above130 = ratio > 1.3
            ravg = ratio.mean()
            rmax = ratio.max()
            rmin = ratio.min()
            #calc how many adjacent frequency points below target spectra
            below = ratio < 1.0

            # # method 1 for consecutive below 1.0
            # window = np.ones(10, 'i')
            # count_below = np.convolve(window, below, 'valid')
            # num_adj_below = count_below.max()

            # method 2 for consectuive below 1.0
            all_below_ranges = contiguous_regions(below)
            if all_below_ranges.size > 0:
                imaxrange = np.argmax(all_below_ranges[:,1] - all_below_ranges[:,0])
                if1, if2 = all_below_ranges[imaxrange]
                num_adj_below = if2 - if1
            else:
                num_adj_below = 0
                if1 = if2 = 0

            stat_str = '''Damping Ratio = {:.1%}
    #Adj. Points Below = {} (<= 9)
    Smallest Spectral Ratio = {:.2f} (>= 0.9)
    Largest Spectral Ratio = {:.2f} (<= 1.3)
    Average Spectral Ratio = {:.2f} (~1.0)
    '''.format(self.damping, num_adj_below, rmin, rmax, ravg)
            return stat_str, self.tfreq[below09], sa[below09], \
                    self.tfreq[above130], sa[above130], \
                    self.tfreq[if1:if2], sa[if1:if2]


        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        ax = self.ax

        # plot target spectrum and the lower and bounds
        ax.semilogx(self.tfreq, self.tsa, linewidth=2)
        ax.semilogx(self.tfreq, 0.9*self.tsa, ':g')
        ax.semilogx(self.tfreq, 1.3*self.tsa, ':b')

        # plot curves
        ax.semilogx(self.tfreq, self.sa, )# marker='.', markeredgecolor='k')

        # plot checking results
        txt_result, freq_below90, sa_below90, freq_above130, sa_above130, \
            freq_range, sa_range = check_op1_app2(self.tsa, self.sa)
        self.results = txt_result
        # tmp = ax.text(0.02, 0.98, txt_result,
        #     horizontalalignment='left',
        #     verticalalignment='top',
        #     transform = ax.transAxes,
        #     #~ backgroundcolor='0.95',
        #     #~ alpha=0.2,
        #     )
        # self.dtext = DraggableText(tmp) # keep in self to be draggable
        self.dtext = ax.annotate(txt_result,
                                 xy=(0.02, 0.98),
                                 xycoords='axes fraction',
                                 xytext=None,
                                 textcoords='axes fraction',
                                 ha='left',
                                 va='top',
            # transform = ax.transAxes,
            #~ backgroundcolor='0.95',
            #~ alpha=0.2,
            )
        self.dtext.draggable()
        ax.semilogx(freq_below90, sa_below90, 'bo')
        ax.semilogx(freq_above130, sa_above130, 'go')
        if len(sa_range) > 9:
            print('\nRange of (frequency, Sa) below DRS:\n', list(zip(freq_range, sa_range)))
            ax.semilogx(freq_range, sa_range, 'ro')

        ax.set_ylabel('Spectral Acceleration (g)')
        ax.set_xlabel('Frequency (Hz)')
        ax.xaxis.grid(True, which='minor', linestyle='-',
           linewidth=0.1, color='gray', alpha=0.5)
        ax.xaxis.grid(True, which='major', linestyle='-',
           linewidth=0.15, color='gray')
        ax.yaxis.grid(True, which='major', linestyle='-',
                linewidth=0.15, color='gray')

        self.fig.suptitle(self.title, fontsize=16)
        #~ plt.tight_layout()
        if self.show:
            plt.show()
        
    def close(self):
        self.fig.close()

