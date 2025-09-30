'''GWM - Greedy Wavelet Method

A greedy algorithm for wavelet-based time domain response spectrum matching 
(Nie, Graizer, and Seber, NED 2023. https://doi.org/10.1016/j.nucengdes.2023.112384)

Jan 08, 2024

Jan 25, 2024: Testing apply taper for each added wavelet. 
    Should this be iteratively for low frequency wavelets? It seems better just use taper once after RS convergence!!!

@author: JS Nie @ US NRC
'''
# import warnings

# import cgitb # a traceback manager for CGI scripts in Python std library
# cgitb.enable(format="text")
DEBUG = False

from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial, wraps  # , partialmethod
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import scipy.integrate as sci
import scipy.optimize as sco

if DEBUG:
    import sys
    print(sys.path)
    print(matplotlib.__version__)


from . import _baseline as eb  # eb.avd requires a(0) == 0 to make a true zero start
from . import _equtils as eu
from . import _rs_time_openmp as rst
from . import eqmodel as em
from .draggables import DraggableLine2D
from . import eqio
from ._equtils import splitcosinebell
from matplotlib.backends.qt_editor._formlayout import fedit
from matplotlib.widgets import Button, Cursor, Slider, RangeSlider, SpanSelector
from matplotlib.transforms import Bbox, TransformedBbox
from .mpl_utils import (LockableSpanSelector, message_figure_annotation,
                       mpl_iter, savefig_reduced_png)
from PyQt5.QtWidgets import (QFileDialog, QInputDialog, QProgressDialog, QSplashScreen, 
                             QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap
# from tqdm import tqdm

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

from .wavelet_match_controls import WM_Controls
from .wavelets import wavelet_gaussian, wavelet_exponential

# from matplotlib import MatplotlibDeprecationWarning
# warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

# import offo

from . import __version__ 

def trycatch(func):
    """ Wraps the decorated function in a try-catch. If function fails print out the exception. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as e:
            print(f"Exception in {func.__name__}: {e}")
    return wrapper

# Approach using QRunnable and QThreadPool gives about the same performance as Python concurrent.futures
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)
    
    
#%%
class char_iter:
    'provide a simplistic progress bar for loops'
    # this and tqdm causes the program to crash when chose seed or reset button
    # this should be placed in text version when this code is refactored
    def __init__(self, items, bar_char='.'):
        self.items = items
        self.bar_char = bar_char

    def __iter__(self):
        # called when for i in char_iter initiatied
        self.iter = iter(self.items)
        return self

    def __next__(self):
        try:
            print(self.bar_char, end='') #, flush=True)
            return next(self.iter)
        except StopIteration:
            print()
            raise StopIteration
           
class qprogressbar_iter:
    'provide a qprogress bar for loops'
    def __init__(self, parent, items):
        self.parent = parent
        self.items = items

    def __iter__(self):
        # called when for i in qprogressbar_iter initiatied
        self.iter = iter(self.items)
        self.qpbar = QProgressDialog(f"RS Matching...up to {len(self.items)} wavelets", 
                                     "Cancel", 0, len(self.items)-1, 
                                     self.parent.window)
        self.qpbar.canceled.connect(self.cancel)
        self.qpbar.setMinimumDuration(1_000) # 1 s
        self.qpbar.setWindowModality(Qt.WindowModal)
        self.qpbar.setValue(0)
        self.qpbar.show()
        self.parent.qpbar = self.qpbar # so parent can close qpbar after converence
        self.to_stop = False
        return self

    def __next__(self):
        if self.to_stop:
            self.qpbar.close()
            raise StopIteration
        
        try:
            n = next(self.iter)
            if n % 5 == 0:
                self.qpbar.setValue(n)
            return n
        except StopIteration:
            self.qpbar.close()
            # del self.qpbar
            raise StopIteration
    
    def cancel(self):
        self.to_stop = True
                 
#%%
def pyqt5_get_fname(toolbar=None, msg='Open Data file', format="Data files (*.csv)"):
    # from PyQt5 import QtGui
    start = ''
    fname = QFileDialog.getOpenFileName(
        toolbar,
        msg, 
        start, format)
    # print(fname)
    fname = str(fname[0])
    return fname

#%%
# def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2)

def rst_openmp(acc, dt, dampings, freq, /):
    '''calculate response spectra in using rs_exactmethod_openmp for
    an acceleration time histories
    '''
    # freq = em.freq_SRP371_Option1_Approach2()
    pd = 1.0 / freq
    ret = rst.rst_exactmethod(dampings, pd, acc, dt)
    # ret: sd, sv, sa, ta, pa, fs
    return ret

# %%
def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

#%%

def time2loc(time, dt=0.005):
    cloc = int(round(time / dt))
    return cloc

#%%
def psd2fas_fft(freq, psd, Nt, dt, interp='loglog', ends=0):
    'convert psd to Fourier Amplitude Spectrum (FAS)'
    freq = np.asarray(freq)
    psd = np.asarray(psd)
    T = Nt * dt
    Nf = Nt // 2 + 1
    df = 1.0 / T
    twopi = 2.0* np.pi
    fo = np.arange(Nf) * df
    if interp == 'loglog':
        if ends == 0:
            left = right = np.log10(psd.min()) - 100.0
        else:
            left = right = ends
        # ends=None to take end points
        So = em.loglog_interp(fo, freq, psd, left=left, right=right)
    elif interp == 'linear':
        So = np.interp(fo, freq, psd)#, left=0.0, righ=0.0)
    elif interp == 'no':
        So = psd  # this requires freq is the same as fo
    fas = np.sqrt(np.pi*T*So)/dt
    return fo, fas


#%%
class WaveletMatch:
    'Perform Response Spectrum Matching for one time history for a given damping ratio using wavelets'
    def __init__(self, dt, seed, tfreq, trs, 
                 ptrise=0.075, ptdecay=0.375, # percent of T, default to NUREG/CR-4357 Envelope Function B
                accname='Unnamed',
                zpa=None,
                damping=0.05,
                scaling='NoScaling',
                tol=0.05, # default 5% tolerance
                maxiter=200,
                gamma=1.0, # relaxation parameter (between 0 and 1) to damp the adjustments
                for_design=False,
                auto=False,
                results_dir='.',
                match_on_select=True,
                use_mpl_iter=True,
                zpa_clipping=False,
                allow_new_peak_times=True,
                seed_dict=None,
                save_png_interval=0,
                step_png_filename='', # also a flag to save png files for stepping
                # target PSD is used in Fourier plot for simplified PSD check
                # because the strong motion duration is not known without user interaction thru a new GUI tool
                psdfreq=None,
                targetpsd=None,
                minpsd_ratio=0.7, # 0.8 for SRP Section 3.7.1 Appendix A, 0.7 for Appendix B
                ):
        '''seed is a numpy array
        '''
        self.accname = accname
        self.target_psd = None
        # self.dt = dt
        # self.seed = seed
        self.tfreq = tfreq
        self.trs = trs
        self.zpa = zpa or trs[-1]
        self.damping = damping
        self.n_trise = round(ptrise * len(seed))
        self.n_tdecay = round(ptdecay * len(seed))
        self.psdfreq = psdfreq
        self.targetpsd = targetpsd 
        self.minpsd_ratio = minpsd_ratio
        assert minpsd_ratio in (0.7, 0.8), 'minpsd_ratio must be 0.7 or 0.8'
        # self.scaling = scaling
        # self.default_tol = tol
        # self.default_gamma = gamma
        self.DESIGN_P3 = 3
        # if for_design:
        #     # for SRP 3.7.1 Option 1 Approach 2 bounds
        #     # elif rmax/0.3 > (-rmin)/0.1: # useful for design purpose
        #     # match proportionally from above and from below based on 
        #     # the 0.3 and 0.1 ratios
        #     self.DESIGN_P3 = 3
        #     # self.RMAX = 0.3
        #     # self.RMIN = 0.1 
        #     # self.tol_above = self.default_tol * 3
        #     # self.tol_below = self.default_tol
        # else:
        #     self.DESIGN_P3 = 1
        #     # for equally close match from above and below
        #     # self.RMAX = self.RMIN = 0.1
        #     # self.tol_above = self.tol_below = self.default_tol
        self.rmax = 1.0 # 100%
        self.auto = auto
        self.results_dir = results_dir
        # self.match_on_select = match_on_select
        # self.use_mpl_iter = use_mpl_iter
        # self.zpa_clipping = zpa_clipping
        self.allow_new_peak_times = allow_new_peak_times
        self.seed_dict = seed_dict # for selecting a new seed from a dict: label->acc
        
        # options will be used by wavelet_match_controls for user interaction
        self.scaling_options = ['NoScaling', 'ZPA', 'SA', 'PSA']
        # self.options = {'ZPA Clipping': zpa_clipping,
        #                 'For Design': for_design,
        #                 'Match on Select': match_on_select,
        #                 'Use MPL Iter': use_mpl_iter,
        #                 'Max Iter': maxiter,
        #                 'Auto Save Interval': 10,
        #                 'Scaling': scaling, 
        #                 'Tolerence': tol,
        #                 'Gamma': gamma,
        #                 }
        # set global variables
        self.zpa_clipping = zpa_clipping # self.options['ZPA Clipping']
        self.for_design = for_design # self.options['For Design']
        self.match_on_select = match_on_select # self.options['Match on Select']
        self.use_mpl_iter = use_mpl_iter # self.options['Use MPL Iter']
        self.maxiter = maxiter # self.options['Max Iter']
        self.auto_save_interval = 10 # self.options['Auto Save Interval']
        self.scaling = scaling # self.options['Scaling']
        self.default_tol = tol # self.options['Tolerence']
        self.default_gamma = gamma # self.options['Gamma']
        self.save_png_interval = save_png_interval  # not working because the progress circle cannot be saved
        self.step_png_filename = step_png_filename 
        # set up GUI
        # set up figure
        self.io_pool = None 
        self.setup_GUI()

        # check zero padding
        self.min_tj = lambda f: 3.9223 * f**-0.845
        
        # Seed processing: RS, Tj, Psigns, etc.
        # this allows to choose a new seed from GUI
        self.orig_seed = seed
        self.orig_dt = dt
        self.proc_seed(seed, dt)
        
        # create self.acc and other initial states
        self.reset_acc()
        self.show()
    
    def show(self):
        plt.show()
        return self
    
    def flashSplash(self, time=2000):
        # about the same time as defined and called in wavelet_match_controls.py
        try:
            if self.splash.isVisible():
                self.splash.hide()
            else:
                self.splash.show()
        except:
            self.logo_file = str(Path(__file__).parent / 'resources/gwm_logo_transparent.svg')
            self.splash = QSplashScreen(QPixmap(self.logo_file).scaledToHeight(400, Qt.SmoothTransformation),
                                        Qt.WindowStaysOnTopHint)
            # By default, SplashScreen will be in the center of the screen.
            # You can move it to a specific location if you want:
            # self.splash.move(10,10)
            self.splash.show()
        finally:
            # Close SplashScreen after 2 seconds (2000 ms)
            QTimer.singleShot(time, self.splash.close)
    
    def show_logo(self, event=None):
        self.flashSplash(10_000) # show for 10 sec for each click
        
    def setup_GUI(self):
        self.fig = plt.figure(figsize=(9, 9))
        self.window = self.fig.canvas.manager.window
        self.flashSplash()
        # self.window.move(0, 0)
        from PyQt5 import QtWidgets
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.screen_height = sizeObject.height()
        self.screen_width = sizeObject.width()
        # self.window.resize(screen_width//2, screen_height-20) # -20 to avoid taskbar
        self.window.setGeometry(0, 30, 
                                int(self.screen_width * 0.5), 
                                self.screen_height - 61) # title bar and taskbar
       
        bottom = 0.059
        left = 0.1 # 0.059
        right = 0.984
        hspace = 0.075 # 0.158
        wspace = 0.2
        top = 0.95 # 0.985 
        axheight = (top - bottom - hspace) / 4
        axwidth = right - left
        self.axrs = plt.axes([left, top - axheight*3, axwidth, axheight*3])
        self.axth = plt.axes([left, bottom, axwidth, axheight])
        self.fig.align_ylabels([self.axrs, self.axth])
        # self.fig.tight_layout()

        # REMOVE DEFAULT KEY MAPS
        # THE FOLLOW MPL_DISCONNECT IS NECESSARY TO MAKE ON_KEY WORKS AS WELL AS
        # ZOOM/PAN WORKS FOR SELF.AXTH
        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id)
        # self.show_message = self.fig.canvas.manager.toolbar.set_message
        self.cids = {}
        self.cids['OnClose'] = self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.cids['OnKey'] = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.span = LockableSpanSelector(
              self.axrs, self.on_select_freq,
              'horizontal',
              useblit=True,
              props=dict(alpha=0.2, facecolor='tab:gray'),
            #   span_stays=True, # deprecated 
              interactive=True,
            #   drag_from_anywhere=True, # does not work well with log scale
        )      
        self.span.active = False
        self.span.set_visible(False)
        # self.status = message_figure_statusbar(
        #     self.fig, msg='', 
        #     x=0.5, y=0.48,
        #     )
        # self.draggable_status = DraggableXY(self.status) # keep a reference to make draggable work
        self.status = message_figure_annotation(
            self.fig, msg='', 
            x=0.72, y=0.41,
        )
        # print(self.options)
        # use QT2 buttons
        self.wm_controls = WM_Controls(
            self, 
            [    # name, action, toggle?
             [('Overall Controls', 0, 0), # first is the group name and location
                ('&Seed', self.run_new_seed),
                ('&Reset', self.run_reset_acc),
                ('&Undo', self.run_undo),
                ('--', None),
                ('Split Cos', self.run_split_cosine),
                ('Filter', self.run_filter),
                ('Enrich', self.run_enrich),
                # ('ZPA Clip', self.run_set_clipping, self.zpa_clipping),
                ('*PSD to FAS', self.run_psd2fas),
                ],
              [('Matching', 0, 1),
                # ('Maxiter', self.maxiter), # this should be text field
                ('Options', self.run_set_options),
                ('Match ', self.run_match),
                ('M Band', self.run_set_band, False),
                ('M 1   ', partial(self.run_n, 1)),
                ('M 5   ', partial(self.run_n, 5)),
                ('M 10  ', partial(self.run_n, 10)),
                ('Add WL', self.run_add_wavelet),
                
                ],
              [('Post Processing',0, 2 ),
                ('BL A-mean', self.remove_average_acc),
                ('BL V-trend', self.detrend_velocity),
                ('BL D-trend  ', self.baseline_spline),
                ('BL Polynomial', self.baseline_polynomial),
                ('BL Lagrange', self.run_baseline_lagrange),
                # ('BL EEMD    ', self.baseline_hht_eemd),
                ('WL@Time', self.run_add_wavelet_at_time, False),
                ('*Logo', self.show_logo),
                ],
              [('Misc', 0, 3),
                ('Summary', self.run_print_summary),
                ('History', self.plot_converge_history),
                ('Clean', self.run_clean_canvas),
                ('Plot AVD', self.run_plot_avd),
                ('Plot FS', self.run_plot_Fourier_spectra),
                ('Check2', self.run_check2),
                # ('--', None),
                ('Save', self.run_save),
                ],
            ]
        )
        self.show_message = self.wm_controls.set_title
        self.window.setWindowIcon(self.wm_controls.gwm_icon)
        self.avd_plot = self.fourier_spectra_plot = self.check_o1a2_plot = None
       
        # print(self.options)
        if self.auto:
            self.wm_controls.hide()
        
    def on_close(self, event):
        try:
            self.wm_controls.close()
        except:
            print('Failed to delete self.wm_controls!')
            pass
        plt.close('all')
    ### 
    # @trycatch    
    def proc_seed(self, seed, dt, accname=None):
        'proc_seed does not plot'
        self.seed = seed.copy()
        self.dt = dt
        if accname is not None: # set accname if provided
            self.accname = accname
        
        print(f'''=== Seed Information ===========================================
    {self.accname}
    {dt=:0.4f} s 
    Nyquist Frequency={1/(2*dt):0.4f} Hz  
    {len(seed)=}
    Duration={len(seed)*dt:0.4f} s
================================================================''')
        
        # calc seed response spectrum
        _, _, sa, times, psigns, _ = rst_openmp(
            self.seed, self.dt, [self.damping], self.tfreq
            )       
        self.seed_rs = sa[0]
        self.seed_tjs = times[0]
        self.seed_psigns = psigns[0]
        
        # print(scaling)
        if self.scaling == 'SA':
            # scale to average SA
            ratios = self.trs / self.seed_rs
            factor = statistics.geometric_mean(ratios)
            self.seed *= factor
            self.seed_rs *= factor
        elif self.scaling == 'PSA':
            # scale to Peak SA
            ifreq = np.argmax(self.trs)
            factor = self.trs[ifreq] / self.seed_rs[ifreq]
            self.seed *= factor
            self.seed_rs *= factor
        elif self.scaling == 'ZPA':
            # scale zpa to target design zpa
            accmax = max(np.abs(self.seed))
            if self.zpa is None:
                # use the SA at the highest frequency
                zpa = self.trs[-1]
                factor = zpa / accmax
            else:
                factor = self.zpa / accmax
            self.seed *= factor
            self.seed_rs *= factor
            # print(self.zpa, max(np.abs(seed)))
        
        pre_time = 0.0
        for tj, f in zip(self.seed_tjs, self.tfreq):
            mtj = self.min_tj(f)
            if tj < mtj:
                tadd = mtj - tj
                if pre_time < tadd:
                    pre_time = tadd
        self.seed_tjs += pre_time
        # print(self.dt, len(self.seed))
        # print(pre_time)
        self.numadd = int(np.ceil(pre_time / self.dt))
        # seed envelop to tame WL that was needed before the seed start
        taper_ratio = 0.1 # tappered with a splitcosinebell, 5% each side
        self.taper = np.r_[np.zeros(self.numadd), 
                           splitcosinebell(len(self.seed), taper_ratio)]
        self.seed = np.r_[np.zeros(self.numadd), 
                          self.seed] # pre zero padding
        self.numAcc = len(self.seed)
        self.numFreq = len(self.tfreq)
        self.time = np.arange(self.numAcc) * self.dt
        self.thrange = slice(self.numadd, self.numAcc)

        # parameters describing the overshape of the seed
        # self.PRE_TIME = self.numadd * self.dt
        # self.TJS_LOW = min(self.seed_tjs)
        # self.TJS_HIGH = max(self.seed_tjs)

    # @trycatch
    def reset_acc(self):
        self.rs = self.seed_rs.copy()
        self.tjs = self.seed_tjs.copy()
        self.psigns = self.seed_psigns.copy()

        # self.wavelets = np.empty((numAcc, numFreq), dtype=np.float32)
        self.acc = self.seed.copy()
        # self.last_acc = None
        self.last_accs = [] # for undo
        # close all old axes
        
        if self.targetpsd is not None:
            Nt = self.numAcc
            dt = self.dt
            # self.fas_psd is used in plot_Fourier_spectra
            # and in self.psd2fas
            self.freq_psd, self.fas_psd = psd2fas_fft(self.psdfreq, self.targetpsd, 
                                            Nt, dt, interp='loglog', ends=None)

        self.axrs.clear()
        self.axth.clear()

        self.axrs.set_xlabel('Frequency (Hz)')
        self.axrs.set_ylabel('Spectral Acceleration')
        self.axrs.set_title('Seed: ' + self.accname)
        self.axth.set_xlabel('Time (sec)')
        self.axth.set_ylabel('Acceleration')
        # self.axrs.semilogx(self.tfreq, self.trs, color='red')
        # self.axrs.semilogx(self.tfreq, 1.3*self.trs, color='blue', linestyle='--')
        # self.axrs.semilogx(self.tfreq, 0.9*self.trs, color='green', linestyle='--')
        # self.axrs.semilogx(self.tfreq, self.rs, color='darkblue', lw=3)
        self.axrs.loglog(self.tfreq, self.trs, color='red')
        self.axrs.loglog(self.tfreq, 1.3*self.trs, color='blue', linestyle='--')
        self.axrs.loglog(self.tfreq, 0.9*self.trs, color='green', linestyle='--')
        self.axrs.loglog(self.tfreq, self.rs, color='darkblue', lw=3)
        
        self.draft_curves = [] # to ermove from axes when cleaning
        self.aux_curves = [] # to turn invisible when cleaning
        for f in self.tfreq:
            l = self.axrs.axline((f, 0.1), (f, 0.11), color='gray', 
                                 lw=0.5, alpha=0.5, visible=False)
            self.aux_curves.append(l)
        self.axth.axvline(self.time[self.numadd], 0, 1, lw=1, 
                          color='black', alpha=0.9, linestyle='-.')
        # self.axth.axvline(self.TJS_LOW, 0, 1, lw=2, color='black')
        # self.axth.axvline(self.TJS_HIGH, 0, 1, lw=2, color='black')
        self.seed_curve, = self.axth.plot(self.time, self.seed, linestyle='--', alpha=0.8,
                                          lw=1, color='black')
        # self.axth.plot(self.tjs, self.seed[(self.tjs/self.dt).astype(int)], 'o', 'red')
        self.acc_curve, = self.axth.plot(self.time, self.acc, lw=2, 
                                         color='orange', alpha=0.5)
        self.axth.axhline(0.0, 0, 1, color='gray', lw=0.5)
        
        self.axrs.grid(visible=True, which='major', axis='both')
        self.axth.grid(visible=True, which='major', axis='x')
        
        self.match_status = 'Initialized'
        self.overall_match_status = ''
        self.status.set_text(self.match_status)
        self.status.set_visible(False)
        
        if self.avd_plot:
            plt.close(self.avd_plot)
            self.avd_plot = None
        
        if self.fourier_spectra_plot:
            plt.close(self.fourier_spectra_plot)
            self.fourier_spectra_plot = None
        
        if self.check_o1a2_plot:
            plt.close(self.check_o1a2_plot)
            self.check_o1a2_plot = None
        # The following lines commented out appear to be the cause of crashes after 
        # changing seeds or resetting
        # try:
        #     plt.close(self.avd_plot)
        # except:
        #     pass
        # finally:
        #     self.avd_plot = None
            
        # try:
        #     plt.close(self.fourier_spectra_plot)
        # except:
        #     pass
        # finally:
        #     self.fourier_spectra_plot = None
        self.axrs.autoscale()
        self.axth.autoscale()

        # self.fig.show()
        # plt.show(block=True)
            
        # internal states
        self.wavelet_history = defaultdict(int)
        self.freq_range = slice(self.numFreq)
        self.last_rs_curve = self.wavelet_curve = self.freq_line = None
        self.added_wavelet = None
        self.show_aux_lines = True
        self.imax = 0
        self.wl_count = 0
        self.checker_o1a2 = None
        self.pre_padded = False
        self.converge_history = []
        self.fig_convergence = None
        self.match_time = 0.0
        self.stepping_count = 0

        # tight_layout is not useful for direct creation of plt.axes
        # self.fig.tight_layout()
        if not self.auto:
            self.fig.canvas.draw_idle()
            self.fig.canvas.setFocus()   
            self.plot_avd()
            # self.fig.canvas.draw_idle()
            # self.window.setFocus(True)
            # self.window.activateWindow()
            # self.window.raise_()
            # self.window.show()
            
            # if plt.isinteractive():
            #     print('Interative')
            #     # eithef fig.show or flush_events works in Spyder Ipython Console
            #     # self.fig.show()
            #     self.fig.canvas.draw_idle()
            #     self.window.setFocus(True)
            #     self.window.activateWindow()
            #     self.window.raise_()
            #     self.window.show()
            #     # plt.show()
            # else:
            #     print('not interactive')
            #     # self.fig.canvas.draw_idle()
            #     plt.show(block=False) # block=True)
        # self.fig.canvas.setFocus()            

    def add_undo_stop(self):
        self.last_accs.append((self.wl_count, self.acc.copy()))
        
    def run_new_seed(self, event=None):
        ''
        if self.seed_dict:
            try:
                seed_name, done = QInputDialog.getItem(
                    self.wm_controls, 'Seed Selection', 
                    f'Select a new seed ({len(self.seed_dict)}):', 
                    self.seed_dict.keys()
                    )
                if done:
                    print(seed_name)
                    acc = self.seed_dict[seed_name]
                    print(acc)
                    seed = np.asarray(acc)
                    dt = acc.dt
                    self.proc_seed(seed, dt, acc.name)
                    # self.accname = acc.name
                    self.reset_acc()
                else:
                    print('No seed selected')
            except Exception as e:
                print(e)
        else:
            at2file = pyqt5_get_fname(msg='Choose a New Seed', 
                                    format="AT2 files (*.AT2)")
            if at2file:
                print(f'Using Seed: {at2file}')
                dt, seed = eqio.read_PEER_NGA_AT2(str(at2file))
                self.proc_seed(seed, dt, Path(at2file).name)
                # self.accname = Path(at2file).name
                self.reset_acc()
    
    # def run_set_clipping(self, event=None):
    #     print(self.zpa_clipping, end=' ')
    #     self.zpa_clipping = self.wm_controls.states['ZPA Clip'].isChecked()
    #     print('-->', self.zpa_clipping)
        
    def run_reset_acc(self, event=None):
        self.proc_seed(self.orig_seed, self.orig_dt) 
        self.reset_acc()
    
    def run_set_options(self, event=None):
        items = [('ZPA Clipping', self.zpa_clipping),
                ('For Design', self.for_design),
                ('Match on Select', self.match_on_select),
                ('Use MPL Iter', self.use_mpl_iter),
                ('Max Iter', self.maxiter),
                ('Auto Save Interval', self.auto_save_interval),
                #          current_value  + avaiable_options
                ('Scaling', [self.scaling] + self.scaling_options),
                ('Tolerence', self.default_tol),
                ('Gamma', self.default_gamma),
                ('Save PNG Interval', self.save_png_interval),
            ]
        
        def apply_callback(data):
            if data is None:
                return
            (self.zpa_clipping, self.for_design, self.match_on_select, self.use_mpl_iter,
                self.maxiter, self.auto_save_interval, self.scaling, self.default_tol,
                self.default_gamma, self.save_png_interval)  = data
            print([('ZPA Clipping', self.zpa_clipping),
                ('For Design', self.for_design),
                ('Match on Select', self.match_on_select),
                ('Use MPL Iter', self.use_mpl_iter),
                ('Max Iter', self.maxiter),
                ('Auto Save Interval', self.auto_save_interval),
                #          current_value  + avaiable_options
                ('Scaling', [self.scaling] + self.scaling_options),
                ('Tolerence', self.default_tol),
                ('Gamma', self.default_gamma),
                ('Save PNG Interval', self.save_png_interval),
            ])
            # update global variables
            # self.zpa_clipping = self.options['ZPA Clipping']
            # self.for_design = self.options['For Design']
            # self.match_on_select = self.options['Match on Select']
            # self.use_mpl_iter = self.options['Use MPL Iter']
            # self.maxiter = self.options['Max Iter']
            # self.auto_save_interval = self.options['Auto Save Interval']
            # self.scaling = self.options['Scaling']
            # self.default_tol = self.options['Tolerence']
            # self.default_gamma = self.options['Gamma']

            # self.options = {'ZPA Clipping': zpa_clipping,
            #             'For Design': for_design,
            #             'Match on Select': match_on_select,
            #             'Use MPL Iter': use_mpl_iter,
            #             'Max Iter': maxiter,
            #             'Auto Save Interval': 10,
            #             'Scaling': scaling, 
            #             'Tolerence': tol,
            #             'Gamma': gamma,
            #             }

        res = fedit(items, title="Wavelet Match Options", 
                    comment='Revise options as you like',
                    parent=self.wm_controls,
                    apply=apply_callback)

    
    def run_split_cosine(self, event=None):
        self.acc *= self.taper
        self.calc_rs()
        self.plot_current()
    
    def run_psd2fas(self, event=None):
        if self.targetpsd is not None:
            self.psd2fas() # change time history
        else:
            dlg = QMessageBox(self.wm_controls)
            dlg.setWindowTitle("No Target PSD Provided!")
            dlg.setText("This function requires target PSD function!!!")
            button = dlg.exec()
            # if button == QMessageBox.Ok:
            #     print("OK!")
        
    def run_set_band(self, event=None):
        self.set_freq_range()
    
    def run_plot_avd(self, event=None):
        self.plot_avd()
    
    def run_plot_Fourier_spectra(self, event=None):
        self.plot_Fourier_spectra()
        
    def run_check2(self, event=None):
        self.check_rs_o1a2()
        
    def run_n(self, n, event=None):
        # self.last_acc = self.acc.copy()
        # self.last_accs.append(self.acc.copy())
        self.add_undo_stop()
        for i in range(n):
            self.stepping_count += 1
            imax, amax = self.match_1(stepping=True)
            self.plot_current(to_draw=False)
            if imax is None:
                print(f'Convergence criterion met: {amax:.1%} <= {self.default_tol:.1%}')
                break
        self.fig.canvas.draw_idle()
        if self.step_png_filename: 
            step_png_fn = self.step_png_filename + f'_{self.stepping_count}.png'
            savefig_reduced_png(self.fig, step_png_fn)
    
    def run_match(self, event=None):
        # self.last_acc = self.acc.copy()
        # self.last_accs.append(self.acc.copy())
        # self.add_undo_stop()
        # self.maxiter = self.wm_controls.maxiter
        # self.maxiter = self.options['MaxIter']
        self.match()
        self.plot_current()
 
    def run_add_wavelet(self, event=None):
        self.add_undo_stop()
        self.add_a_wavelet()
        # self.plot_current()
    
    def run_add_wavelet_at_time(self, event=None):
        self.add_undo_stop()
        self.add_a_wavelet_at_time()
    
    def run_filter(self, event=None):
        # self.add_undo_stop()
        self.filter()
        # self.plot_current()

    def run_baseline_lagrange(self, event=None):
        # self.add_undo_stop()
        self.baseline_lagrange()
        # self.plot_current()
    
    def run_undo(self, event=None):
        self.undo_last()
        self.plot_current()
    
    def run_enrich(self, event=None):
        # self.add_undo_stop()
        self.enrich()
        self.plot_current()
    
    def run_clean_canvas(self, event=None):
        self.clean_canvas()
    
    def run_print_summary(self, event=None):
        self.print_summary()
        
    def run_save(self, event=None):
        self.dump(f'Converged-{self.rmax:.1%}-{self.accname}')
        
    def on_key(self, event):
        key = event.key
        # print(key)
        if key is None:
            return
        if key in '013456789':
            n = int(key)
            if n == 0:
                n = 10
            self.run_n(n)
        elif key == 'm':
            self.run_match()
        elif key == 'u':
            self.run_undo()
        elif key == 'a':
            self.run_add_wavelet()
        elif key == 'r':
            self.set_freq_range()
        elif key == 'o':
            self.reset_acc()
        elif key == 'e':
            # enrich frequency content by adding white noise
            self.run_enrich()
        elif key == 'f':
            # bandwidth filter 
            # need to pick upper bound frequency
            self.run_filter()
        elif key == 'b':
            # baseline correction
            self.run_baseline_lagrange()
        elif key == 'd':
            self.plot_avd()
        elif key == 'c':
            self.clean_canvas()
        # elif key == 'k':
        #     # psd check
        #     self.check_psd()
        elif key == '2':
            # approach 2 check
            self.check_rs_o1a2()
        elif key == 'p':
            self.print_summary()
        elif key == 's':
            self.dump(f'Converged-{self.rmax:.1%}-{self.accname}')
        elif key == 'f1':
            if self.wm_controls.isVisible():
                self.wm_controls.hide()
            else:
                self.wm_controls.set_location_relative_to_parent()
                self.wm_controls.show()
                self.window.activateWindow()
    
    def check_psd(self, event=None):
        'TODO '

    def check_rs_o1a2(self, event=None):
        'check response spectrum based on Option 1, Approach 2'
        from . import s371a as srp371
        acc = em.Accelerogram(dt=self.dt, data=self.acc.copy(),
                              unit='$g$',
                              name=self.accname)
        self.checker_o1a2  = srp371.SRP371_Option1_Approach2_Checker(
                tfreq=self.tfreq,
                tsa=self.trs,
                acc=acc,
                damping=self.damping,
                start_freq=self.tfreq[0],
                end_freq=self.tfreq[-1],
                title='Option 1 Approach 2 Check',
                show=False)
        
        
        def close_check_o1a2_plot(event):
            if self.check_o1a2_plot:
                self.check_o1a2_plot.canvas.mpl_disconnect(cid_check_o1a2_plot)
                self.check_o1a2_plot = None
            
        if self.check_o1a2_plot is None:
            self.check_o1a2_plot = plt.gcf()
            cid_check_o1a2_plot = self.check_o1a2_plot.canvas.mpl_connect(
                'close_event', close_check_o1a2_plot)
        
        plt.gca().set_yscale('log')
        checkerwin = self.checker_o1a2.fig.canvas.manager.window
        checkerwin.move(self.window.x() + self.window.width() + 1,
                    self.window.y())
        
        checkerwin.show()
    
    def plot_current(self, to_draw=True):
        if self.last_rs_curve:
            self.last_rs_curve.set_linewidth(0.5)
            self.last_rs_curve.set_color('gray')
            self.draft_curves.append(self.last_rs_curve)
        if self.wavelet_curve:
            self.wavelet_curve.set_linewidth(0.5)
            self.wavelet_curve.set_color('gray')
        if self.freq_line:
            self.freq_line.set_linewidth(2.0)
            
        self.last_rs_curve , = self.axrs.loglog(
            self.tfreq, self.rs, lw=3, color='r', alpha=0.8)
        if self.imax is not None:
            self.freq_line = self.axrs.axvline(
                [self.tfreq[self.imax]], 0, 0.1, color='black',
                lw=5)
            self.draft_curves.append(self.freq_line)
            t = self.tjs[self.imax]
            l = self.axth.axvline(t, 0, 1, color='gray', 
                                 lw=0.5, alpha=0.5)
            self.draft_curves.append(l)
        if self.pre_padded:
            self.seed_curve.set_data(self.time, self.seed)
            self.acc_curve.set_data(self.time, self.acc)
            self.pre_padded = False
        else:
            self.acc_curve.set_ydata(self.acc)
            
        # self.acc_curve.remove()
        # self.acc_curve, = self.axth.plot(self.time, self.acc, lw=2, 
        #                                  color='orange', alpha=0.5)
        if self.added_wavelet is not None:
            self.wavelet_curve, = self.axth.plot(
                self.time, self.added_wavelet, color='black', lw=3, alpha=0.9)
            self.draft_curves.append(self.wavelet_curve)
        # # refresh limits after set_ydata
        self.axth.relim()
        self.axth.autoscale_view(True, True, True)
        self.axrs.relim()
        self.axrs.autoscale_view(True, True, True)
        self.print_summary(show_status=True)
        if to_draw:
            self.fig.canvas.draw_idle()
    
    def clean_canvas(self, force_clean=False):
        if force_clean:
            self.show_aux_lines = False
        else:
            self.show_aux_lines ^= True
        if not self.show_aux_lines:
            # remove all draft curves
            for l in self.draft_curves:
                # l.set_visible(self.show_aux_lines)
                try:
                    l.remove()
                except:
                    pass
            self.draft_curves.clear()
        
        for l in self.aux_curves:
            l.set_visible(self.show_aux_lines)
        self.status.set_visible(self.show_aux_lines)
        # self.draft_curves.clear()
        # self.axrs.autoscale()
        # self.axth.autoscale()
        self.fig.canvas.draw_idle()

#%%
    def print_summary(self, show_status=None):
        rdiff = self.rs / self.trs - 1.0
        #
        if self.for_design: # options['For Design']:
            rdiff[rdiff[:] > 0.0] /= self.DESIGN_P3
        self.irmax = np.argmax(np.abs(rdiff))
        self.rmax = rdiff[self.irmax]
        if self.for_design and self.rmax > 0.0:
            self.rmax *= self.DESIGN_P3
        if self.for_design:
            ubound = self.default_tol*self.DESIGN_P3    
        else:
            ubound = self.default_tol
        self.overall_match_status = (f'Overall max. error={self.rmax:.1%} ~~ '
                                     f'[{-self.default_tol:.1%}, {ubound:+.1%}]')
        count_wavelets = f'Added {self.wl_count} wavelets.'
        print(self.overall_match_status)
        self.status.set_text('\n'.join((count_wavelets,
                                        self.match_status,
                                        self.overall_match_status,
                                        f'Time used for RS matching: {self.match_time:.3g} s')))
        visible = show_status or not self.status.get_visible()
        self.status.set_visible(visible)
        self.fig.canvas.draw_idle()

    def test_convergence(self):
        import math

        from scipy.interpolate import LSQUnivariateSpline
        if not self.converge_history:
            return 
        numwl = list(range(len(self.converge_history)))
        self.converge_spline = LSQUnivariateSpline(
            numwl, self.converge_history,
            range(10, numwl[-1], 10))
        self.converge_spline_deriv = self.converge_spline.derivative(n=1)
        if math.isclose(test:=self.converge_spline_deriv(numwl[-1]), 
                        0.0, abs_tol=0.1):
            converged = True
        else:
            converged = False
        print('IN test_convergence: Converged?', converged, test)
        return converged, self.converge_spline, self.converge_spline_deriv

    @staticmethod
    def _find_closest(myList, myNumber):
        """ Found from Internet
        Assumes myList is sorted. Returns closest value to myNumber.
    
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0, myList[0]
        if pos == len(myList):
            return pos - 1, myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos, after
        else:
            return pos - 1, before  
    
    def plot_converge_history(self, event=None):
        import matplotlib.ticker as mtick
        fig, ax = plt.subplots(1, 1, figsize=[6.4, 3],)
        win = fig.canvas.manager.window
        left = self.window.x() + (self.window.width() - win.width())//2
        top = self.window.y() + (self.window.height() - win.height() - 75)
        win.move(left, top)        
        self.fig_convergence = fig
        ax.plot(self.converge_history, lw=1)
        ax.axhline(self.default_tol, 0, 0.95, color='gray', lw=1, alpha=0.5)
        if len(self.converge_history) >= 3:
            # ax.text(0.5, 0.5, 
            #         (f'Max Error: {max(self.converge_history):.1%}\n'
            #         f'Min Error: {min(self.converge_history):.1%}\n'
            #         f'Last Error: {self.converge_history[-1]:.1%}'),
            #         transform=ax.transAxes,
            #         bbox=dict(boxstyle='round',
            #                   facecolor='red', 
            #                   alpha=0.5)
            #         )
            error_summary = (f'(Max/Min/Last Error: {max(self.converge_history):.1%}/'
                            f'{min(self.converge_history):.1%}/'
                            f'{self.converge_history[-1]:.1%})'
            )
            converged, conv_spline, conv_spline_deriv = self.test_convergence()
            ax.plot(conv_spline(range(len(self.converge_history))), 
                    color='red', lw=2, alpha=0.8)
            
            ax_rate = ax.twinx()
            ax_rate.set_ylabel('Convergence Rate')
            ax_rate.plot(conv_spline_deriv(range(len(self.converge_history))),
                          linestyle='--',
                          color='red', lw=2, alpha=0.8)
            ax_rate.axhline(0, 0.05, 1, color='gray', lw=1, alpha=0.5)
            fig.convergence_cursor = Cursor(ax_rate, horizOn=True, 
                            vertOn=True, useblit=True, 
                            color='black',
                            lw=1,
                            alpha=0.8,
                            # linestyle='-',
                            )
        else:
            ax_rate = None
            error_summary = ''
        ax.set_ylabel('Maximum Error')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)) # 1.0 to 100%
        ax.set_xlabel(f'Number of Wavelets {error_summary}')
        ax.set_title('Convergence History')
        fig.tight_layout()
        loc_undo = last_undo = None
        undo_to_curve = None
        saved_wlcs = [i for i, a in self.last_accs]
        # print(saved_wlcs)
        def on_click(event):
            nonlocal loc_undo, last_undo, undo_to_curve
            if event.inaxes in (ax_rate, ax):
                if event.key and event.key == 'control' and saved_wlcs:
                    wlc = round(event.xdata)
                    last_undo, loc_undo = self._find_closest(saved_wlcs, wlc)
                    # print(loc_undo, last_undo)
                    # print(wlc, loc_undo)
                    if undo_to_curve:
                        undo_to_curve.set_xdata([loc_undo, loc_undo])
                    else:
                        undo_to_curve = ax.axvline(loc_undo, 0, 1, 
                                                    color='black', linestyle='--')
                else:
                    # set limit
                    if on_click.old_ax_ylims:
                        ax.set_ylim(on_click.old_ax_ylims)
                        on_click.old_ax_ylims = None
                    else:
                        on_click.old_ax_ylims = ax.get_ylim()
                        ax.set_ylim((0, 1.0))
                    if ax_rate:
                        if on_click.old_ax_rate_ylims:
                            ax_rate.set_ylim(on_click.old_ax_rate_ylims)
                            on_click.old_ax_rate_ylims = None
                        else:
                            on_click.old_ax_rate_ylims = ax_rate.get_ylim()
                            ax_rate.set_ylim((-0.3, 0.1))  
                fig.canvas.draw_idle()
                # return
            
        def on_button_undo_to(event):
            nonlocal loc_undo, last_undo
            # print(loc_undo, last_undo)
            if (loc_undo is not None) and (last_undo is not None):
                self.undo_last(last=last_undo)
                # self.wl_count = loc_undo
                self.plot_current()
                del self.converge_history[loc_undo+1:]
                fig.canvas.mpl_disconnect(fig.cid)
                button_undo_to.disconnect(fig.cid_undo_to)
                plt.close(fig)
                # self.plot_converge_history()
        ax_undo_to = fig.add_axes([0, 0.93, .25, .07])
        button_undo_to = Button(ax_undo_to, 'Undo to Dashed Line')
        fig.cid_undo_to = button_undo_to.on_clicked(on_button_undo_to)
        
        on_click.old_ax_ylims = None
        on_click.old_ax_rate_ylims = None
        fig.cid = fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.draw_idle()
        plt.show(block=False)
        
    def set_freq_range(self):
        if self.freq_range.start is None:
            # Currently full range, need to set range
            # span selector
            # self.span.set_visible(True)
            self.span.active = True
        else:
            # restore to full range
            self.freq_range = slice(self.numFreq)
            print('New frequency range:', self.freq_range)
            # matplotlib version 3.6.3 does not have span.stay_rect
            # self.span.stay_rect.set_visible(False)
            self.span.set_visible(False)
            self.span.active = False
            self.fig.canvas.draw()
            # self.fig.canvas.draw_idle()
            
    def on_select_freq(self, fmin, fmax):
        imin = self.tfreq.searchsorted(fmin)
        imax = self.tfreq.searchsorted(fmax)
        if imin == imax:
            return
        self.freq_range = slice(imin, imax)
        print('New frequency range:', self.freq_range)
        if self.match_on_select:
            # self.last_acc = self.acc.copy()
            self.add_undo_stop()
            self.match()
            self.plot_current()
        
    def next_max_loc_top(self):
        # only process in the freq_range
        s = self.freq_range
        rdiff = self.rs[s] / self.trs[s] - 1.0
        if self.for_design: # options['For Design']:
            rdiff[rdiff[:] > 0.0] /= self.DESIGN_P3
        self.irmax = np.argmax(np.abs(rdiff))
        self.rmax = rdiff[self.irmax]
        # if self.options['For Design'] and self.rmax > 0.0:
        #     self.rmax *= 3
            
        # imax = rdiff.argmax()
        # imin = rdiff.argmin()
        # rmax = rdiff[imax]
        # rmin = rdiff[imin]
        # self.rmax = amax = max(abs(rmax), abs(rmin))
        start = self.freq_range.start or 0
        # rabove = rmax / self.tol_above
        # rbelow = -rmin / self.tol_below
        amax = abs(self.rmax)
        if abs(self.rmax) < self.default_tol:
            loc = None
        else:
            loc = self.irmax + start
        return loc, amax
            
        # if 0.0 <= rabove <= 1.0 and 0.0 <= rbelow <= 1.0:
        #     # converged
        #     loc = None
        #     if rabove > rbelow:
        #         amax = rmax
        #     else:
        #         amax = rmin
        # # elif rmax/0.3 > (-rmin)/0.1: # useful for design purpose
        # elif rmax/self.RMAX > -rmin/self.RMIN:  # useful for close match
        #     loc = imax + start
        # else:
        #     loc = imin + start
        # return loc, amax

    # def _key(self, k):
    #     r, i = k
    #     return r/self.RMAX if r > 0 else -r/self.RMIN
    
    # def next_max_loc(self):
    #     # only process in the freq_range
    #     s = self.freq_range
    #     rdiff = self.rs[s] / self.trs[s] - 1.0
    #     # sort rdiff
    #     rdiff = sorted(zip(rdiff, range(self.numFreq)[s]), 
    #                    key=self._key, 
    #                    reverse=True)

    #     print('\n', rdiff[0:4])
    #     # fstart = self.freq_range.start or 0
    #     for rmax, imax in rdiff:
    #         # imax += fstart
    #         f = self.tfreq[imax]
    #         mtj = self.min_tj(f)
    #         tj = self.tjs[imax]
    #         print((f'Trying {rmax:.3f}, {imax}: '
    #               f' {mtj=:.3f} <= {tj=:.3f} IN [{self.TJS_LOW:.3f},'
    #               f' {self.TJS_HIGH:.3f}]')
    #               )
    #         if mtj <= tj and self.TJS_LOW <= tj <= self.TJS_HIGH:
    #             self.rmax = rmax
    #             self.imax = imax
    #             amax = abs(self.rmax)
    #             if amax < self.default_tol:
    #                 loc = None
    #             else:
    #                 loc = self.imax
    #             return loc, amax
    #     else:
    #         # strange to be here
    #         print(rdiff)
    #         print(rmax, imax)
    #         print(mtj, tj, self.TJS_LOW, self.TJS_HIGH )
    #         print('*** none of rdiff meets the criteria')
    #         return None, 100.0
                
    def calc_num_pre_padding(self, imax):
        f = self.tfreq[imax]
        mtj = self.min_tj(f)
        tj = self.tjs[imax]
        pre_time = max(mtj - tj, 0.0)
        num_pre_padding = int(np.round(pre_time / self.dt))
        return num_pre_padding
    
        # if numadd > 0:
        #     pret_time = numadd * self.dt
        #     self.seed_tjs += pre_time
        #     self.tjs += pre_time
        #     self.seed = np.r_[np.zeros(numadd), self.seed] # pre zero padding
        #     self.acc = np.r_[np.zeros(numadd), self.acc] # pre zero padding
        #     self.numAcc = len(self.seed)
        #     self.time = np.arange(self.numAcc) * self.dt
        #     self.pre_padded = True
        #     print(f"Prepadded {numadd} 0's")
        #     padded = True
        # else:
        #     padded = False
        # return padded

    def match_1(self, stepping=False):
        'add waivlets one by one'
        # pick the largest ratio
        
        # imax, amax = self.next_max_loc()
        imax, amax = self.next_max_loc_top()
        self.converge_history.append(amax) # Actually amax before adding the next WL
        self.imax = imax
        if imax is None:
            return imax, amax
        
        self.wl_count += 1
        
        if stepping:
            self.wavelet_history[imax] += 1
            if (uses := self.wavelet_history[imax]) > 1:
                print(f'{uses} uses of Wavelet {imax} at {self.tfreq[imax]:.1f} Hz!')
            else:
                print(f'New wavelet {imax} at {self.tfreq[imax]:.1f} Hz.')
            print(f'Iter {self.wl_count}: Max. Error={amax:.1%} ~ Tol={self.default_tol:.1%}')
       
        # Prevent too many use of the last imax
        # issue is when a wavelet is used multiple times
        # if imax == self.numFreq - 1 and self.wavelet_history[imax] > 30:
        #     print('Too many use of last wavelet. Skiping')
        #     return None, amax
        self.commit_wavelet_addition(imax, self.default_gamma)
        return imax, amax

    def commit_wavelet_addition(self, imax, gamma):
        pmax = self.psigns[imax]
        tjmax = self.tjs[imax]
        rsdiff = (self.trs[imax] - self.rs[imax]) * pmax
        # need_baseline_correction = self.ensure_pre_padding(imax)
        
        f = self.tfreq[imax]
        wl, wl_tc, wl_width = wavelet_exponential(self.tfreq[imax], self.damping, 
                              self.time, tjmax, gamma=3)
        # wl, wl_tc, wl_width = wavelet_gaussian(self.tfreq[imax], self.damping, 
        #                       self.time, tjmax, gamma=1) # 1 did not solve problem with drift
        # BUT, using gamma=1 (RSPMatch09 paper wavelet width) makes the example converge with 392 wavelets.
        # while using gamma=2 (damped frequency based) makes the exmaple converge with 638 wavelets
        # using gamma=3 (undampled frequency basesd) makes the exmaple converge with 1132 wavlets.
        # so to use gamma = 1
        # self.ensure_pre_padding(imax)
        # num_pre_padding = self.calc_num_pre_padding(imax)
        # num_left = round((wl_tc - 2.25 * wl_width) / self.dt)
        # # num_peak = round(tjmax/self.dt)
        # num_pre_padding = self.numadd - num_left
        # if num_pre_padding > 0:
        #     print(num_pre_padding)
        #     print(len(wl), len(self.acc), len(self.taper))
        #     self.wl_tapered = wl_tapered = wl * self.taper
        #     _, _, sa, tjs, psign, _ = rst_openmp(wl_tapered, self.dt, [self.damping], np.array([f]))
        #     wlrs = sa[0][0]
        #     tj = tjs[0][0]
        #     p = psign[0][0] # for the most time p > 0, but at high frequecies, wl lacks of refined dt to have a fine representation
        #     print(tjmax, tj, p, pmax, wlrs)
        #     self.added_wavelet = (gamma * rsdiff * p / wlrs) * wl  
        #     self.psign = self.psigns[imax]  # how is self.psign used?
        #     self.acc += self.added_wavelet 
        # else:
        _, _, sa, tjs, psign, _ = rst_openmp(wl, self.dt, [self.damping], np.array([f]))
        wlrs = sa[0][0]
        tj = tjs[0][0]
        p = psign[0][0] # for the most time p > 0, but at high frequecies, wl lacks of refined dt to have a fine representation
  
        ndiff = int(np.round((tj - tjmax) / self.dt))
        # print(ndiff)
        # print('Wavelet peak time:', tj, ', Acc peak time:', tjmax,
        #       f'{tj/tjmax:.3f}')
        # print('Wavelet peak sign:', p, ', Acc peak sign:', pmax)
        # self.wlrs[:, fid] = rs
        # Adjustment for ndiff and inclusion of * p resolves the issue of RS blowing
        # up at very high frequencies. This is due to high frequency wavelets do
        # not have adequate points to have a suffciently smooth description.
        if ndiff > 0:
            wl = np.r_[wl[ndiff:], np.zeros(ndiff)]
        elif ndiff < 0: # this should not happen often based on testing
            wl = np.r_[np.zeros(-ndiff), wl[:ndiff]]
        self.added_wavelet = (gamma * rsdiff * p / wlrs) * wl  
        self.psign = self.psigns[imax]  # how is self.psign used?
        # self.acc += self.added_wavelet * self.taper # ensure wavelet conforms zero start and end of the seed
        self.acc += self.added_wavelet
            
        if self.zpa_clipping: # options['ZPA Clipping']:
            self.acc, clip_count = eu.clip_at_zpa(self.acc, self.zpa)

        self.calc_rs()


    def calc_rs(self):    
        _, _, sa, tjs, psign, _  = rst_openmp(self.acc, self.dt, [self.damping], self.tfreq)
        self.rs = sa[0]
        self.tjs = tjs[0] # disable time locations, and see if keep displacement better
        self.psigns = psign[0]

    def get_iter(self, maxiter):
        if self.use_mpl_iter and not self.fig.canvas.widgetlock.locked():
            return mpl_iter(self.fig, range(maxiter), stoppable=True)
        else:
            return qprogressbar_iter(self, range(maxiter))
        # elif not self.auto:
        #     return range(maxiter)
            # return char_iter(range(maxiter))
        # else:
        #     return tqdm(range(maxiter)) # tqdm causes GUI to crash if it is used for the second time

    def undo_last(self, last=None):
        if self.last_accs:
            # not empty
            if last is not None:
                del self.last_accs[last+1:]
            self.wl_count, self.acc = self.last_accs.pop()
            self.imax = None # so not to draw a short thick freq line
            self.calc_rs()
    
    def save_animation(self, i, rgba_buffer):
            # self.fig.savefig(f'GWMImage{i:03}.png') # This does not save the iter circle
            # self.fig.canvas.print_png(f'GWMImage{i:03}.png') # This does not save it either
            # the above command calls draw() which erases the animated cirle
            matplotlib.image.imsave(
                f'GWM_Animation_Image{i:03}.png', 
                rgba_buffer,
                format='png', 
                origin="upper", 
                dpi=self.fig.dpi, 
                metadata=None, 
                pil_kwargs=None
            )
                    
    def match(self, maxiter=None, tol=None):
        ''' Perform a mathcing pass using Improved tapered cosing function (wavelet)
        maxiter=5: Maximum number of iterations for spectral matching. This value is typically set between 5 and 20 depending on how close the initial response spectrum is to the target spectrum. 
        tol=0.05: Tolerance for maximum mismatch in fraction of target. This value is typically set to 0.05 for 5% maximum deviation from the target spectrum. 
        gamma=1: Convergence damping. This factor specifies the fraction of adjustment made to the acceleration time series at each iteration. This parameter is usually set to 1.
        
        This method follows:
            
        An Improved Method for Nonstationary Spectral Matching 
        by Linda Al Atika and Norman Abrahamson
        DOI: 10.1193/1.3459159
        Earthquake Spectra, Volume 26, No. 3, pages 601–617, August 2010 
        
        * A fundamental assumption of this methodology is that the time of the peak response does not change as a result of the wavelet adjustment.
        * Since the short period spectral acceleration is influenced by long period wavelets, only the short period range of the response spectrum is matched in the first pass. What about frequency by freqeuncy but from high to low?
        
        Experience indicates that maxiter=200 usually gets to a decent match. More
        iterations can be added by running match again.
        '''
        # use SRSS of wavelets RS would be a good approximation
        
        if self.save_png_interval > 0 and self.io_pool is None:
            # self.io_pool = QThreadPool()
            # self.io_pool.setMaxThreadCount(60)
            self.io_pool = ThreadPoolExecutor(max_workers=60)
            
        t0 = datetime.now()
        if self.fig_convergence:
            plt.close(self.fig_convergence)
            self.fig_convergence = None
        
        if maxiter is None:
            maxiter = self.maxiter # options['Max Iter']
        if tol is not None: # this should not use. Confusing design here.
            self.default_tol = tol

        # print(maxiter)
        auto_save_interval = self.auto_save_interval or 10 # options['Auto Save Interval'] or 10
        
        # main loop for RS matching
        for i in self.get_iter(maxiter):
            # save undo acc
            if i % auto_save_interval == 0:
                self.add_undo_stop()
            
            # save png files for each i for creating animation    
            if self.save_png_interval > 0 and i % self.save_png_interval == 0:
                # worker = Worker(self.save_animation, i, np.array(self.fig.canvas.buffer_rgba()))
                # self.io_pool.start(worker)
                self.io_pool.submit(self.save_animation, i, 
                                    # self.fig.canvas.buffer_rgba()
                                    np.array(self.fig.canvas.buffer_rgba())
                                    )
                
            # test tolerance
            imax, amax = self.match_1()
            if imax is None:
                self.match_status = (f'Convergence criterion met: {amax:.1%} '
                                     f'<= {self.default_tol:.1%}')
                print(self.match_status )
                if self.auto:
                    self.dump(f'Converged-{amax:.1%}-{self.accname}')
                    
                # try to close the qbar if exists
                if hasattr(self, 'qpbar'): # qpbar is added by qpprogressbar if used
                    self.qpbar.close()
                break
        else:
            self.match_status  = (f'Iteration: {i+1}, {maxiter=}!\n'
                      f'Convergence criterion not met: {amax:.1%} '
                      f'> {self.default_tol:.1%}')
            print(self.match_status )
            if self.auto:
                self.dump(f'Unconverged-{amax:.1%}-{self.accname}')
          
        t1 = datetime.now()
        self.match_time += (t1 - t0).total_seconds()
        if not self.auto:
            self.print_summary(show_status=True)
            self.plot_converge_history()
            if self.save_png_interval > 0: # always save last one 
                self.plot_current()
                self.fig.canvas.draw()
                self.save_animation(i + 1, 
                                    self.fig.canvas.buffer_rgba()
                                    ) 

    # -------------------------------------------------------------------------    
    # add a wavelet at a picked freq.
    # time is from tjmax
    def add_a_wavelet(self):
        'add waivlet by select a frequency'
        self.show_message('Pick a frquency (Hz)')
        self.fig.canvas.draw_idle()
        self.cursor_add_wl = Cursor(self.axrs, horizOn=True, 
                        vertOn=True, useblit=True, 
                        color='black',
                        linestyle='--',
                        )
        self.cursor_add_wl.connect_event(
            'button_press_event', 
            self.onclick_add_wavelet)
               
    def onclick_add_wavelet(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #         event.x, event.y, event.xdata, event.ydata))
        freq = event.xdata
        if hasattr(self, 'cursor_add_wl'):
            self.show_message()
            self.cursor_add_wl.disconnect_events()
            self.cursor_add_wl.linev.remove()
            self.cursor_add_wl.lineh.remove()
            del self.cursor_add_wl
        self.imax, self.amax = self.add_a_wavelet_at_freq(freq)
        # self.imax = None
        self.plot_current()

    def add_a_wavelet_at_freq(self, freq):
        ifreq = self.tfreq.searchsorted(freq)
        self.commit_wavelet_addition(ifreq, gamma=1.1)
        return ifreq, self.default_tol*2 # so not to meet convergence

    # -------------------------------------------------------------------------    
    # add wavelet at a picked freq and time
    def add_a_wavelet_at_time(self):
        'add waivlet by select a frequency at picked center time'
        self.wl_freq = self.wl_time = None
        if self.wm_controls.get_state('WL@Time'):
            print('WL@Time Pushed')
            self.show_message('Pick a frquency (Hz)')
            self.fig.canvas.draw_idle()
            self.cursor_add_wl_rs = Cursor(self.axrs, horizOn=True, 
                            vertOn=True, useblit=True, 
                            color='black',
                            linestyle='--',
                            )
            self.cursor_add_wl_rs.connect_event(
                'button_press_event', 
                self.onclick_add_wavelet_rs)
        else:
            print('WL@Time Released')
            # clean up
            try:
                self.cursor_add_wl_rs.disconnect_events()
                self.cursor_add_wl_rs.linev.remove()
                self.cursor_add_wl_rs.lineh.remove()
                del self.cursor_add_wl_rs
            except:
                pass
            try:
                self.cursor_add_wl_time.disconnect_events()
                self.cursor_add_wl_time.linev.remove()
                self.cursor_add_wl_time.lineh.remove()
                del self.cursor_add_wl_time
            except:
                pass
            try:
                self.wl_handle_rs.remove()
            except:
                pass
            try:
                self.wl_handle_th.remove()
            except:
                pass
            self.draggable_wl_handle_rs = self.draggable_wl_handle_th = None
            self.wl_handle_th = self.wl_handle_rs = None
            self.fig.canvas.draw_idle()
            
    def onclick_add_wavelet_rs(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #         event.x, event.y, event.xdata, event.ydata))
        freq = event.xdata
        if hasattr(self, 'cursor_add_wl_rs'):
            # self.show_message()
            self.cursor_add_wl_rs.disconnect_events()
            self.cursor_add_wl_rs.linev.remove()
            self.cursor_add_wl_rs.lineh.remove()
            del self.cursor_add_wl_rs
        self.wl_freq = freq
        self.show_message('Pick a time (sec)')
        self.fig.canvas.draw_idle()
        self.cursor_add_wl_time = Cursor(self.axth, horizOn=True, 
                            vertOn=True, useblit=True, 
                            color='black',
                            linestyle='--',
                            )
        self.cursor_add_wl_time.connect_event(
                'button_press_event', 
                self.onclick_add_wavelet_time)
            
    def onclick_add_wavelet_time(self, event):
        time = event.xdata
        if hasattr(self, 'cursor_add_wl_time'):
            self.show_message()
            self.cursor_add_wl_time.disconnect_events()
            self.cursor_add_wl_time.linev.remove()
            self.cursor_add_wl_time.lineh.remove()
            del self.cursor_add_wl_time
        self.wl_time = time
        self.imax, self.amax = self.add_a_wavelet_at_freq_time(
                    self.wl_freq, self.wl_time, gamma=1.0)
        self.plot_current()
        
    def add_a_wavelet_at_freq_time(self, freq, time, gamma=1.0):
        'add handles on axrs and axth for manual dragging'
        ifreq = self.tfreq.searchsorted(freq)
        imax = ifreq
        pmax = self.psigns[imax]
        tjmax = self.dt * round(time / self.dt)  
        rsdiff = (self.trs[imax] - self.rs[imax]) * pmax
        # rsdiff = self.trs[imax] # just to try jump to the target RS
        # need_baseline_correction = self.ensure_pre_padding(imax)
            
        f = self.tfreq[imax]
        wl, wl_tc, wl_width = wavelet_exponential(f, self.damping, 
                                  self.time, tjmax, gamma=3)
        # wl, wl_tc, wl_width = wavelet_gaussian(f, self.damping, 
        #                           self.time, tjmax, gamma=1) # 1 did not solve problem with drift
           
        _, _, sa, tjs, psign, _ = rst_openmp(wl, self.dt, [self.damping], np.array([f]))
        wlrs = sa[0][0]
        tj = tjs[0][0]
        p = psign[0][0] # for the most time p > 0, but at high frequecies, wl lacks of refined dt to have a fine representation
        # ndiff = int(np.round((tj - tjmax) / self.dt))
        # if ndiff > 0:
        #     wl = np.r_[wl[ndiff:], np.zeros(ndiff)]
        # elif ndiff < 0: # this should not happen often based on testing
        #     wl = np.r_[np.zeros(-ndiff), wl[:ndiff]]
        # wlfactor = (gamma * rsdiff * p / wlrs)
        wlfactor = (gamma * rsdiff / wlrs) # assuming jsut the effect of WL to reach TRS
        self.added_wavelet = wlfactor * wl  
        self.psign = self.psigns[imax]  # how is self.psign used?
        # self.acc += self.added_wavelet * self.taper # ensure wavelet conforms zero start and end of the seed
        self.acc += self.added_wavelet 
        
        # if self.options['ZPA Clipping']:
        #     self.acc, clip_count = eu.clip_at_zpa(self.acc, self.zpa)
        self.calc_rs()

        # add handles
        self.wl_handle_rs, = self.axrs.loglog(f, self.rs[imax], 'yo',
                                              markersize=10, 
                                              markeredgecolor='black',
                                              markeredgewidth=2,
                                              alpha=0.9,
                                              zorder=10
                                              )
        self.draggable_wl_handle_rs = DraggableLine2D(self.wl_handle_rs,
                                         user_draw=self.update_wl_freq,
                                         move_y=False,
                                         )
        if wlfactor > 0:
            tloc = self.added_wavelet.argmax()
        else:
            tloc = self.added_wavelet.argmin()
        thval = self.added_wavelet[tloc]
        thtime = tloc * self.dt
        self.wl_params = (tloc, thtime, thval, wlfactor)
        self.wl_handle_th, = self.axth.plot(thtime, thval, 'yo', 
                                            markersize=10, 
                                            markeredgecolor='black',
                                            markeredgewidth=2,
                                            alpha=0.9,
                                            zorder=10
                                            )
        self.draggable_wl_handle_th = DraggableLine2D(self.wl_handle_th,
                                         user_draw=self.update_wl_mag_time,
                                         # move_y=False,
                                         )

        return ifreq, self.default_tol*2 # so not to meet convergence

    def update_wl_freq(self, artist):
        ''
        print('update_wl_freq')
        self.acc -= self.added_wavelet 

        freq = float(artist.get_xdata())
        # rsmag = float(artist.get_xdata())
        self.imax = self.tfreq.searchsorted(freq)
        f = self.tfreq[self.imax]
        tloc, thtime, thval, wlfactor = self.wl_params 
        wl, wl_tc, wl_width = wavelet_exponential(f, self.damping, 
                                  self.time, thtime, gamma=3, Dtj=0.0)
        # wl, wl_tc, wl_width = wavelet_gaussian(f, self.damping, 
        #                           self.time, thtime, gamma=1, Dtj=0.0) # 1 did not solve problem with drift
        # wlfactor = rsmag / self.rs[self.imax]
        # self.wl_params = (tloc, thtime, thval, wlfactor)
        self.added_wavelet =  wlfactor * wl  
        # self.acc += self.added_wavelet * self.taper # ensure wavelet conforms zero start and end of the seed
        self.acc += self.added_wavelet 
        self.calc_rs()
        self.wavelet_curve.set_ydata(self.added_wavelet)
        if max(self.added_wavelet) >-min(self.added_wavelet):
            # positive up
            wlpeak= max(self.added_wavelet)
        else:
            wlpeak = min(self.added_wavelet)
        self.wl_handle_th.set_ydata(wlpeak)
        self.plot_current()
        # self.axrs.relim()
        # self.axth.relim()
        self.fig.canvas.draw_idle()
        
    def update_wl_mag_time(self, artist):
        ''
        print('update_wl_mag_time')
        self.acc -= self.added_wavelet 
        
        tjmax = round(float(artist.get_xdata())/self.dt) * self.dt # approximate
        ymag = float(artist.get_ydata())
        f = self.tfreq[self.imax]
        wl, wl_tc, wl_width = wavelet_exponential(f, self.damping, 
                                  self.time, tjmax, gamma=3, Dtj=0.0)
        # wl, wl_tc, wl_width = wavelet_gaussian(f, self.damping, 
        #                           self.time, tjmax, gamma=1, Dtj=0.0) # 1 did not solve problem with drift
        tloc, thtime, thval, wlfactor = self.wl_params 
        tloc = wl.argmax()
        thval = wl[tloc]
        thtime = tloc * self.dt
        
        wlfactor = ymag / thval
        self.added_wavelet =  wlfactor * wl  
        # self.acc += self.added_wavelet * self.taper # ensure wavelet conforms zero start and end of the seed
        self.acc += self.added_wavelet 
        self.wl_params = (tloc, thtime, thval, wlfactor)
        self.calc_rs()
        self.wavelet_curve.set_ydata(self.added_wavelet)
        self.wl_handle_rs.set_ydata(self.rs[self.imax])
        self.plot_current()
        self.fig.canvas.draw_idle()   

    def _baseline_lagrange(self, acc):
        # filter very low frequencies
        # fmin = self.tfreq[0] # / 2
        # fmax = 0 #self.tfreq[-1] * 3 # (1 + self.damping) / (1 - self.damping)
        # self._filter(fmin, fmax)
        # run baseline correction
        # acc = self.acc[self.thrange]
        # time = self.time[self.thrange]
        # updated acc
        uacc = eb.baseline_lagrange_multipliers(acc, self.dt)
        # self.calc_rs()    
        # self.plot_current()
        return uacc

    def _baseline_lagrange_vd_ends_only(self, acc):
        '''Following F. Borsoi and A. Richard, 1985, SMiRT8, K2-7
        "A Simple Accelerogram Corection Method to Prevent Unrealistic Displacement Shift"
        
        BUT Consider only Vn = 0; Dn = 0;
        Leaving out En = 0, which seams to create large cycle (?)
        '''
        # acc = self.acc[self.thrange]
        N = len(acc)
        h = self.dt
        
        # Appendix - Final velocity
        alpha = np.full_like(acc, h)
        alpha[-1] /= 2.0
        # Appendix - Final displacement
        beta = np.fromfunction(
            lambda i: (N - i - 1) * h*h, 
            (N,) )
        beta[-1] = h*h / 4
        
        # C {x} = b
        C = np.empty((2, 2), dtype=np.float64)
        C[0, 0] = alpha @ alpha
        C[0, 1] = C[1, 0] = alpha @ beta
        C[1, 1] = beta @ beta
        b = np.array([ alpha @ acc, beta @ acc])
        x = np.linalg.solve(C, b)
        
        # updated acc
        uacc = acc - x[0] * alpha - x[1] * beta
        return uacc

    def baseline_lagrange(self, event=None):
        self.add_undo_stop()
        acc = self.acc[self.thrange]
        time = self.time[self.thrange] - self.time[self.numadd]
        vel0, dis0 = self.a2vd()
       
        # several ugly local "globals"
        uacc = None
        def on_vde(event):
            'Vn = Dn = En = 0'
            nonlocal uacc
            uacc = self._baseline_lagrange(acc)
            update_plot(uacc)

        def on_vd(event):
            'Vn = Dn = En = 0'
            nonlocal uacc
            uacc = self._baseline_lagrange_vd_ends_only(acc)
            update_plot(uacc)
        
        cur_vel = cur_dis = None
        def update_plot(uacc):
            # print('On Click')
            nonlocal cur_vel, cur_dis, fig
            v, d = self.a2vd(uacc)
            if cur_vel is None:
                cur_vel, = axvel.plot(time, v)
                cur_dis, = axdis.plot(time, d)
            else:
                cur_vel.set_ydata(v)
                cur_dis.set_ydata(d)
                axvel.relim()
                axvel.autoscale_view(True, True, True)
                axdis.relim()
                axdis.autoscale_view(True, True, True)
            fig.canvas.draw_idle()
            
        VDONLY = 'Only Vel/Dis Ends'
        VDE    = 'Ends and Dis Ensemble'
        mosaic = [[VDONLY, VDE]] 
        mosaic.extend([['Vel'] * 2] * 2)
        mosaic.extend([['Dis'] * 2] * 2)
        fig, axs = plt.subplot_mosaic(mosaic, figsize=[6.4, 3], 
                                         constrained_layout=True)
        axvd = axs[VDONLY]
        buttonvd = Button(axvd, VDONLY)
        cid_vd = buttonvd.on_clicked(on_vd)
        axvde = axs[VDE]
        buttonvde = Button(axvde, VDE)
        cid_vde = buttonvde.on_clicked(on_vde) 
        # cid_vd and cid_ve NEED TO BE SAVED FOR ON_CLIKCED TO WORK.
        axvel = axs['Vel']
        axdis = axs['Dis']
        def onclick_update(event):
            # print('on button_press_event')
            if event.inaxes in (axvel, axdis):
                if event.key and event.key == 'control':
                    acc[:] = uacc 
                    self.calc_rs()
                    self.plot_current()
                    fig.canvas.mpl_disconnect(cid)
                    buttonvde.disconnect(cid_vde)
                    buttonvd.disconnect(cid_vd)
                    plt.close(fig)
                return
            # else:
            #     return event # this allows button.on_click to work
                
        axvel.plot(time, vel0)
        axvel.axhline(0, 0, 1, color='gray', lw=1)
        axvel.set_title('Ctrl+click velocity or displacement plot below to commit change')
        axvel.set_ylabel('Velocity')
        axdis.plot(time, dis0)
        axdis.axhline(0, 0, 1, color='gray', lw=1)
        axdis.set_ylabel('Displacement')
        axdis.set_xlabel('Time (s)')
        axdis.sharex(axvel)
        cid = fig.canvas.mpl_connect('button_press_event', onclick_update)
        # set fig location
        win = fig.canvas.manager.window
        sh3 = self.screen_height // 3 
        win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -20 to avoid taskbar
        win.show()      
        
    # def baseline_hht_eemd(self, event=None):
    #     # self.add_undo_stop()
    #     from .hht.eemd import eemd
    #     acc = self.acc[self.thrange]
    #     time = self.time[self.thrange]
    #     v, d = self.a2vd()
    #     self.rslt = rslt = eemd(d, 0.2, 10) # max(np.abs(d))/10, 10)
    #     # self.rslt = rslt = eemd(d, 0.0, 1)
    #     # A matrix of (m+1)*N matrix, where N is the length of the input
    #     # data x, and m=fix(log2(N))-1. Row 0 is the original data, rows 1, 2, ...
    #     # m-1 are the IMFs from HIGH to LOW frequency, and row m is the
    #     # residual (over all trend).
    #     mp1 = len(rslt)
    #     numIMF = mp1 - 2
    #     mosaic = [[f'{i+1}' for i in range(numIMF)]] 
    #     mosaic.extend([['Corrected'] * numIMF] * 3)
    #     fighht, axs = plt.subplot_mosaic(
    #         mosaic, figsize=(9,3), 
    #         # sharey=True,
    #         constrained_layout=True)
    #     thax = axs['Corrected']
       
    #     win = fighht.canvas.manager.window
    #     sh3 = self.screen_height // 3 
    #     win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -30 to avoid taskbar
    #     win.show()
        
    #     IMFs = [None] * mp1
    #     states = np.zeros(mp1, dtype=bool)
        
    #     def onclick_update(event):
    #         nonlocal acc
    #         if (eax := event.inaxes) is None:
    #             return

    #         if eax is thax:
    #             if event.key is None or event.key != 'control':
    #                 return
    #             # DON"T DELETE THE FOLLOWING
    #             # disp = rslt[0] - rslt[states].sum(axis=0)
    #             # velo = np.gradient(disp) / self.dt
    #             # acce = np.gradient(velo) / self.dt
    #             # acc[:] = acce
    #             # self.calc_rs()
    #             # self.plot_current()
    #             # fighht.canvas.mpl_disconnect(chht)
    #             # plt.close(fighht)
    #             fighht.canvas.mpl_disconnect(chht)
    #             plt.close(fighht)
    #             return
            
    #         imfid = int(eax.get_title())
    #         if states[imfid]:
    #             states[imfid] = False
    #             IMFs[imfid].set_alpha(1.0)
    #             eax.title.set_alpha(1.0)
    #         else:
    #             states[imfid] = True
    #             IMFs[imfid].set_alpha(0.3)
    #             eax.title.set_alpha(0.3)
    #         th = rslt[0] - rslt[states].sum(axis=0)
    #         thcurve.set_ydata(th)
    #         thax.relim()
    #         thax.autoscale_view(True, True, True)
    #         fighht.canvas.draw_idle()
        
    #     firstax = axs['1']
    #     for label, ax in axs.items():
    #         if label.startswith('Corrected'):
    #             thcurve, = ax.plot(time, rslt[0])
    #             ax.plot(time, rslt[0])
    #             ax.axhline(0, 0, 1, color='gray', lw=1)
    #             ax.set_xlabel('Time (s)')
    #             # ax.set_title('Ctrl+click displacement plot below to accept')
    #             ax.set_title("Demo only; don't know how to pass IMF to acceleration yet\n"
    #                          "based on the ensemble empirical mode decomposition method")
    #         else:
    #             if ax is not firstax:
    #                 ax.sharey(firstax)
    #             imfid = int(label)
    #             c, = ax.plot(time, rslt[imfid])
    #             IMFs[imfid] = c
    #             ax.set_title(label)
    #             ax.set_axis_off()
                
    #     # print(rslt.shape)
    #     chht = fighht.canvas.mpl_connect('button_press_event', onclick_update)
    #     # plt.show(block=False)

    def a2vd(self, acc=None):
        "This is about the same as David Boore's Fortran implementation"
        if acc is None:
            acc = self.acc[self.thrange]
        self.vel = v = sci.cumulative_trapezoid(acc, dx=self.dt, initial=0.0)
        self.dis = d = sci.cumulative_trapezoid(v, dx=self.dt, initial=0.0)
        # print(acc[0], v[0], d[0])
        return self.vel, self.dis
    
    def baseline_polynomial_scipy_version(self, event=None):
        # this method is saved for reference only. A replacement is provided below based on numpy.polynomial
        # which is more efficient. THIS METHOD WAS USED FOR THE NED PAPER ON GWM.
        
        acc = self.acc[self.thrange]
        time = self.time[self.thrange] - self.time[self.numadd]
        vel, dis = self.a2vd()
        
        # def fit_func(x, a, b, c, d):
        #     # return a * x**3 + b * x**2 
        #     # return a * x**4 + b * x**3 + c * x**2
        #     return a * x**5 + b * x**4 + c * x**3 + d * x**2

        # def fit_func_deriv2(x, a, b, c, d):
        #     # return 6*a*x + 2*b
        #     # return 12*a*x**2 + 6*b*x + 2*c
        #     return 20*a*x**3 + 12*b*x**2 + 6*c*x + 2*d

        def fit_func(x, *params):
            f = np.zeros_like(x)
            xcum = x.copy() 
            for i, p in enumerate(reversed(params)):
                # o = i + 2
                xcum *= x
                f += p * xcum
            return f

        
        def fit_func_deriv2(x, *params):
            fd2 = np.zeros_like(x)
            xcum = np.ones_like(x)
            for i, p in enumerate(reversed(params)):
                # o = i
                c = (i + 1) * (i + 2)
                fd2 += c * p * xcum
                xcum *= x
            return fd2
        
        # several ugly local "globals"
        curve_cfit = curve_corrected = None
        cfit = delta = None
        orig_order = 3
        params = np.ones(orig_order - 1)
        def onslide(order):
            nonlocal cfit, params, time, dis, curve_cfit, curve_corrected, axs
            # p0 defines the order for curve_fit, Nice!
            params, pcov = sco.curve_fit(fit_func, time, dis, p0=np.ones(order-1))
            cfit = fit_func(time, *params)
            if curve_cfit is None:
                curve_cfit, = axs.plot(time, cfit, color='red', lw=3, alpha=0.8)
                curve_corrected, = axs.plot(time, dis - cfit, linestyle='--')
            else:
                curve_cfit.set_ydata(cfit)
                curve_corrected.set_ydata(dis - cfit)
                axs.relim()
                axs.autoscale_view(True, True, True)
        
        fig = plt.figure(figsize=[6.4, 3])
        axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        axslide = plt.axes([0.3, 0.9, 0.6, 0.07], 
                           facecolor='lightblue'
                          )
        self.slider = Slider( #define slider properties 
                ax=axslide,
                label='Degree of Polynomial',
                valmin=2,
                valmax=12,
                valinit=orig_order,
                valstep=1,
                )
        self.slider.on_changed(onslide)
        
        def onclick_update(event):
            nonlocal acc, delta
            if event.inaxes is not axs:
                return
            # print('Key:', event.key)
            if event.key is None or event.key != 'control':
                return
            self.add_undo_stop()
            delta = fit_func_deriv2(time, *params)
            acc -= delta 
            self.calc_rs()
            self.plot_current()
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
    
        axs.plot(time, dis)
        axs.axhline(0, 0, 1, color='gray', lw=1)
        axs.set_title('Ctrl+click the displacement plot below to commit the correction')
        cid = fig.canvas.mpl_connect('button_press_event', onclick_update)
        onslide(orig_order)       
        # set fig location
        win = fig.canvas.manager.window
        sh3 = self.screen_height // 3 
        win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -20 to avoid taskbar
        win.show()

    def baseline_polynomial(self, event=None):
        'baseline_polynommial is now based on numpy.polynomial for efficiency'
        acc = self.acc[self.thrange]
        time = self.time[self.thrange] - self.time[self.numadd]
        T = self.dt * len(time)
        vel, dis = self.a2vd()
        
        # several ugly local "globals"
        curve_cfit = curve_corrected = None
        cfit = delta = None
        cur_order = 3
        w = np.ones_like(time)
        # reinforcing both ends
        imin = 5
        imax = len(w) - 5
        w[:imin] = np.linspace(1000.0, 1.0, 5)
        w[imax:] = np.linspace(1.0, 1000, 5) 
        p = None
        
        # ugly nonlocals can be avoided if this functionality is reimplemented in a seperate class
        def onslide(order):
            nonlocal p, cfit, time, dis, curve_cfit, curve_corrected, axs, w, cur_order
            # The lowest deg must be 2 so the 2nd order derivative dose not lose 
            # information when converting correction in displacement down to 
            # acceleration.
            cur_order = order
            deg = np.arange(2, order + 1) # skip the constant (order=0)
            p = Polynomial.fit(time, dis, deg=deg, domain=[0.0, T], window=[0.0, 1],
                               w=w) # no mapping, otherwise deriv would be messy
            # print(p)
            # print(p.domain, p.window)
            cfit = p(time)
            if curve_cfit is None:
                curve_cfit, = axs.plot(time, cfit, color='red', lw=3, alpha=0.8)
                curve_corrected, = axs.plot(time, dis - cfit, linestyle='--')
            else:
                curve_cfit.set_ydata(cfit)
                curve_corrected.set_ydata(dis - cfit)
                axs.relim()
                axs.autoscale_view(True, True, True)
        
        fig = plt.figure(figsize=[6.4, 3])
        # set fig location
        win = fig.canvas.manager.window
        sh3 = self.screen_height // 3 
        win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -20 to avoid taskbar
        
        axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        axs_wt = plt.axes([0.1, 0.12, 0.8, 0.07], sharex=axs, facecolor='ivory')
        # axs_wt.patch.set_edgecolor('ivory')
        axs_wt.get_xaxis().set_visible(False)
        axs_wt.get_yaxis().set_visible(False)
        for side in "top bottom left right".split():
            axs_wt.spines[side].set_visible(False)
        axs.set_zorder(axs_wt.get_zorder()+1)
        axs.patch.set_alpha(0.2)

        axslide = plt.axes([0.3, 0.9, 0.6, 0.07], 
                           facecolor='lightblue'
                          )
        fig._bl_poly_slider = Slider( #define slider properties 
                ax=axslide,
                label='Degree of Polynomial',
                valmin=2,
                valmax=20,
                valinit=cur_order,
                valstep=1,
                )
        fig._bl_poly_slider.on_changed(onslide)
        
        def onclick_update(event):
            nonlocal p, acc, delta
            if event.inaxes is not axs:
                return
            # print('Key:', event.key)
            if event.key is None or event.key != 'control':
                return
            self.add_undo_stop()
            pd2 = p.deriv(2)
            # print(pd2)
            # print(pd2.domain, pd2.window)
            delta = pd2(time) 
            acc -= delta 
            self.calc_rs()
            self.plot_current()
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
    
        axs.plot(time, dis)
        axs.axhline(0, 0, 1, color='gray', lw=1)
        fig.wt_line, = axs_wt.plot(time, w, color='k', ls='--', 
                                #    clip_box=TransformedBbox(Bbox([[0, 0], [1, 0.5]]), axs_wt.transAxes), 
                                #    clip_on=True,
                                   )
        axs_wt.set_ylim(-20, 100)
        # axs_wt.set_ylabel('Weights')
        axs.set_title('Ctrl+click displacement plot below to commit the cubic correction')
        cid = fig.canvas.mpl_connect('button_press_event', onclick_update)
        onslide(cur_order)   
        
        # SpanSelector for manually entering weights for a selected time domain
        def set_time_range(tmin, tmax):
            nonlocal imin, imax
            imin = round(tmin / self.dt)
            imax = round(tmax / self.dt)
            # ensure not overwriting at ends
            imin = max(5, imin)
            imax = min(len(w) - 5, imax)
            # print(imin, imax)
            
        def set_weights_in_range(event):
            nonlocal w, imin, imax
            tmin = imin * self.dt
            tmax = imax * self.dt
            print(imin, imax)
            avg_weight = np.average(w[imin:imax])
            new_weight, ok = QInputDialog.getDouble(
                win,
                f'New Weight in Range {tmin:.4f}s-{tmax:.4f}s', 
                f'Enter new weight:',
                avg_weight,
                1.0,
                50.0,
                3,
            )
            if ok:
                w[imin:imax] = new_weight
                print('New Weight', new_weight)
                onslide(cur_order)
                fig.wt_line.set_ydata(w)
                fig.canvas.draw_idle() 
                 
        # self.x is required to keep the widgets work
        fig._bl_poly_weight_span = LockableSpanSelector( # SpanSelector( #LockableSpanSelector(
              axs, set_time_range,
              'horizontal',
              useblit=True,
              props=dict(alpha=0.2, facecolor='tab:gray'),
            #   span_stays=True, # deprecated 
              interactive=True,
            #   drag_from_anywhere=True, # does not work well with log scale
        )      
        #  axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        ax_button_new_weight = fig.add_axes([0.45, 0, .1, .05])
        fig._bl_poly_button_new_weight = Button(ax_button_new_weight, 'New Weight')
        fig.cid_new_weight = fig._bl_poly_button_new_weight.on_clicked(set_weights_in_range)

        fig.canvas.draw()
        win.show()


    def baseline_spline(self, event=None):
        from scipy.interpolate import (CubicSpline, UnivariateSpline,
                                       make_interp_spline)
        acc = self.acc[self.thrange]
        time = self.time[self.thrange] - self.time[self.numadd]
        vel, dis = self.a2vd()
        
        disstd = dis.std()
        disstd2 = disstd * disstd
        n = len(acc)
        k = 3 # 5th degree spline
        # s = 0.5 * (n - np.sqrt(2*n)) # smoothing factor
        # smin = 0.0 #disstd2 * (n - np.sqrt(2*n)) # smoothing factor
        smax = disstd2 * (n - np.sqrt(2*n))
        s = 0.5 * smax
        # sstep = (smax - smin) / 40
        # x = time
        # y = dis
        # w = np.ones_like(acc)
        # w[0] = 1e3 # so the spline will pass the first and last points
        # w[-1] = 1e3
        # reinforcing both ends
        w = np.ones_like(time)
        imin = 5
        imax = len(w) - 5
        w[:imin] = np.linspace(1000.0, 1.0, 5)
        w[imax:] = np.linspace(1.0, 1000, 5) 

        # First, a smooth represenation
        spline_us = UnivariateSpline(time, dis, w=w, k=k, s=s)
        
        # estiamte a better std after removing the trend
        stdp2 = np.std(dis - spline_us(time))**2
        smin = 0.0
        smax = 5 * stdp2 * (n - np.sqrt(2*n))
        s = 0.5 * smax
        sstep = (smax - smin) / 200
        # update spline with new s
        spline_us = UnivariateSpline(time, dis, w=w, k=k, s=s)
        # Second, use a cubicsplie to create a clamped condition at both ends
        # so that Velocity will be at rest at time 0, and end
        numknots = 77
        xc = np.linspace(time[0], time[-1], num=numknots, endpoint=True)
        yc = spline_us(xc)
        # self.spline = spline_0 = make_interp_spline(xc, yc, k=3, bc_type='clamped')
        # self.spline = spline_0 = CubicSpline(xc, yc, bc_type='clamped')
        spline_0 = CubicSpline(xc, yc, bc_type='clamped')

        def onslide_smooth(s_slider):
            # print('onslide_smooth')
            nonlocal spline_us, spline_0, numknots, cur_spl, cur_diff_spl, s, w
            s = s_slider
            # p0 defines the order for curve_fit, Nice!
            # spline_us = UnivariateSpline(time, dis, w=w, k=k, s=s)
            spline_us = UnivariateSpline(time, dis, w=w, k=3, s=s)
            sspl = spline_us(time)
            cur_smooth.set_ydata(sspl)
            onslide_numknots(numknots)
            
        def onslide_numknots(nknots):
            # print('onslide_numknots')
            nonlocal numknots, spline_0, spline_us, cur_spl, cur_diff_spl
            numknots = nknots
            xc = np.linspace(time[0], time[-1], num=numknots, endpoint=True)
            yc = spline_us(xc)
            # spline_0 = make_interp_spline(xc, yc, k=3, bc_type='clamped')
            spline_0 = CubicSpline(xc, yc, bc_type='clamped')
            dspl = spline_0(time)
            cur_spl.set_ydata(dspl)
            cur_diff_spl.set_ydata(dis - dspl)
            axs.relim()
            axs.autoscale_view(True, True, True)
    
        fig = plt.figure(figsize=[6.4, 3])
        win = fig.canvas.manager.window
        sh3 = self.screen_height // 3 
        win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -20 to avoid taskbar
        
        axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        axs_wt = plt.axes([0.1, 0.12, 0.8, 0.07], sharex=axs, facecolor='ivory')
        # axs_wt.patch.set_edgecolor('ivory')
        axs_wt.get_xaxis().set_visible(False)
        axs_wt.get_yaxis().set_visible(False)
        for side in "top bottom left right".split():
            axs_wt.spines[side].set_visible(False)
        axs.set_zorder(axs_wt.get_zorder()+1)
        axs.patch.set_alpha(0.2)
        
        axsmooth = plt.axes([0.1, 0.9, 0.3, 0.07], 
                           facecolor='lightblue'
                          )
        axknots = plt.axes([0.6, 0.9, 0.3, 0.07], 
                           facecolor='lightblue'
                          )
        slider_smooth = Slider( #define slider properties 
                ax=axsmooth,
                label='Smooth',
                valmin=smin,
                valmax=smax,
                valinit=s,
                valstep=sstep,
                )
        # cid_smooth needs to be saved to function
        cid_smooth = slider_smooth.on_changed(onslide_smooth)
        
        slider_knots = Slider( #define slider properties 
                ax=axknots,
                label='#Knots',
                valmin=3,
                valmax=151,
                valinit=77,
                valstep=2, # avoid even number of knots
                )
        # cid_knots needs to be saved to function
        cid_knots = slider_knots.on_changed(onslide_numknots)
        
        def onclick_update(event):
            # print('onclick')
            nonlocal acc, spline_0, slider_knots, slider_smooth
            if event.inaxes is axs:
                if event.key and event.key == 'control':
                    self.add_undo_stop()
                    spline_2 = spline_0.derivative(nu=2) 
                    delta = spline_2(time)
                    acc -= delta
                    self.calc_rs()
                    self.plot_current()
                    slider_smooth.disconnect(cid_smooth)
                    slider_knots.disconnect(cid_knots)
                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)
                return
        
        sspl = spline_us(time)
        dspl = spline_0(time)
        axs.plot(time, dis, color='tab:blue')
        cur_smooth, = axs.plot(time, sspl, linewidth=3, color='red',
                               alpha=0.8)
        cur_spl, = axs.plot(time, dspl, linewidth=3, color='red',
                            alpha=0.8, linestyle='--')
        cur_diff_spl, = axs.plot(time, dis - dspl, color='tab:blue',
                                 linestyle='--')
        axs.axhline(0, 0, 1, color='gray', lw=1)
        axs.set_ylabel('Displacement')
        axs.set_title('Ctrl+click displacement plot below to commit the correction')
        cid = fig.canvas.mpl_connect('button_press_event', onclick_update)
        
        # handle weights
        fig.wt_line, = axs_wt.plot(time, w, color='k', ls='--', 
                                #    clip_box=TransformedBbox(Bbox([[0, 0], [1, 0.5]]), axs_wt.transAxes), 
                                #    clip_on=True,
                                   )
        axs_wt.set_ylim(-20, 100)
        
        # SpanSelector for manually entering weights for a selected time domain
        def set_time_range(tmin, tmax):
            nonlocal imin, imax
            imin = round(tmin / self.dt)
            imax = round(tmax / self.dt)
            # ensure not overwriting at ends
            imin = max(5, imin)
            imax = min(len(w) - 5, imax)
            # print(imin, imax)
            
        def set_weights_in_range(event):
            nonlocal w, imin, imax
            tmin = imin * self.dt
            tmax = imax * self.dt
            print(imin, imax)
            avg_weight = np.average(w[imin:imax])
            new_weight, ok = QInputDialog.getDouble(
                win,
                f'New Weight in Range {tmin:.4f}s-{tmax:.4f}s', 
                f'Enter new weight:',
                avg_weight,
                1.0,
                50.0,
                3,
            )
            if ok:
                w[imin:imax] = new_weight
                print('New Weight', new_weight)
                onslide_smooth(s)
                fig.wt_line.set_ydata(w)
                fig.canvas.draw_idle() 
                 
        # self.x is required to keep the widgets work
        fig._bl_poly_weight_span = LockableSpanSelector( # SpanSelector( #LockableSpanSelector(
              axs, set_time_range,
              'horizontal',
              useblit=True,
              props=dict(alpha=0.2, facecolor='tab:gray'),
            #   span_stays=True, # deprecated 
              interactive=True,
            #   drag_from_anywhere=True, # does not work well with log scale
        )      
        #  axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        ax_button_new_weight = fig.add_axes([0.45, 0, .1, .05])
        fig._bl_poly_button_new_weight = Button(ax_button_new_weight, 'New Weight')
        fig.cid_new_weight = fig._bl_poly_button_new_weight.on_clicked(set_weights_in_range)

        fig.canvas.draw()
        
        win.show()

    
    def remove_average_acc(self, event=None):
        # num1sec = int(np.round(1/self.dt))
        # num_keep = self.numAcc - self.numadd - num1sec
        # env = np.r_[np.zeros(self.numadd), 
        #             np.linspace(0.0, 1.0, num1sec),
        #             np.ones(num_keep)
        #             ]   
        # self.acc *= env # env will change acc a lot for the 1 sec ramp period
        # detrend from self.numadd to the end, this op is not useful and generally
        # may not be valid becuase long period motion will be removed.
        self.add_undo_stop()
        acc = self.acc[self.thrange]
        # self.trend = np.linspace(acc[0], acc[-1], len(acc), endpoint=True)
        # acc -= self.trend
        acc -= acc.mean() # remove a constant avg should help
        # padding zeros for below self.numadd
        self.acc[:self.numadd] = 0.0
        self.calc_rs()
        self.plot_current()

    def detrend_velocity(self, event=None):
        from scipy.interpolate import CubicSpline, UnivariateSpline
        acc = self.acc[self.thrange]
        time = self.time[self.thrange] - self.time[self.numadd]
        vel, dis = self.a2vd()
        
        velstd = vel.std()
        velstd2 = velstd * velstd
        n = len(acc)
        k = 3 # 5th degree spline
        smax = velstd2 * (n - np.sqrt(2*n))
        s = 0.5 * smax
        # w = np.ones_like(acc)
        # w[0] = 1e3 # so the spline will pass the first and last points
        # w[-1] = 1e3
        # reinforcing both ends
        w = np.ones_like(time)
        w[:5] = np.linspace(1000.0, 1.0, 5)
        w[-5:] = np.linspace(1.0, 1000, 5)  
        # First, a smooth represenation
        spline_us = UnivariateSpline(time, vel, w=w, k=k, s=s)
        
        # estiamte a better std after removing the trend
        stdp2 = np.std(vel - spline_us(time))**2
        smin = 0.0
        smax = 5 * stdp2 * (n - np.sqrt(2*n))
        s = 0.5 * smax
        sstep = (smax - smin) / 200
        # update spline with new s
        spline_us = UnivariateSpline(time, vel, w=w, k=k, s=s)
        # Second, use a cubicsplie to create a clamped condition at both ends
        # so that Velocity will be at rest at time 0, and end
        numknots = 77
        xc = np.linspace(time[0], time[-1], num=numknots, endpoint=True)
        yc = spline_us(xc)
        spline_0 = CubicSpline(xc, yc, bc_type='clamped')

        def onslide_smooth(s):
            # print('onslide_smooth')
            nonlocal spline_us, spline_0, numknots, cur_spl, cur_diff_spl
            # p0 defines the order for curve_fit, Nice!
            spline_us = UnivariateSpline(time, vel, w=w, k=k, s=s)
            sspl = spline_us(time)
            cur_smooth.set_ydata(sspl)
            onslide_numknots(numknots)
            
        def onslide_numknots(nknots):
            # print('onslide_numknots')
            nonlocal numknots, spline_us, spline_0
            numknots = nknots
            xc = np.linspace(time[0], time[-1], num=numknots, endpoint=True)
            yc = spline_us(xc)
            spline_0 = CubicSpline(xc, yc, bc_type='clamped')
            dspl = spline_0(time)
            cur_spl.set_ydata(dspl)
            cur_diff_spl.set_ydata(vel - dspl)
            axs.relim()
            axs.autoscale_view(True, True, True)
    
        fig = plt.figure(figsize=[6.4, 3])
        axs = plt.axes([0.1, 0.1, 0.8, 0.7])
        axsmooth = plt.axes([0.1, 0.9, 0.3, 0.07], 
                           facecolor='lightblue'
                          )
        axknots = plt.axes([0.6, 0.9, 0.3, 0.07], 
                           facecolor='lightblue'
                          )
        slider_smooth = Slider( #define slider properties 
                ax=axsmooth,
                label='Smooth',
                valmin=smin,
                valmax=smax,
                valinit=s,
                valstep=sstep,
                )
        # cid_smooth needs to be saved to function
        cid_smooth = slider_smooth.on_changed(onslide_smooth)
        
        slider_knots = Slider( #define slider properties 
                ax=axknots,
                label='#Knots',
                valmin=3,
                valmax=151,
                valinit=77,
                valstep=2, # avoid even number of knots
                )
        # cid_knots needs to be saved to function
        cid_knots = slider_knots.on_changed(onslide_numknots)
        
        def onclick_update(event):
            # print('onclick')
            nonlocal acc, spline_0, slider_knots, slider_smooth
            if event.inaxes is axs:
                if event.key and event.key == 'control':
                    self.add_undo_stop()
                    spline_1 = spline_0.derivative(nu=1) 
                    delta = spline_1(time)
                    acc -= delta
                    self.calc_rs()
                    self.plot_current()
                    slider_smooth.disconnect(cid_smooth)
                    slider_knots.disconnect(cid_knots)
                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)
                return
        
        sspl = spline_us(time)
        dspl = spline_0(time)
        axs.plot(time, vel, color='tab:blue')
        cur_smooth, = axs.plot(time, sspl, linewidth=3, color='red',
                               alpha=0.8)
        cur_spl, = axs.plot(time, dspl, linewidth=3, color='red',
                            alpha=0.8, linestyle='--')
        cur_diff_spl, = axs.plot(time, vel - dspl, color='tab:blue',
                                 linestyle='--')
        axs.axhline(0, 0, 1, color='gray', lw=1)
        axs.set_ylabel('Velocity')
        axs.set_xlabel('Time (s)')
        axs.set_title('Ctrl+click velocity plot below to commit the correction')
        cid = fig.canvas.mpl_connect('button_press_event', onclick_update)

        win = fig.canvas.manager.window
        sh3 = self.screen_height // 3 
        win.setGeometry(0, sh3, self.screen_width//2, sh3 * 2 - 31) # -20 to avoid taskbar
        win.show()
            
    def filter(self):
        # fmin = self.tfreq[0]
        # try:
        span_status = self.span.visible
        self.span.set_visible(False)
        self.fig.canvas.draw_idle()
        self.cursor_filter = Cursor(self.axrs, horizOn=True, 
                        vertOn=True, 
                        useblit=True, 
                        color='black',
                        linestyle='--',
                        )
        # (freq, _), = ginput_mouseonly(self.fig)
        self.cursor_filter.connect_event('button_press_event', self.onclick_filter)
        self.span.set_visible(span_status)
        

    def onclick_filter(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #         event.x, event.y, event.xdata, event.ydata))
        freq = event.xdata

        self.filter_at_freq(freq)
        self.imax = None
        self.plot_current()
        if hasattr(self, 'cursor_filter'):
            self.cursor_filter.disconnect_events()
            self.cursor_filter.linev.remove()
            self.cursor_filter.lineh.remove()
            # self.cursor_filter.clear(None)
            del self.cursor_filter
        
    def filter_at_freq(self, freq):
        ifreq = self.tfreq.searchsorted(freq)
        fmax = self.tfreq[ifreq]
        self._filter(0.0, fmax)
        self.calc_rs()
    
    def _filter(self, fmin, fmax):
        self.add_undo_stop()
        fc = em.fft(self.acc, padding=True)
        df = 1.0 / (self.dt * self.numAcc)
        filtered = eu.butterworth(fc, df, fmin, fmax)
        accfiltered = em.ifft(filtered).real
        self.acc = accfiltered[:self.numAcc]  # fft & ifft added a point
        
    def plot_avd(self):
        acc = self.acc[self.thrange]
        time = self.time[self.thrange]

        def close_avd_plot(event):
            if self.avd_plot:
                self.avd_plot.canvas.mpl_disconnect(cid_avd_plot)
                self.avd_plot = None
            
        if self.avd_plot is None:
            self.avd_plot, (self.avdp_acc, self.avdp_vel, 
                            self.avdp_dis) = plt.subplots(3 ,1, sharex=True)
            cid_avd_plot = self.avd_plot.canvas.mpl_connect(
                'close_event', close_avd_plot)
            avdwin = self.avd_plot.canvas.manager.window
            avdwin.move(self.window.x() + self.window.width()+1, 
                        0)
                        # self.window.y()) # 97 for control window
            self.avdp_acc.axhline(0, 0, 1, color='gray', lw=1)
            self.avdp_vel.axhline(0, 0, 1, color='gray', lw=1)
            self.avdp_dis.axhline(0, 0, 1, color='gray', lw=1)
            self.avdp_acc.set_ylabel('Acceleration')
            self.avdp_vel.set_ylabel('Velocity')
            self.avdp_dis.set_ylabel('Displacement')
            self.avdp_dis.set_xlabel('Time (s) [Black dashed lines represent the seed]')   
            seed = self.seed[self.thrange]
            vseed, dseed = self.a2vd(seed)
            self.avdp_acc.plot(time, seed, color='black', 
                               linestyle='--', alpha=0.8)
            self.avdp_vel.plot(time, vseed, color='black', 
                               linestyle='--', alpha=0.8)
            self.avdp_dis.plot(time, dseed, color='black', 
                               linestyle='--', alpha=0.8)
        # else:
            # self.avd_plot.canvas.manager.window.show()
        
        # v, d = eb.avd(acc, self.dt)
        # self.vel = v
        # self.dis = d
        v, d = self.a2vd()
        self.avdp_acc.plot(time, acc, alpha=0.8)
        self.avdp_vel.plot(time, v, alpha=0.8)
        self.avdp_dis.plot(time, d, alpha=0.8)
        
        for nam in ('acc', 'vel', 'dis'):
            ax = getattr(self, f'avdp_{nam}')
            ax.relim()
            ax.autoscale_view(True, True, True)
        self.avd_plot.align_ylabels([self.avdp_acc, self.avdp_vel, self.avdp_dis])
        self.avd_plot.suptitle('Acceleration, Velocity, and Displacement Histories')
        self.avd_plot.tight_layout()
        self.avd_plot.canvas.draw_idle()
        plt.show(block=False)
        # self.avd_plot.canvas.manager.window.show()
    
    def plot_Fourier_spectra(self):
        def close_fs_plot(event):
            if self.fourier_spectra_plot:
                self.fourier_spectra_plot.canvas.mpl_disconnect(cid_fs_plot)
                self.fourier_spectra_plot = None
            
        if self.fourier_spectra_plot is None:
            self.fourier_spectra_plot, (axfas, axfps) = plt.subplots(
                2, 1, height_ratios=[3,1],
                sharex=True,
                figsize=(6.4, 7.4))
            fspwin = self.fourier_spectra_plot.canvas.manager.window
            fspwin.move(self.window.x() + self.window.width()+1, 
                        0)
            cid_fs_plot = self.fourier_spectra_plot.canvas.mpl_connect(
                'close_event', close_fs_plot)
            self.fourier_spectra_plot.suptitle('Fourier Ampllitude and Phase Spectra')
            axfas.set_ylabel('FAS (g-sec)')   
            axfas.grid(linestyle='-', linewidth=0.5)
            axfps.set_xlabel('Frequency (Hz)')   
            axfps.set_ylabel('FPS (Radians)')
            axfps.yaxis.set_major_formatter(tck.FuncFormatter(
                            lambda val, pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
                    ))
            axfps.yaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
            axfps.grid(linestyle='-', linewidth=0.5)
            acc = self.acc[self.thrange]
            time = self.time[self.thrange]
            seed = self.seed[self.thrange]
            dt = self.dt
            Nt = len(time)
            Fourier_freq = em.freq_Fourier_transform(dt, Nt)
            acc = em.Accelerogram(dt, acc)
            seed = em.Accelerogram(dt, seed)
            fc_acc = acc.fft()
            fc_seed = seed.fft()
            fas_seed = np.abs(fc_seed)
            sm_fas_seed = em.rsmoothen_perc(fas_seed, .40)
            fas_acc = np.abs(fc_acc)
            sm_fas_acc = em.rsmoothen_perc(fas_acc, .40)
            axfas.loglog(Fourier_freq, fas_seed, color='black', 
                               linestyle='--', lw=0.5, alpha=0.5, label='Seed')
            axfas.loglog(Fourier_freq, sm_fas_seed, color='black', linestyle='--', lw=2,
                         alpha=0.9, label='Seed Smoothed')
            axfas.fill_between(Fourier_freq, sm_fas_seed, 0.84*sm_fas_seed, 
                        color='black', alpha=0.3, label=r'Seed Smoothed*84%'
                        )
            axfas.loglog(Fourier_freq, np.abs(fc_acc), color='C0', lw=0.5, alpha=0.5, label='Modified')
            axfas.loglog(Fourier_freq, sm_fas_acc, color='C0', lw=2,
                         alpha=0.9, label='Modified Smoothed')
            axfas.fill_between(Fourier_freq, sm_fas_acc, 0.84*sm_fas_acc, color='C0',
                         alpha=0.3, label='Modified Smoothed*84%')
            axfas.legend(loc='best')
            axfps.semilogx(Fourier_freq, np.angle(fc_seed), color='black', 
                               linestyle='--', alpha=0.8, label='Seed')
            axfps.semilogx(Fourier_freq, np.angle(fc_acc), alpha=0.8, label='Modified')
            
            # plot target PSD based FAS if available
            if self.targetpsd is not None:
                # Nt = self.numAcc
                # dt = self.dt
                minpsd_ratio = self.minpsd_ratio
                # self.freq_psd, self.fas_psd = psd2fas_fft(self.psdfreq, self.targetpsd, 
                #                                 Nt, dt, interp='loglog', ends=None)
                fas_psd = 0.45 * self.fas_psd # to approximate the effect of strong motion duration. Industry factor for SMD: 0.7 * T
                axfas.loglog(self.freq_psd, fas_psd, color='tab:red', lw=3, alpha=0.8, label='From Target PSD')
                axfas.loglog(self.freq_psd, minpsd_ratio*fas_psd, color='tab:red', lw=3, linestyle='--',
                             alpha=0.8, label=f'{self.minpsd_ratio:.1f} x Target PSD')
            self.fourier_spectra_plot.align_ylabels([axfas, axfps])
            self.fourier_spectra_plot.tight_layout()
            self.fourier_spectra_plot.canvas.draw_idle()
            plt.show(block=False)
            # for nam in ('acc', 'vel', 'dis'):
            #     ax = getattr(self, f'avdp_{nam}')
            #     ax.relim()
            #     ax.autoscale_view(True, True, True)
            # self.avd_plot.tight_layout()
            # self.avd_plot.canvas.draw_idle()
            # plt.show(block=False)
        
    def enrich(self, noise=0.1):
        from random import gauss
        self.add_undo_stop()
        acc = self.acc[self.thrange]
        noise = np.array([gauss(0.0, noise*abs(a)) for a in acc])
        # print(noise, noise.max(), noise.min())
        acc += noise
        self.calc_rs()
        self.imax = None # to avoid plot frequency indicator
        self.plot_current()
        # self.match_1() # to avoid errors
    
    def psd2fas(self):
        'replace FAS based on the target PSD'
        # self.freq_psd, self.fas_psd =  
        # acc = em.Accelerogram(dt, self.acc)
        # fc_acc = acc.fft()
        # print(self.acc.dtype) # float64
        # self.orig_seed = seed
        # self.orig_dt = dt
        # avoid using the zero-prepadded time history in replacing FAS
        acc = self.acc[self.thrange].copy()
        Nt = len(acc)
        dt = self.dt
        fc_acc = np.fft.rfft(acc)
        
        # Nt = self.numAcc
        # dt = self.dt
        # fc_acc = np.fft.rfft(self.acc)
        # print(fc_acc.dtype) # complex128
        phi = np.angle(fc_acc)
        # print(phi.dtype) # float64
        fc1 = np.exp(1j*phi) # unit FC
        # print(fc1.dtype) # complex128
        freq_psd, fas_psd = psd2fas_fft(self.psdfreq, self.targetpsd, Nt, dt, interp='loglog', ends=None)
        fcmag = fas_psd
        # fcmag = self.fas_psd
        # print(fcmag.dtype) # float64
        acc = np.fft.irfft(fc1*fcmag, Nt) # use Nt to force to the same lenght as the orignal acc
        # print(acc.dtype) # float64
        # scale to average SA
        # reconstruct the prepadded acc
        acc_prepadded = np.r_[np.zeros(self.numadd), acc] # zero prepadding
        _, _, sa, _, _, _  = rst_openmp(acc_prepadded, dt, [self.damping], self.tfreq)
        rs = sa[0]
        # print(rs.dtype) #float32
        ratios = self.trs / rs
        factor = statistics.geometric_mean(ratios)
        self.acc = factor * acc_prepadded
        # print('PSD2FAS DEBUG')
        # print(len(self.time), len(self.seed))
        # print(len(self.time), len(self.acc))
        # print(len(self.orig_seed), self.numadd)
        self.calc_rs()
        self.imax = None # to avoid plot frequency indicator
        self.plot_current()
            
    def dump(self, basefilename):
        if basefilename is None:
            return

        if not basefilename.startswith('GWM'):
            basefilename = 'GWM+' + basefilename
            
        basefile = f'{self.results_dir}/{basefilename}'
        pngfile = basefile + '_rs_acc.png'
        avdpngfile = basefile + '_avd.png'
        accfile = basefile + '.acc'
        velfile = basefile + '.vel'
        disfile = basefile + '.dis'
        vel, dis = self.a2vd() # probably already availabe, but to make sure consistency between acc, vel, and dis
        
        header = f'GWM: {timestamp()} {self.accname}, matched with damping={self.damping:.1%}'
        with open(accfile, 'w') as fh:
            print(header, file=fh)
            print("DT=%.4f" % self.dt, file=fh)
            for a in self.acc[self.thrange]:
                print("%.6G" % a, file=fh)
            print('Saved:', accfile)
                
        with open(velfile, 'w') as fh:
            print(header, file=fh)
            print("DT=%.4f" % self.dt, file=fh)
            for v in vel:
                print("%.6G" % v, file=fh)
            print('Saved:', velfile)
                
        with open(disfile, 'w') as fh:
            print(header, file=fh)
            print("DT=%.4f" % self.dt, file=fh)
            for d in dis:
                print("%.6G" % d, file=fh)
            print('Saved:', disfile)
                                
        self.plot_current(to_draw=False)
        self.clean_canvas(force_clean=True)
        # self.fig.savefig(pngfile)
        savefig_reduced_png(self.fig, pngfile)
        print('Saved:', pngfile)
        savefig_reduced_png(self.avd_plot, avdpngfile)
        print('Saved:', avdpngfile)
        if self.fourier_spectra_plot:
            fspngfile = basefile + '_fas_fps.png'
            savefig_reduced_png(self.fourier_spectra_plot, fspngfile)
            print('Saved:', fspngfile)
        if self.check_o1a2_plot:
            o1a2pngfile = basefile + '_check_o1a2.png' 
            savefig_reduced_png(self.check_o1a2_plot, o1a2pngfile)
            print('Saved:', o1a2pngfile)
        return self


    