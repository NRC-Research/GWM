# -*- coding: utf-8 -*-
""" Modeling codes to support GWM

Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
import numpy as np
from ._equtils import rsmoothen_perc, rsmoothen, cavstd, splitcosinebell
from . import _baseline as bl


def freq_SRP371_Option1_Approach1(cutoff_freq):
    '''retrn the list of frequencies for Response Spectrum check, per
    SRP 3.7.1, Opption 1, Approach 1, Table 3.7.1-1

    75 frequency points up to 33 Hz

    frequence points and frequency range are significant because the
    criteria has a reqruiement of 5 points in total below target.
    '''
    f = np.r_[ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
         1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
         1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5,
         2.6, 2.7, 2.8, 2.9, 3., 3.15, 3.3, 3.45,
         3.6, 3.8, 4., 4.2, 4.4, 4.6, 4.8,
         5., 5.25, 5.5, 5.75, 6., 6.25, 6.5, 6.75,
         7., 7.25, 7.5, 7.75, 8., 8.5, 9., 9.5,
        10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5,
        14., 14.5, 15., 16., 17., 18., 20.,
        22.0:cutoff_freq:3.0,
        cutoff_freq]
    return f


def freq_SRP371_Option1_Approach2(npoints=301):
    '''return the frequencies for Response Spectrum calculation, per
    SRP 3.7.1 and RG 1.208 App F: at least 100 points per frequency
    decade. '''
    f = np.logspace(-1, 2, npoints)
    #~ f = np.linspace(-1.0, 2.0, 301)
    #~ f = 10**f
    return f

def freq_Fourier_transform(dt=0.005, Nt=4096):
    Nf = Nt//2 + 1
    df = 1/Nt/dt
    return df*np.arange(Nf)

def loglog_interp(x, xp, yp, left=None, right=None):
    'return the linear interpolation of yp using x on the loglog scale'
    #~ if left is None: # set to a very small number
        #~ left = -100
    #~ if right is None:
        #~ right = -100
    # left=right=None default to take the end points
    tmp = 10**np.interp(np.log10(x), np.log10(xp), np.log10(yp),
                    left=left, right=right)
    #~ print tmp[0], yp[0], tmp[-1], yp[-1]
    return tmp

def arias_intensity(acc):
    '''calculate the Arias Intensity, normalized to unity at the end of
    the time history and return the strong motion duration pair
    (t5, t75)

    return t5, t75, IA'''
    a2 = np.cumsum(acc*acc)
    a2 /= a2[-1]  # normalize to unity
    i5, i75, i95 = np.searchsorted(a2, [0.05, 0.75, 0.95])
    return i5, i75, i95, a2

def fft(th, padding=True):
    '''calculate the Fourier spectra of a time history (e.g. acc, vel,
    dis)

    th include a member dt and data

    return the complex Fourier Spectrum up to the Nyquist Frequency

    ---- Theory Background -------------------------------------------
    the correct Discrete Fourier transform is fc(i)*dt
    [Bendat and Piersol, 1986], eq 11.100, pg 392
        Xi (fk) = dt Xik = dt {SUM(n=0, N-1) {xin exp(-i 2 pi k n / N)} }
        [Theory] X = {INT(0,T) x(t) exp(-i 2 pi f t) dt}

    *** PCARES version
        [Theory] X = 1/T {INT(0,T) x(t) exp(-i 2 pi f t) dt}
        ==> fc(i) * dt / T is comparable to PCARES fft result
           i.e. fc(i) / N

    *** FROM WWW.NETLIB.ORG/FFTPACK/DOC
    subroutine cfftf computes the forward complex discrete fourier
    transform (the fourier analysis). equivalently , cfftf computes
    the fourier coefficients of a complex periodic sequence.
    the transform is defined below at output parameter c.

    the transform is not normalized. to obtain a normalized transform
    the output must be divided by n. otherwise a call of cfftf
    followed by a call of cfftb will multiply the sequence by n.

    the array wsave which is used by subroutine cfftf must be
    initialized by calling subroutine cffti(n,wsave).

    input parameters


    n      the length of the complex sequence c. the method is
           more efficient when n is the product of small primes. n

    c      a complex array of length n which contains the sequence

    wsave   a real work array which must be dimensioned at least 4n+15
            in the program that calls cfftf. the wsave array must be
            initialized by calling subroutine cffti(n,wsave) and a
            different wsave array must be used for each different
            value of n. this initialization does not have to be
            repeated so long as n remains unchanged thus subsequent
            transforms can be obtained faster than the first.
            the same wsave array can be used by cfftf and cfftb.

    output parameters

    c      for j=1,...,n

               c(j)=the sum from k=1,...,n of

                     c(k)*exp(-i*(j-1)*(k-1)*2*pi/n)

                           where i=sqrt(-1)

    wsave   contains initialization calculations which must not be
            destroyed between calls of subroutine cfftf or cfftb

    The frequencies of the result are [ 0, 1, 2, 3, 4, -3, -2, -1]

    The use of rfft comput only upto [ 0, 1, 2, 3, 4]
    '''
    if padding:
        power2len = next_power2(th.size)
        #~ power2len = closest_power2(th.size)
        #~nNyquist = power2len/2+1
        #~fc = np.fft.fft(th.data, power2len)
        fc = np.fft.rfft(th, power2len)  # the same as Excel fft, JRN20130703
    else:
        fc = np.fft.rfft(th)
    return fc

def next_power2(n):
    '''return the next integer of 2**power after n'''
    power2len = 2**int(np.ceil(np.log(n)/np.log(2)))
    return power2len

def ifft(fc):
    return np.fft.irfft(fc)

# ------------------------------------------------------------------------------
class TimeHistory(np.ndarray):
    '''common functions for time history'''
    __abr_name = 'TH'
    def __new__(cls, dt, data, unit='m/s**2', name='', note=''):
        ''' dt: the time increment,
        data: array or list of seismic record data'''
        th = np.asarray(data, np.float32).view(cls) # make a new copy
        th.dt = dt
        th.unit = unit
        th.name = name
        th.note = note
        return th

    def __array_finalize__(self, th):
        if th is None: 
            return
        self.dt = getattr(th, 'dt', None)
        self.unit = getattr(th, 'unit', None)
        self.name = getattr(th, 'name', None)
        self.note = getattr(th, 'note', None)

    def __reduce__(self): # for pickling
        # Get the parent's __reduce__ tuple
        pickled_state = super(TimeHistory, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.dt, self.unit, self.name, self.note)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):  # for unpickling
        self.dt, self.unit, self.name, self.note = state[-4:]
        # Call the parent's __setstate__ with the other tuple elements.
        super(TimeHistory, self).__setstate__(state[0:-4])

    def __repr__(self):
        ret = np.ndarray.__repr__(self)
        return ret.replace('([', '''(
        name: %(name)s
        dt: %(dt)f
        unit: %(unit)s
        data: [''' % self.__dict__, 1)

    def __str__(self):
        ret = np.ndarray.__str__(self)
        return '''name: %(name)s
        dt: %(dt)f
        unit: %(unit)s
        data: ''' % self.__dict__ + ret

    @property  # readonly property
    def rawdata(self):
        return self.view(np.ndarray)  # a view of the data

    @property
    def N(self):
        return self.size    #self.size = n. if n is of a power of 2,
                            # n = 2(m-1), where m is the length of FC

    @property
    def T(self): return self.dt * self.N

    @property
    def time(self):
        return self.dt*np.arange(self.N, dtype=np.float32)

    @property
    def peak(self):
        argpga = np.argmax(np.abs(self))
        pga = np.abs(self[argpga])
        return self.dt * argpga, pga

    @property
    def NyquistFreq(self):
        return 1.0/2/self.dt

    def saveData(self, fname):
        with open(fname, 'w') as fh:
            print("Time (sec)\tData(%s)" % self.unit, file=fh)
            for i, a in enumerate(self):
                print("%.4f\t%.6G" %(i*self.dt, a), file=fh)


    def saveData_1C(self, fname, header=''):
        with open(fname, 'w') as fh:
            print(header, file=fh)
            print("DT=%.4f" % self.dt, file=fh)
            for a in self:
                print("%.6G" % a, file=fh)


    def chop(self, left, right):
        c = self[left:right].copy()
        c.name = 'chopped-' + self.name
        return c

    def fft(self, padding=False):
        '''need to multiply by self.dt to make the self.fc as DFT!'''
        fc = fft(self.rawdata, padding) # * self.dt
        if self.unit.endswith('/s'):
            unit = self.unit[:-2]
        else:
            unit = self.unit + '*s'
        self.fc = FourierSpectrum(self.dt, fc, unit=unit,
            name='fc_'+self.name,
            th=self)
        return self.fc


# -------------------------------------------------------------------
def arias(func):
    def wrap_func(self):
        try:
            return func(self)
        except:
            self.i5, self.i75, self.i95, self.ai = \
                arias_intensity(self.rawdata)
            return func(self)
    return wrap_func


class Accelerogram(TimeHistory):
    __abr_name = 'ACC'

    def rs(self, dampings):
        freq, sa, sv, sd, fs = rs(self, self.dt, dampings)
        return freq, sa, sv, sd, fs

    @property
    @arias
    def AI(self): return self.ai

    def normalized_arias_intensity(self):
        return self.AI

    @property
    @arias
    def SMD(self): return (self.i75 - self.i5)*self.dt

    @property
    @arias
    def T5(self): return self.i5*self.dt

    @property
    @arias
    def T75(self): return self.i75*self.dt

    @property
    @arias
    def T95(self): return self.i95*self.dt

    @property
    def pga(self):
        argpga = np.argmax(np.abs(self))
        pga = np.abs(self[argpga])
        return self.dt * argpga, pga
    PGA = pga
    ZPA = pga

    def baseline_corrected(self):
        bl_acc = bl.baseline(self.rawdata.copy(), self.dt)
        return Accelerogram(self.dt, bl_acc,
                    unit=self.unit,
                    name='bl_'+self.name)


class Spectrum(np.ndarray):
    __abr_name = 'SPECTRUM'
    '''encapsulates a spectrum type sequence'''
    def __new__(cls, dt, sv, unit='m/s**2*s', name='', th=None):
        ''' dt: the time increment, use of dt to aviod numerical
            roundoff
            either dt or freq must be provided, dt overwrites freq
        sv: array or list of two-sided Fourier Components
        '''
        fc = np.asarray(sv, np.complex64).view(cls)    # 1 side, rfft
        fc.dt = dt
        fc.unit = unit
        fc.name = name
        fc.th = th  # a reference to original time history
        return fc

    def __array_finalize__(self, fc):
        if fc is None: return
        self.dt = getattr(fc, 'dt', None)
        self.unit = getattr(fc, 'unit', None)
        self.name = getattr(fc, 'name', None)
        self.th = getattr(fc, 'th', None)

    # __reduce__ and __setstate__ for pickle and multimpressing
    def __reduce__(self):
        pickled_state = super(Spectrum, self).__reduce__()
        new_state = pickled_state[2] + (self.dt, self.unit, self.name, self.th)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.dt, self.unit, self.name, self.th = state[-4:]
        super(Spectrum, self).__setstate__(state[0:-4])

    @property  # readonly property
    def rawdata(self):
        return self.view(np.ndarray)  # a view of the data

    @property
    def NyquistFreq(self):
        return 1.0/2.0/self.dt

    @property
    def df(self):
        return 1.0 / (self.dt * 2 * (self.size - 1) )

    @property
    def freq(self):
        return np.arange(self.size) * self.df

    def saveData(self, fname):
        with open(fname, 'w') as fh:
            print(self.__class__.__name__, file=fh)
            print("Freq (Hz)\t Spectral Value", file=fh)
            for f, a in zip(self.freq, self):
                print("%.4f\t%.6G" %(f, np.abs(a)), file=fh)

    def filter(self, ft):
        self *= ft

class FourierSpectrum(Spectrum):
    '''encapsulates Fourier spectrum '''
    __abr_name = 'FC'

    @property
    def N(self):
        '''The length of the time history record, significant especially
        when fft changed the original record size to a power of 2'''
        return 2*(self.size - 1) #self.size = m

    @property
    def T(self):
        '''duration of the time history, important when fft changed the
        record size to a power of 2'''
        return self.dt * self.N


    def ifft(self):
        '''calculate the inverse Fourier transform of the complex Fourier
        Spectrum

        return the time history

        '''
        th = np.fft.irfft(self)
        if self.unit.endswith('*s'):
            unit = self.unit[:-2]
        else:
            unit = self.unit + '/s'

        return self.th.__class__(self.dt, th, unit, name='th_'+self.name)

