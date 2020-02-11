import os
import cPickle
import collections
import numpy as np
from scipy.stats import expon

class pVal_calc(object):
    def __init__(self, fit_dict):
        if type(fit_dict) is str and os.path.exists(fit_dict):
            with open(fit_dict, "r") as open_file:
                fit_dict = cPickle.load(open_file)
        self._fit_dict  = fit_dict
        self._decs      = np.array(sorted(self._fit_dict.keys()))
        self._thres     = np.array([self._fit_dict[d]["thres"] for d in self._decs])
        self._fract     = np.array([float(self._fit_dict[d]["Nthres"])/float(self._fit_dict[d]["Ntrials"]) for d in self._decs])
        self._low_TS    = np.array([self._fit_dict[d]["lowTS_param"] for d in self._decs])
        self._expon      = np.array([expon(loc=0, scale=self._fit_dict[d]["params"][1]) for d in self._decs])
        self._norm_expon = np.array([1-self._expon[i].cdf(thr) for i, thr in enumerate(self._thres)])
        
    def _pVal_from_round_dec(self, TS, dec):
        dec = np.atleast_1d(dec)
        TS = np.atleast_1d(TS)
        
        idx = np.searchsorted(self._decs, dec)
        thres                  = self._thres[idx]
        fract_above_thres      = self._fract[idx]
        low_TS                 = self._low_TS[idx]
        expon_funct            = self._expon[idx]
        expon_norm             = self._norm_expon[idx]

        # split in low TS region and high TS region
        ma = [TS > thr for thr in thres]

        # use pValue spline for low TS values
        pval     = np.array([np.array([low_TS[i](t) for t in TS]) for i in xrange(len(low_TS))])
        
        # use fit function for high TS region
        for i, (d, func) in enumerate(zip(dec, expon_funct)):
            pval[i][ma[i]] = func.sf(TS[ma[i]])/expon_norm[i]*fract_above_thres[i]
            
        return pval

    def _pVal_grid(self, TS, dec):
        idx_up = np.searchsorted(self._decs, dec)
        idx_low = idx_up-1

        p_low = self._pVal_from_round_dec(TS, self._decs[idx_low])
        p_up  = self._pVal_from_round_dec(TS, self._decs[idx_up])

        ones = np.ones_like(TS)
        _, dec_low = np.meshgrid(ones, self._decs[idx_low])
        _, dec_up  = np.meshgrid(ones, self._decs[idx_up])
        _, d       = np.meshgrid(ones, dec)

        return (p_low-p_up)/(dec_low-dec_up)*(d-dec_low)+p_low

    def __call__(self, TS, dec):
        TS_single = not isinstance(TS, (collections.Sequence, np.ndarray, np.recarray))
        dec_single = not isinstance(dec, (collections.Sequence, np.ndarray, np.recarray))
        
        if (not dec_single) and (not TS_single) and len(TS) == len(dec):
            # if TS and dec match to each other, return a single array
            dec = np.array(dec, dtype=float)
            TS = np.array(TS, dtype=float)
            pVal = np.zeros_like(dec)
            for d in sorted(set(dec)):
                m = dec==d
                pVal[m] = self._pVal_grid(TS[m], d)[0,:]
            return pVal
        
        pVal = self._pVal_grid(TS, dec)
        if dec_single and TS_single:
            return pVal[0,0]
        if dec_single:
            return pVal[0,:]
        if TS_single:
            return pVal[:,0]
        if len(TS) != len(dec):
            return pVal 
        else:
            raise NotImplementedError("This is a case that never should happen.")
