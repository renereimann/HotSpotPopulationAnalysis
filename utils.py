#!/usr/bin/env python

#import matplotlib.pyplot as plt
import cPickle as pickle
import glob, os, re, argparse
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.stats import poisson, binom, gamma
from scipy.interpolate import UnivariateSpline

from statistics import poisson_percentile, poisson_weight
try:
    from ps_analysis.plotting.hpa import counts_above_plot, gamma_fit_to_histogram, gamma_fit_contour, gamma_fit_survival_plot
except:
    pass


def deltaPsi(dec1, ra1, dec2, ra2):
    """Calculate angular distance between two directions.
    
    Parameters
    ----------
    dec1: float, array_like
        Declination of first direction. Units: radian
    ra1: float, array_like
        Right ascension of first direction. Units: radian
    dec2: float, array_like
        Declination of second direction. Units: radian
    ra2: float, array_like
        Right ascension of second direction. Units: radian

    Returns
    -------
    ndarray
        Angular distance. Units: radian
    """

    cDec1 = np.cos(dec1) 
    cDec2 = np.cos(dec2)
    cosTheta = cDec1*np.cos(ra1)*cDec2*np.cos(ra2) + cDec1*np.sin(ra1)*cDec2*np.sin(ra2) + np.sin(dec1)*np.sin(dec2)
    cosTheta[cosTheta>1.] = 1.
    cosTheta[cosTheta<-1.] = -1.
    return np.arccos(cosTheta)

class expectation(object):
    
    def __init__(self, path):
        # read in expectation spline
        if not os.path.exists(path):
            raise ValueError("You try to load a spline from a path that does not exist. You specified: {path}".format(**locals()))
        with open(path, "r") as open_file:
            self.spl = pickle.load(open_file)
        try:
            self.spl(10.)
        except:
            print("Backup solution")
            with open(path.replace("spline_expectation", "parametrization_expectiation"), "r") as open_file:
                parametrization = pickle.load(open_file)
            self.spl = UnivariateSpline(parametrization[:,0], np.log10(parametrization[:,1]+1e-20), s=0, k=1)    
        
        self.path = path
        
    def __str__(self):
        print("Read in expectation spline from {self.path}".format(**locals()))
        print("Expectation for a pVal threshold of [2,3,4,5,6] is", self.__call__(range(2,7,1)))
        
    def __call__(self, x):
        # the spline is saved as log10(expectation) thus we have to return 10^spline
        return np.power(10, self.spl(x))
    
    def poisson_test_all_pVals(self, data):
        r"""Calculate population p-values at all thresholds """

        if len(data) < 1: 
            return 0., 0., 0, 0.

        c = np.arange(len(data), 0, -1)
        expect = self.__call__(data)
        logP = -poisson(expect).logsf(c - 1) / np.log(10)
        
        return logP
        
    def poisson_test(self, data):
        r"""Calculate population p-values at all thresholds """

        if len(data) < 1: 
            return 0., 0., 0, 0.

        c = np.arange(len(data), 0, -1)
        expect = self.__call__(data)
        logP = -poisson(expect).logsf(c - 1) / np.log(10)

        idx = np.argmax(logP)

        return logP[idx], data[idx], c[idx], expect[idx]
    
    def poisson_prob(self, data):
        return self.poisson_test(data)[0]
    
class background_pool(object):
    
    def __init__(self, **kwargs):
        self.min_ang_dist = kwargs.pop("min_ang_dist", 1.0)
        self.cutoff       = kwargs.pop("cutoff",       2.0)
        self.cut_on_nspot = kwargs.pop("cut_on_nspot", None)
        seed              = kwargs.pop("seed",         None)
        self.random       = np.random.RandomState(seed)
        
    def load_trials(self, glob_path):
        infiles = glob.glob(glob_path)
        self.n_files = len(infiles)
        trials = get_all_sky_trials(infiles, min_ang_dist=self.min_ang_dist)
        
        if self.cut_on_nspot is None:
            # in this mode we use all spots and draw a poisson number base on expecation
            self.spots_per_trial = [len(t) for t in trials]
            bgd_pool = np.concatenate([t["pVal"] for t in trials])
            self.n_trials = len(self.spots_per_trial)
            self.n_expected = np.mean(self.spots_per_trial)
        else:
            min_threshold = []
            for i, t in enumerate(trials):
                # sort events and take first n spots
                t.sort(order="pVal")
                min_threshold.append(t["pVal"][0])
                trials[i] = t[-self.cut_on_nspot:]
            self.min_threshold = np.max(min_threshold)
            self.n_trials = len(min_threshold)
            trials = np.array(trials)
            bgd_pool = trials["pVal"].ravel()
            
    
        # clean if there are nans or infs
        self.n_pool = np.shape(bgd_pool)
        self.bgd_pool = bgd_pool[np.isfinite(bgd_pool)]
        self.validate()
    
    def validate(self):
        
        if len(self.bgd_pool) != np.sum(np.isfinite(self.bgd_pool)):
            raise ValueError("You have pValues in the list that are NaN." 
                         +"We already tried to catch them but it did not work!")
        
        # The cut on n_spots has to be softwer than on the p-value threshold 
        if hasattr(self, "min_threshold") and self.min_threshold > self.cutoff: 
                raise ValueError("You want to use a min pValue threshold that is too small."
                                 +"Some spot sets from scans do not reach the min pValue threshold."
                                 +"The spot sets reach down to %f."%self.min_threshold)

    def __str__(self):
        doc = "Number of background all sky scan trial files: {self.n_files}\n".format( **locals() )
        doc += "Number of background all sky scan trials: {self.n_trials}\n".format( **locals() )
        doc += "Your background pool consists of {self.n_pool} spots\n".format( **locals() )
        if hasattr(self, "min_threshold"):
            doc += "The minimal allowed pValue threshold would be {self.min_threshold}\n".format( **locals() )
        
        return doc
        
    def plot_pool(self, savepath=None):
        plt.figure()
        plt.hist(self.bgd_pool, bins=np.linspace(self.cutoff, 8, 100))
        plt.yscale("log", nonposy="clip")
        plt.ylim(ymin=0.5)
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()
    
    def get_pseudo_experiment(self):
        nspots = self.random.poisson(self.n_expected) if self.cut_on_nspot is None else self.cut_on_nspot
        return self.random.choice(self.bgd_pool, nspots, replace=False)
    
    def load(self, load_path, seed=None):
        if not os.path.exists(load_path): raise IOError("load_path does not exist {load_path}".format(**locals()))
        with open(load_path, "r") as open_file:
            state = pickle.load(open_file)
         
        self.min_ang_dist = state["min_ang_dist"]
        self.cutoff       = state["cutoff"]
        self.cut_on_nspot = state["cut_on_nspot"]
        self.n_files      = state["n_files"]
        self.n_trials     = state["n_trials"]
        self.n_pool       = state["n_pool"]
        self.bgd_pool     = state["bgd_pool"]
        if self.cut_on_nspot is None: 
            self.spots_per_trial = state["spots_per_trial"]
            self.n_expected      = state["n_expected"]
        else: 
            self.min_threshold   = state["min_threshold"]
        self.random       = np.random.RandomState(seed)
        
    def save(self, save_path):
        state = {}
        state["min_ang_dist"] = self.min_ang_dist
        state["cutoff"]       = self.cutoff
        state["cut_on_nspot"] = self.cut_on_nspot
        state["n_files"]      = self.n_files
        state["n_trials"]     = self.n_trials
        state["n_pool"]       = self.n_pool
        state["bgd_pool"]     = self.bgd_pool
        if self.cut_on_nspot is None: 
            state["spots_per_trial"] = self.spots_per_trial
            state["n_expected"]      = self.n_expected
        else: 
            state["min_threshold"]   = self.min_threshold
        
        with open(save_path, "w") as open_file:
            pickle.dump(state, open_file)
    

class signal_trials(object):
    def __init__(self, ntrials):
        self.trials = np.empty(ntrials, dtype=[("n_inj", np.int),
                                               ("logP", np.float),
                                               ("pVal", np.float),
                                               ("count", np.int),
                                               ("exp", np.float)])
        self.count  = 0

    def __str__(self):
        doc = "Out shape: {}\n".format(np.shape(self.trials))
        doc += "Out dtype: {}\n".format(self.trials.dtype.names)
        doc += "Maximal injected N: {}\n".format(np.max(self.trials["n_inj"]))
        doc += "Mean n_inj: {}\n".format(np.mean(self.trials["n_inj"]))
        doc += "Median n_inj: {}\n".format(np.median(self.trials["n_inj"]))
        doc += "Std n_inj: {}\n".format(np.std(self.trials["n_inj"]))
        return doc
        
    def clean_nans(self):
        # double check that there are no nan pvalues
        m = np.isfinite(self.trials["logP"])
        if np.any(~m):
            print("Found infinite logP: {0:d} of {1:d}".format(np.count_nonzero(~m), len(m)))
            for n in self.trials.dtype.names:
                print(self.trials[~m][n])
        self.trials = self.trials[m]

    def plot_ninj(self, nsrc, n_inj):
        plt.figure()
        plt.hist(self.trials["n_inj"], bins=np.linspace(-0.5, np.max(self.trials["n_inj"])+0.5, np.max(self.trials["n_inj"])+2))
        plt.yscale("log", nonposy="clip")
        plt.xlabel("$n_{inj}$")
        plt.ylabel("count")
        plt.title("$N_{src}$ %d, $N_{inj}$ %.2f"%(nsrc, n_inj))

    def need_more_trials(self):
        return self.count < len(self.trials)
        
    def add_trial(self, n_tot_inj, exp_result, ping_each=250):
        self.trials[self.count] = (n_tot_inj, ) + exp_result
        if (self.count+1) % ping_each == 0: print("Finished %d"%(self.count+1))
        self.count += 1
