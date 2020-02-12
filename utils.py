#!/usr/bin/env python

#import matplotlib.pyplot as plt
import cPickle as pickle
import glob, os, re, argparse
import numpy as np

from scipy.stats import poisson
from scipy.interpolate import UnivariateSpline

from statistics import poisson_percentile, poisson_weight

class HPA_analysis(object):
    dtype = [("hpa_ts", float), ("log10p_thres", float), ("n_observed", int), ("n_expected", float)]

    def __init__(self, expectation):
        self.expectation = expectation

    def scan(self, data):
        r"""Calculate population p-values at all thresholds """
        # data contains the sorted -log10(p-value) list

        if len(data) < 1:
            return 0., 0., 0, 0.

        # n_observed in decending order, because p-value list is sorted
        n_observed = np.arange(len(data), 0, -1)
        # evaluate expectation at -log10(p-value) of data
        n_expected = self.expectation(data)
        # the poisson p-value
        poisson_pValues = -poisson(n_expected).logsf(n_observed - 1) / np.log(10)

        return poisson_pValues, data, n_observed, n_expected

    def best_fit(self, data):
        poisson_pValues, data, n_observed, n_expected = self.scan(data)
        idx = np.argmax(poisson_pValues)
        return np.array((poisson_pValues[idx], data[idx], n_observed[idx], n_expected[idx]), dtype=self.dtype)

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


class BackgroundLocalWarmSpotPool(object):
    def __init__(self, **kwargs):
        seed = kwargs.pop("seed", None)
        self.random = np.random.RandomState(seed)

    def load_trials(self, infiles):
        # read in all sky trials
        trials = []
        for file_name in infiles:
            with open(file_name, "r") as open_file:
                temp = pickle.load(open_file)
            trials.extend(temp)
        print "Read in %d trials"%len(trials)

        # in this mode we use all spots and draw a poisson number base on expecation
        # the mean expected number of events above p-value threshold
        self.n_expected = np.mean([len(t) for t in trials])
        # the pool of background p-values
        bgd_pool = np.concatenate([t["pVal"] for t in trials])
        # clean if there are nans or infs
        self.bgd_pool = bgd_pool[np.isfinite(bgd_pool)]

    def get_pseudo_experiment(self):
        nspots = self.random.poisson(self.n_expected)
        return self.random.choice(self.bgd_pool, nspots, replace=False)

    def load(self, load_path, seed=None):
        with open(load_path, "r") as open_file:
            state = pickle.load(open_file)
        self.bgd_pool = state["bgd_pool"]
        self.n_expected = state["n_expected"]
        if seed is not None:
            self.random = np.random.RandomState(seed)

    def save(self, save_path):
        state = {"bgd_pool": self.bgd_pool, "n_expected": self.n_expected}
        with open(save_path, "w") as open_file:
            pickle.dump(state, open_file)

########################################################################

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

        return logP, data, c, expect

    def poisson_test(self, data):
        logP, data, c, expect = self.poisson_test_all_pVals(data)
        idx = np.argmax(logP)
        return logP[idx], data[idx], c[idx], expect[idx]

    def poisson_prob(self, data):
        return self.poisson_test(data)[0]


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
