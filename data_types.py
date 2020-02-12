import cPickle, os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import kstest, poisson, binom

class LocalWarmSpotList(object):
    dtype = [("dec", float), ("ra", float), ("pVal", float)]

    def __init__(self, **kwargs):
        self.warm_spot_list = np.recarray((0,), dtype=self.dtype)

    def __len__(self):
        return len(self.warm_spot_list)

    def add(self, theta, phi, pVal):
        spots = np.zeros(len(theta), dtype=self.dtype)
        spots["dec"] = np.pi/2-theta
        spots["ra"] = phi
        spots["pVal"] = pVal
        self.warm_spot_list = np.concatenate([self.warm_spot_list, spots])
        self.warm_spot_list.sort(order="pVal")

    @property
    def list(self):
        return self.warm_spot_list
    @list.setter
    def list(self, value):
        self.warm_spot_list = value

class LocalWarmSpotExpectation(object):
    def __init__(self, **kwargs):
        self._parametrization = None
        load_path = kwargs.pop("load_path", None)
        if load_path is not None:
            self.load(load_path)

    def save(self, path):
        with open(path, "w") as open_file:
            cPickle.dump(self._parametrization, open_file)

    def load(self, path):
        with open(path, "r") as open_file:
            param = cPickle.load(open_file)
        self._parametrization = param
        self.spline()

    def n_spots_above_threshold(self, trials, thres, verbose=False):
        r""" Count the number of local warm spots above a -log10 pValue threshold.
        In addition calculated KS test p-values if distribution fits
        a poisson and binomial distribution.

        Parameters
        ----------
        trials: array of LocalWarmSpotList
            Array containing the LocalWarmSpotList of several background trials.
        thres: float
            The threshold value in -log10(p-value)
        verbose: bool (optional), default=False
            Print in verbose mode
        """
        counts_above = [np.sum(t["pVal"] > thres) for t in trials]
        mean = np.mean(counts_above)
        p = 10**(-thres)
        ks_poisson = kstest(counts_above, poisson(mean).cdf, alternative="greater")[1]
        ks_binom   = kstest(counts_above,  binom(int(mean/p), p).cdf, alternative="greater")[1]

        if verbose:
            print "-log10(p):", thres
            print "Mean:", mean
            print "KS-Test (Poisson), p-val:", ks_poisson
            print "KS-Test (Binomial), p-val:", ks_binom

        return thres, mean, ks_poisson, ks_binom

    def generate(self, bgd_trials, log10p_steps=None):
        parametrization = []
        if log10p_steps is None:
            log10p_steps = np.linspace(2,7, 5*10+1)
        for log10p in log10p_steps:
            result = self.n_spots_above_threshold(bgd_trials, thres=log10p)
            parametrization.append(result)
        self._parametrization = np.array(parametrization)
        self.spline()

    def spline(self):
        threshold = self._parametrization[:,0]
        n_spots = self._parametrization[:,1]
        # make the spline in log10
        # add 1e-20 to avoid problems with log10(0)
        self._log10_n_expected = UnivariateSpline(threshold,
                                        np.log10(n_spots+1e-20),
                                        s=0, k=1)

    def __call__(self, logP_thres):
        if self._parametrization is None:
            raise RuntimeError("No parametrization loaded or generated.")
        # we made the spline in log10
        return 10**self._log10_n_expected(logP_thres)

class HPA_TS_value(object):
    def __init__(self, **kwargs):
        pass

class HPA_TS_Parametrization(object):
    def __init__(self, **kwargs):
        pass

class SingleSpotTrialPool(object):
    def __init__(self, **kwargs):
        pass

class BackgroundLocalWarmSpotPool(object):
    def __init__(self, **kwargs):
        pass

class Sensitivity(object):
    def __init__(self, **kwargs):
        pass
