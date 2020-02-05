#!/usr/bin/env python

import matplotlib.pyplot as plt
import cPickle, glob, os, copy, re, healpy, argparse
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy import stats
from scipy.stats import poisson, norm, binom, gamma
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import erfinv, erf

from ps_analysis.scripts.llh_functions import llh_loader
from ps_analysis.scripts.stager import FileStager

try:
    from ps_analysis.plotting.hpa import counts_above_plot, gamma_fit_to_histogram, gamma_fit_contour, gamma_fit_survival_plot
except:
    pass
from SourceUniverse.SourceUniverse import SourceCountDistribution

def poisson_percentile(mu, x, y, yval):
    r"""Calculate upper percentile using a Poisson distribution.

    Parameters
    ----------
    mu : float
        Mean value of Poisson distribution
    x : array_like,
        Trials of variable that is expected to be Poisson distributed
    y : array_like
        Observed variable connected to `x`
    yval : float
        Value to calculate the percentile at

    Returns
    -------
    score : float
        Value at percentile *alpha*
    err : float
        Uncertainty on `score`

    """
    x = np.asarray(x, dtype=np.int)
    y = np.asarray(y, dtype=np.float)

    w = poisson_weight(x, mu)

    # Get percentile at yval.
    m = y > yval
    u = np.sum(w[m], dtype=np.float)

    if u == 0.:
        return 1., 1.

    err = np.sqrt(np.sum(w[m]**2)) / np.sum(w)

    return u / np.sum(w, dtype=np.float), err

def poisson_weight(vals, mean, weights=None):
    r"""Calculate weights for a sample that it resembles a Poisson.

    Parameters
    ----------
    vals : array_like
        Random integers to be weighted
    mean : float
        Poisson mean
    weights : array_like, optional
        Weights for each event

    Returns
    -------
    ndarray
        Weights for each event

    """
    mean = float(mean)
    vals = np.asarray(vals, dtype=np.int)

    if weights is None:
        weights = np.ones_like(vals, dtype=np.float)

    # Get occurences of integers.
    bincount = np.bincount(vals, weights=weights)

    n_max = len(bincount)

    # Get poisson probability.
    if mean > 0:
        p = scipy.stats.poisson(mean).pmf(range(n_max))
    else:
        p = np.zeros(n_max, dtype=np.float)
        p[0] = 1.

    # Weights for each integer
    w = np.zeros_like(bincount, dtype=np.float)
    m = bincount > 0
    w[m] = p[m] / bincount[m]

    w = w[np.searchsorted(np.arange(n_max), vals)]

    return w * weights

def llh2Sigma(llh, dof, alreadyTimes2=False, oneSided=False):
    llh = np.atleast_1d(llh)
    if np.any(llh < 0): raise ValueError(  "Can not calculate the significance for a negative value of llh!"  )
    if not alreadyTimes2:   dLlhTimes2 = llh*2.0
    else:                   dLlhTimes2 = llh
    return pval2Sigma(  chiSquaredVal2pVal(dLlhTimes2, dof), oneSided  )

def pval2Sigma(pval, oneSided=False):
    if oneSided: pval *= 2.0 # usually not done one-sided
    sigma = erfinv(1.0 - pval)*np.sqrt(2)
    return sigma

def deltaPsi(dec1, ra1, dec2, ra2):
    """
    Calculate angular distance.
    
    Args:
>------->-------dec1: Declination of first direction in radian
>------->-------ra1: Right ascension of first direction in radian
>------->-------dec2: Declination of second direction in radian
>------->-------ra2: Right ascension of second direction in radian
>------->-------
>-------Returns angular distance in radian
    """
    return deltaPsi2(np.sin(dec1), np.cos(dec1), np.sin(ra1), np.cos(ra1), np.sin(dec2), np.cos(dec2), np.sin(ra2), np.cos(ra2))


def deltaPsi2(sDec1, cDec1, sRa1, cRa1, sDec2, cDec2, sRa2, cRa2):
    """
    Calculate angular distance.
    
    Args:
>------->-------sDec1: sin(Declination of first direction)
>------->-------cDec1: cos(Declination of first direction)
>------->-------sRa1: sin(Right ascension of first direction)
>------->-------cRa1: cos(Right ascension of first direction)
>------->-------sDec2: sin(Declination of second direction)
>------->-------cDec2: cos(Declination of second direction)
>------->-------sRa2: sin(Right ascension of second direction)
>------->-------cRa2: cos(Right ascension of second direction)
>------->-------
>-------Returns angular distance in radian
    """
    tmp = cDec1*cRa1*cDec2*cRa2 + cDec1*sRa1*cDec2*sRa2 + sDec1*sDec2
    tmp[tmp>1.] = 1.
    tmp[tmp<-1.] = -1.
    return np.arccos(tmp)


def get_spots(p_map, cutoff_pval=3):
    """ extract local warm spots from a p-value skymap.
    Returns list of pVal, theta, phi values
    """
    
    # get npix and nside
    npix = len(p_map)
    nside = healpy.npix2nside(npix)

    # mask large p-values and infs
    m = np.logical_or(p_map < cutoff_pval, np.isinf(p_map))

    warm_spots = []
    # loop over remaining pixels
    for pix in np.arange(npix)[~m]:
        theta, phi = healpy.pix2ang(nside, pix)
        ids = healpy.get_all_neighbours(nside, theta, phi)
        # if no larger neighbour: we are at a spot --> save the idx
        if not any(p_map[ids] > p_map[pix]):
            warm_spots.append(pix)

    # get pVal and direction of spots
    p_spots = p_map[warm_spots]
    theta_spots, phi_spots = healpy.pix2ang(nside, warm_spots)

    # fill into record-array
    spots = np.recarray((len(p_spots),), dtype=[("theta", float), ("phi", float), ("pVal", float)])
    spots["theta"] = theta_spots
    spots["phi"]   = phi_spots
    spots["pVal"]  = p_spots
    spots.sort(order="pVal")

    return spots
    
def counts_above_pval(trials, thres, plotting=False, verbose=False, plot_path=None, naive_expectation=None):
    """ Count the number of spots above a certain pValue threshold.
    You have to give the 'trials' and the threshold pValue as 'pVal'.
    Beside a fit of the poissonian also the ks-probability for a poissonian to data is performed.

    If plotting is True we will produce a histogram of number of trials together with fitted Poissonian.
    If plot_path is not None this plot is saved at that path and will not be shown otherwise it is send to display.
    If verbose is True you get a lot of print outs with different parameted.
    If naive_expectation is not None you have to give the number of effective trials. KS-Tests will be tested vs naive_expectations.
    Naive expectation is: N_expected = N_eff_trials*pValue
    """
    
    counts_above = [np.sum(t["pVal"] > thres) for t in trials]
    mean = np.mean(counts_above)
    if not naive_expectation is None:
        mean = 10**(-thres)*naive_expectation 
    ks_poisson = stats.kstest(counts_above, poisson(mean).cdf, alternative="greater")[1]
    N_trials = mean/np.power(10, -thres)
    ks_binom   = stats.kstest(counts_above,  binom(int(N_trials), np.power(10, -thres)).cdf, alternative="greater")[1]
    
    if plotting and (np.max(counts_above)-np.min(counts_above)) > 0 :
        curr_plot = counts_above_plot(counts_above, thres)
        curr_plot.plot(savepath=plot_path)
    
    if verbose:
        print "-log10(p):", thres
        print "Mean:", mean
        print "KS-Test (Poisson), p-val:", ks_poisson
        print "KS-Test (Binomial), p-val:", ks_binom
    
    return thres, mean, ks_poisson, ks_binom

def get_all_sky_trials(glob_path, min_ang_dist=1.):
    """ Read in spots of all sky trials that are under 'glob_path'.
    It is assumed that the spots are already sorted by pvalue.
    An additional cut is performed on the minimal angular distance between two hot spots. 
    Spots with an angular distance smaller than 'min_ang_dist' (in degree) are cut away. 
    """
     
    trials = [] 
    for path in glob.glob(glob_path):
        with FileStager(path, "r") as open_file:
            temp = cPickle.load(open_file)
        trials.extend(temp)
    print "Read in %d files"%len(trials)
    
    # require min dist > x deg
    for j, t0 in enumerate(trials):
        remove = []
        for i in np.arange(0, len(t0)):
            temp_dist = np.degrees(deltaPsi(np.pi/2.-t0[i]["theta"], t0[i]["phi"], np.pi/2.-t0[i+1:]["theta"], t0[i+1:]["phi"]))
            m = np.where(temp_dist < min_ang_dist)[0]
            if len(m) == 0: continue
            # we have at least 2 points closer than 1 deg
            if any(t0[m+i+1]["pVal"] >= t0[i]["pVal"]) or np.isnan(t0[i]["pVal"]): remove.append(i)
            else:
                print t0[m+i+1]["pVal"]
                print t0[i]["pVal"]
                raise ValueError("Should never happen because spots should be sorted by pValue")
        trials[j] = t0[~np.in1d(range(len(t0)), remove)]
        # we have to access trials[j] , because changes in t0 would just be local and would not be visible outside the loop
    return trials
    
def make_gamma_fit(read_path, verbose=False, 
                   grid_size=[16,16], hold_fig=False,  
                   plot_hist=True, plot_path_hist=None, 
                   plot_contour=True, plot_path_contour=None, 
                   plot_survival=True, plot_path_survival=None, label=None):
    with FileStager(read_path, "r") as read_file:
        trials = cPickle.load(read_file)
    
    params = gamma.fit(trials, floc=0)
    
    ks_gamma = stats.kstest(trials, gamma(*params).cdf, alternative="greater")[1]
        
    # histogram
    hi, edg = np.histogram(trials, bins=np.linspace(0, 5, 21), density=True)
    
    # calculate LLH landscape
    x = np.linspace(params[0]-0.3, params[0]+0.3, grid_size[0])
    y = np.linspace(params[2]-0.1, params[2]+0.1, grid_size[1])
    xv, yv = np.meshgrid(x, y)
    llh = [-np.sum(gamma(x1, 0, x2).logpdf(trials)) for x1, x2 in zip(xv.flatten(), yv.flatten())]
    dllh = np.reshape(llh, np.shape(xv))-np.min(llh)
    xmin = xv.flatten()[np.argmin(dllh)]
    ymin = yv.flatten()[np.argmin(dllh)]
    sigma = llh2Sigma(dllh, dof=2, alreadyTimes2=False, oneSided=False)
    
    # median
    trials.sort()    
    median = np.median(trials)
    median_lower = trials[int(len(trials)/2.-np.sqrt(len(trials))*0.5)]
    median_upper = trials[int(len(trials)/2.+np.sqrt(len(trials))*0.5)+1]
    
    # extrapolation
    ma = sigma < 1
    TS_3sig = []
    TS_5sig = []
    for xx, yy in zip(xv[ma], yv[ma]):    
        func_3sig = lambda p: np.log((gamma(xx, 0, yy).sf(p)-0.00135)**2)
        func_5sig = lambda p: np.log((gamma(xx, 0, yy).sf(p)-2.867e-7)**2)
        res_3sig = minimize(func_3sig, x0=[5],  method="L-BFGS-B")
        res_5sig = minimize(func_5sig, x0=[10], method="L-BFGS-B")
        TS_3sig.append(res_3sig["x"][0])
        TS_5sig.append(res_5sig["x"][0])
    func_3sig = lambda p: np.log((gamma(*params).sf(p)-0.00135)**2)
    func_5sig = lambda p: np.log((gamma(*params).sf(p)-2.867e-7)**2)
    res_3sig = minimize(func_3sig, x0=[5],  method="L-BFGS-B")
    res_5sig = minimize(func_5sig, x0=[10], method="L-BFGS-B")
    ext_3sig = res_3sig["x"][0]
    ext_5sig = res_5sig["x"][0] 
    lower_3sig = np.min(TS_3sig)
    upper_3sig = np.max(TS_3sig)
    lower_5sig = np.min(TS_5sig)
    upper_5sig = np.max(TS_5sig)
        
    if verbose: 
        print "Fit parameters:", params
        print "KS-Test between fit and sample", ks_gamma
        print "Median at {:.2f} + {:.2f} - {:.2f}".format(median, median-median_lower, median_upper-median)
        print "3 sigma at {:.2f} + {:.2f} - {:.2f}".format(ext_3sig, ext_3sig-lower_3sig, upper_3sig-ext_3sig)
        print "5 sigma at {:.2f} + {:.2f} - {:.2f}".format(ext_5sig, ext_5sig-lower_5sig, upper_5sig-ext_5sig)    
    
    if plot_hist:
        curr_plot = gamma_fit_to_histogram(params, hi, edg)
        curr_plot.plot(savepath=plot_path_hist)
            
    if plot_contour:
        curr_plot = gamma_fit_contour(xv, yv, sigma, xmin, ymin)
        curr_plot.plot(savepath=plot_path_contour)
        
    if plot_survival:
        curr_plot = gamma_fit_survival_plot( params, trials, 
                                             (median_lower, median_upper), 
                                             (lower_3sig, upper_3sig), 
                                             (lower_5sig, upper_5sig))
        curr_plot.plot(savepath=plot_path_survival)
            
    return {"params": params, "ks_pval": ks_gamma, "median": (median, (median_lower, median_upper)), "3sig": (ext_3sig, (lower_3sig, upper_3sig)), "5sig": (ext_5sig, (lower_5sig, upper_5sig))}

def job_num_2_nsrc_ninj_nsrcIdx(job_num):
    """ Convert a job number to number of sources 
    and number of injected events"""
    
    k = 2
    j, i = divmod(job_num // k, 50)
    nsrc = 2**j
    n_inj = float((i + 1)**2) / nsrc
    
    return nsrc, n_inj, i
    
class expectation(object):
    
    def __init__(self, path):
        # read in expectation spline
        if not FileStager.exists(path):
            raise ValueError("You try to load a spline from a path that does not exist. You specified: {path}".format(**locals()))
        with FileStager(path, "r") as open_file:
            self.spl = cPickle.load(open_file)
        try:
            self.spl(10.)
        except:
            print("Backup solution")
            with FileStager(path.replace("spline_expectation", "parametrization_expectiation"), "r") as open_file:
                parametrization = cPickle.load(open_file)
            self.spl = UnivariateSpline(parametrization[:,0], np.log10(parametrization[:,1]+1e-20), s=0, k=1)    
        
        self.path = path
        
    def __str__(self):
        print "Read in expectation spline from {self.path}".format(**locals())
        print "Expectation for a pVal threshold of [2,3,4,5,6] is", self.__call__(range(2,7,1))
        
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
        self.n_files = len(glob.glob(glob_path))
        trials = get_all_sky_trials(glob_path, min_ang_dist=self.min_ang_dist)
        
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
        if not FileStager.exists(load_path): raise IOError("load_path does not exist {load_path}".format(**locals()))
        with FileStager(load_path, "r") as open_file:
            state = cPickle.load(open_file)
         
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
        
        with FileStager(save_path, "w") as open_file:
            cPickle.dump(state, open_file)
    
class signal_pool(object):
    def __init__(self, **kwargs):
        self.set_seed( kwargs.pop("seed", None))
        
    def set_seed(self, seed):
        self.random            = np.random.RandomState(seed)
        
    def __str__(self):
        doc = ""
        print "Different declinations:", np.shape(self.injects)
        print "Typical shape of inject per declination:", np.shape(self.injects[0])
        print "Declination range:", np.degrees(np.min(self.declinations)), np.degrees(np.max(self.declinations))
        print "Det weights shape, min and max:", np.shape(self.det_weight), np.min(self.det_weight), np.max(self.det_weight)
        print "Singal pool shape:", np.shape(self.signal_pool)
        if hasattr(self, "flux_weights"):
            print self._weight_str
        return doc
        
    def load_trials(self, pathes, pathes_add, pValue_calculator, pathes_add2=[]):
        """ Read in sensitivity trials from given file path
        You get back a list of injection trials, detector weights and declination positions.
        You should previde two lists of file names. In addition you have to give a pValue_calculator, 
        thus the local pValue can be calculated."""

        # we want to get uniform distributed trials in sindec, thus just take equidist trials in sinDec. 
        # This is how we generate the dag files
        equidist_sindec = 100
        equidist_dec = np.arcsin(np.linspace(np.sin(np.radians(-5)), 1, equidist_sindec+1))
        pathes_equidist = []
        for f in pathes:
            dec_str = re.findall("_declination_([+-]?\d*\.\d+|[+-]?\d+)", f)[0]
            if np.min(np.abs(equidist_dec - float(dec_str))) < 1e-10: pathes_equidist.append(f)
        assert len(pathes_equidist) == equidist_sindec, "you did not get all declinations"

        # det_weight should be used to correct for zenith acceptance 
        # incects will be a list of signal trials
        declinations, det_weight, injects, mus, fluxes = list(), list(), list(), list(), list()
        for f1 in pathes_equidist: 
                # read single source trials from sensitivity calculation
                dec_key, mu, flux, inj = self.read_in_sens_file(f1)
                
                # add more files with same declination
                float_re = "([+-]?\d*\.\d+|[+-]?\d+)"
                dec_str = re.findall("_declination_{float_re}".format(**locals()), f1)[0]
                for f_add in pathes_add:
                    if not dec_str in f_add: continue
                    dec_key_add, _, _,  inj_add = self.read_in_sens_file(f_add)
                    assert(np.fabs(dec_key   - dec_key_add)   < 2. * np.finfo(type(dec_key)).eps), "{dec_key}, {dec_key_add}".format(**locals())
                    inj = np.append(inj, inj_add)
                
                for f_add in pathes_add2:
                    if not dec_str in f_add: continue
                    inj_add = self.read_in_trial_file(f_add)
                    inj = np.append(inj, inj_add)
                
                # add pValue and relative weighting
                inj = append_fields(inj, ["pVal", "w"],
                                 [np.ones(len(inj), dtype=np.float),
                                  np.ones(len(inj), dtype=np.float)])
                inj["pVal"] = pValue_calculator.neglogpValFunc(inj["TS"], dec_key)
                # weights are 1/N for the moment, uniform distribution  
                inj["w"]    = np.ones(len(inj), dtype=np.float) / len(inj)
                inj         = inj[np.isfinite(inj["pVal"])]
                
                declinations.append(dec_key)
                det_weight.append(mu/flux)
                mus.append(mu)
                fluxes.append(flux)
                injects.append(inj) 

        # the detector weights is normed relatively to the mean of the hemisphere
        det_weight        = np.asarray(det_weight)
        self.flux_2_mu    = det_weight
        self.flux         = np.array(fluxes)
        self.mu           = np.array(mus)
        det_weight       /= np.mean(det_weight)
        self.mean_flux_2_mu = np.mean(self.mu/self.flux)

        self.injects      = injects
        self.det_weight   = det_weight
        self.declinations = np.array(declinations)
        self.signal_pool  = np.concatenate(self.injects)
    
    def read_in_sens_file(self, path):
        """ Read in file and return declinationa flux2mu factor and trial-array with "n_inj", "TS" 
        Raises if file does not exist.
        """
        if not FileStager.exists(path):
            raise ValueError("Could not load signal trials, because file does not exist.\nYou specified {path}".format(**locals()))
        with FileStager(path) as open_file:
            job_args, trials = cPickle.load(open_file)
        dec_key    = trials.keys()[0]
        mu         = np.mean(trials[dec_key][0]["mu"]) 
        flux       = np.mean(trials[dec_key][0]["flux"])
        
        return dec_key, mu, flux, trials[dec_key][1][["n_inj", "TS"]]
        
    def read_in_trial_file(self, path):
        """ Read in file and return declinationa flux2mu factor and trial-array with "n_inj", "TS" 
        Raises if file does not exist.
        """
        if not FileStager.exists(path):
            raise ValueError("Could not load signal trials, because file does not exist.\nYou specified {path}".format(**locals()))
        with FileStager(path) as open_file:
            job_args, trials = cPickle.load(open_file)
        
        return trials[["n_inj", "TS"]]
      
    def plot_sinDec_distribution(self):
        plt.figure()
        plt.hist(np.sin(self.declinations))
        plt.xlabel("$\sin(\delta)$")
        plt.ylabel("counts")
 
    def plot_mu_2_flux(self):
        idxs = np.argsort(self.declinations)
        plt.plot(np.array(self.declinations)[idxs], np.array(self.flux)[idxs] / np.array(self.mu)[idxs])
        plt.xlabel("$\sin(\delta)$")
        plt.ylabel("flux/mu")
    
    def _weight_str(self):
        return "N_inj: {}".format(self.n_inj)
        
    def get_signal_weights(self, n_inj):
        """Calculates weights for each trials of injects. The mean expectation is n_inj.
        Take detector acceptance into account and poissoinan spread.
        Return signal pool with corresponding weights"""

        # scale the number of injected events for all trials according to the relative weight
        # w = Poisson(lambda=1/N*det_w*n_inj_expectation).pdf(n_inj_simulated)
        
        w = np.concatenate([poisson_weight(inject["n_inj"], n_inj * det_w) for inject, det_w in zip(self.injects, self.det_weight)])
        
        # sum over w has to be equal to len of injects because sum over w for each declination has to be 1. 
        # the sum of weighted trials is 10% less then what it should be, missing edges
        # if not fullfilled essential regions of the poisson distributions are not covered
        
        if np.fabs(1. - w.sum() / len(self.injects)) > 0.1:
            print("{}, {}".format(w.sum(), len(self.injects)))
            raise ValueError("Poissonian weighting not possible, too few trials.\nGenerate events within ns-range from {} to {}".format(np.min(n_inj*self.det_weight), np.max(n_inj*self.det_weight)) )
        
        # draw equal distributed between sinDec
        w /= w.sum()
        assert np.all(w <= 0.1), "Too few trials in this regime, exit.\nSome events would have a probability larger then 10%.\nMax weights: {}".format( np.sort(w)[-10:] )
        
        self.flux_weights   = w
        self.n_inj             = n_inj
        
    def get_pseudo_experiment(self, nsrc):
        """ Draws n src p-values from signal pool and returns the local p-values as well as the n_injected summed over all nsrc."""
        if not hasattr(self, "random"):       raise ValueError("Random not yet set. Please set a seed with set_seed(seed)")
        if not hasattr(self, "flux_weights"): raise ValueError("Weights not yet initialized. Please initialize them with get_signal_weights(n_inj)")
        
        sig       = self.random.choice(self.signal_pool, nsrc, p=self.flux_weights)
        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj
       
    def save(self, save_path):
        state = {}
        state["flux_2_mu"]    = self.flux_2_mu
        state["injects"]      = self.injects
        state["declinations"] = self.declinations
        state["flux"]         = self.flux
        state["mu"]           = self.mu
        with FileStager(save_path, "w") as open_file:
            cPickle.dump(state, open_file, protocol=2)
      
    def load(self, load_path, **kwargs):
        """ Loads signal pool from file,
        you can set in addition the seed"""
        if not FileStager.exists(load_path): raise IOError("Inpath does not exist. {load_path}".format(**locals()))
        with FileStager(load_path, "r") as open_file:
            state = cPickle.load(open_file) 
        self.flux_2_mu    = state["flux_2_mu"]
        self.injects      = state["injects"]
        self.det_weight   = self.flux_2_mu/np.mean(self.flux_2_mu)
        self.declinations = state["declinations"]
        self.flux         = state["flux"]
        self.mu           = state["mu"]
        self.signal_pool  = np.concatenate(self.injects)
        self.set_seed( kwargs.pop("seed", None))
        self.mean_flux_2_mu = np.mean(self.mu/self.flux)
       
class signal_pool_FIRESONG(signal_pool):
        
    def load_firesong_representation(self, infile, density=None):
        fluxes = []
        with open(infile, "r") as open_file:
            for line in open_file:
                if line.startswith("#"): continue
                if len(line.split()) != 3: continue
                fluxes.append( line.split()[2] )
        self.firesong_fluxes = np.array(fluxes, dtype=float) # in GeV cm^-2 s^-1
        # self.mu   # ranges from about 6-12
        # self.flux # in 5e-10 - 4e-9 in GeV cm^-2 s^-1
        # fluxes in FIRESONG are Phi0(at 100TeV) * (100TeV)^2
        # this is the same as Phi0(at 1 GeV) *1GeV^2
        # as GeV is the basic unit this is the same as Phi0(at 1GeV) with change of units from 1/GeV cm^2 s to GeV/cm^2 s
        if density is not None:
            self.default_density = density
    
    def get_pseudo_experiment(self, density=None):
        """ Draws poisson mus from firesong expectations and get corresponding p-values from signal pool and returns the local p-values as well as the n_injected summed over all sources."""
        # firesong mues are expectation values and have declination dependence already integrated
        # the final mu however is a poisson random number
        
        if (density is not None) and (hasattr(self, "default_density")):
            if density > self.default_density:
                raise ValueError("The given density is above the default density. That is not possible.")
            fluxes = self.random.choice(self.firesong_fluxes, int(density/self.default_density*len(self.firesong_fluxes)))
        else:
            fluxes = self.firesong_fluxes

        dec_idx = self.random.choice(range(len(self.declinations)), len(fluxes))
        mus = self.random.poisson( fluxes * self.mu[dec_idx] / self.flux[dec_idx] )
        # get rid of all the null events we have to save computing

        dec_idx = dec_idx[mus!=0]
        mus = mus[mus != 0]
        sig = np.zeros(len(mus), dtype=self.signal_pool.dtype)
        for i, (mu, idx) in enumerate(zip(mus, dec_idx)):
            mask_ninj = self.injects[idx]["n_inj"] == mu
            sig[i] = self.random.choice(self.injects[idx][mask_ninj])

        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj
                                 
class signal_pool_SourceUnivers(signal_pool):
        
    def load_SourceUnivers_representation(self, infile, seed=None):
        self.flux_dist = SourceCountDistribution(file=infile, seed=seed)
        
    def get_pseudo_experiment(self, density):
        """ Draws poisson mus from firesong expectations and get corresponding p-values from signal pool and returns the local p-values as well as the n_injected summed over all sources."""
        # firesong mues are expectation values and have declination dependence already integrated
        # the final mu however is a poisson random number
        
        dec_idx = []
        mus = []
        n_sources = self.flux_dist.calc_N(density)

        count = 0
        while count < n_sources:
            n_loop = np.min([n_sources-count, 10000000])
            fluxes = 10**self.flux_dist.random(n_loop) # in GeV cm^-2 s^-1
            dec_idx_loop = self.random.choice(range(len(self.declinations)), len(fluxes))
            mus_loop = self.random.poisson( fluxes * self.mu[dec_idx_loop] / self.flux[dec_idx_loop] )
            count += len(mus_loop)

            # get rid of all the null events we have to save computing
            dec_idx.append(dec_idx_loop[mus_loop!=0])
            mus.append(mus_loop[mus_loop != 0])

        dec_idx = np.concatenate(dec_idx)
        mus = np.concatenate(mus)

        sig = np.zeros(len(mus), dtype=self.signal_pool.dtype)
        for i, (mu, idx) in enumerate(zip(mus, dec_idx)):
            try:
                mask_ninj = self.injects[idx]["n_inj"] == mu
                sig[i] = self.random.choice(self.injects[idx][mask_ninj])
            except:
                pass

        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj
                                 
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
        
def get_mu_2_flux_factor(season, sinHem, spectral_index=2.):
    """ Returns the conversion factor from mu to flux.
    
    Parameters:
        - season: name of seasons
        - sinHem: float, that gives the boarder of the hemisphere
        - spectral_index: float, give the spectral index of the flux that should be converted to
    Returns:
        ConversionFactor: Float
    """
    
    # first get mc via llh object
    # we can use any model because we just need the MC which just depend on season
    # combine all mc samples to one array
    loader = llh_loader()
    args = llh_loader.get_argparser().parse_args(["--model", "energy", "--season", season])
    llh = loader.get_llh(args)
    mc = []
    for samp, livetime in zip(llh.mc.itervalues(), llh.livetime.itervalues()):
        samp["ow"] *= livetime
        mc.append(samp)
    mc = np.concatenate(mc)
    del llh
    
    assert type(mc) == np.ndarray, "mc is not a structured array. Its type is {}".format(type(mc))
    assert "ow"    in mc.dtype.names, "mc is not a complet MC structured array, ow is missing. mc contains {}".format(mc.dtype.names)
    assert "trueE" in mc.dtype.names, "mc is not a complet MC structured array, trueE is missing. mc contains {}".format(mc.dtype.names)
    assert "dec"   in mc.dtype.names, "mc is not a complet MC structured array, dec is missing. mc contains {}".format(mc.dtype.names)
    assert type(sinHem) in [float, np.float64] , "sinHem is not a float. Its tpye is {}".format(type(sinHem))
    assert sinHem < 1 and sinHem > -1, "sinHem is not in the range -1 to 1. It is {}".format(sinHem)
    assert type(spectral_index) in [float, np.float64], "spectral_index is not a float. Its type is {}".format(type(spectral_index))
    
    weights = mc["ow"] * mc["trueE"]**(-np.abs(spectral_index))
    mask_hem = mc["dec"] > np.arcsin(sinHem)

    solAng = 2. * np.pi * (1. - sinHem)
    denominator = weights[mask_hem].sum()
    
    return solAng/denominator
    
def get_mu2flux_from_sens_files(glob_path):
    equidist_sindec = 100
    equidist_dec = np.arcsin(np.linspace(np.sin(np.radians(-5)), 1, equidist_sindec+1))
    pathes_equidist = []
    for f in  sorted(glob.glob(os.path.join(glob_path,  "sens_*.pickle"))):
        dec_str = re.findall("_declination_([+-]?\d*\.\d+|[+-]?\d+)", f)[0]
        if np.min(np.abs(equidist_dec - float(dec_str))) < 1e-10: pathes_equidist.append(f)
    assert len(pathes_equidist) == equidist_sindec, "you did not get all declinations"

    flux_div_mu = []
    for f in pathes_equidist:
        with open(f, "r") as open_file:
            job_args, tmp = cPickle.load(open_file)
        flux_div_mu.append((np.mean(np.array(tmp[tmp.keys()[0]][0]["flux"]) / np.array(tmp[tmp.keys()[0]][0]["mu"]))))
    mu2flux = np.mean(flux_div_mu)
    return mu2flux
    
def get_files_for_nsrc(nsrc, glob_path):
    """ Returns a list of files that are generated with nsrc as parameter. 
    Therefor we look in all file-pathes for the first occurence of eight digits 
    in a row and compare to number to nsrc.
    
    Parameters:
        - nsrc: float Number of injected sources
        - glob_path: string A glob path where we will find the HPA signal trials.
        
    Returns:
        - list of file pathes
    """
    
    files = sorted(glob.glob(glob_path))
    assert len(files) > 0, "There are no files in the glob_path: {glob_path}".format(**locals()) 
    
    # first get the nsrc number of each file
    nsrc_file_list = []
    for f in files:
        # get FIRST match in a string
        number = re.search('_'+8*'[0-9]', f)
        if number is None:
            raise ValueError("There is no 8 digit number in the file path.")
        else:
            # get the found string and remove the leading "_"
            number = int(number.group(0)[1:])
            nsrc_file_list.append((number, f))

    assert len(files)==len(nsrc_file_list), "We have lost files. Thats not good."
            
    # select just file pathes with nsrc in path
    selected_file_list = []
    for f in nsrc_file_list:
        if f[0] == nsrc:
            selected_file_list.append(f[1])
    
    if len(selected_file_list) == 0:
        raise ValueError("Did not find any file that matches nsrc. nsrc is {}".format(nsrc))
        
    return selected_file_list
    
def get_data_for_nsrc(nsrc, glob_path):
    """ Returns a structured array with signal trials for nsrc.
    
    Parameters:
        - nsrc: float Number of injected sources
        - glob_path: string A glob path where we will find the HPA signal trials.
        
    Returns:
        - Structured array of singal trials
    """
    
    files = get_files_for_nsrc(nsrc, glob_path)
    
    # signal trials should be a structured array with this data format
    data = np.empty(0, dtype=[("n_inj", np.int),
                              ("logP", np.float),
                              ("pVal", np.float),
                              ("count", np.int),
                              ("exp", np.float)])

    for f in files:
        with FileStager(f, "r") as open_file:    
            temp = np.load(open_file)

        #isStructArrayWith(temp, "loaded signal trials (temp)", data.dtype.names)
        #assert len(temp) > 0, "The loaded list is empty. The file was {f}".format(**locals())
        assert np.all(np.isfinite(temp["logP"])), "Found nan logP values in trials from file {f}".format(**locals())
        data = np.concatenate([data, temp])

    assert len(data) > 0, "There are no trials to load for {nsrc} from {glob_path}".format(**locals())
    assert np.all(np.isfinite(data["logP"])), "Found nan logP values in data."
        
    return data  

def sens_estimation(trials, TS_thres, perc, eps):
    """ Estimates sensitivity and discovery potential for a fixed number of nsources.
    Returns mu at sensitivity or discovery potential
    
    Parameters:
        - trials, structured array, Signal trials with at least n_inj and logP
        - TS_thres, float, bgd TS threshold that should be beaten by signal distribution
        - perc, float, fraction of signal distribution tha should beat the bgd TS threshold
        - eps, float tolerance of minimizer
        
    Returns:
        - mu, number of events per source at sensitivity level
        - w, poisson weight of trials at sensitivity level
        - beta, fraction of signal distribution that beat the TS threshold
        - beta_err, uncertainty on fraction of signal distribution that beat the TS threshold
        
    """
    
    # poisson_percentil Calculate upper percentile using a Poissonian distribution.
    TS_quantile = lambda mu: poisson_percentile(mu, trials["n_inj"], trials["logP"], TS_thres)[0]
    
    # function that has a deep minimum at intersection of TS_quantile and perc
    fun = lambda n: np.log10((TS_quantile(n)- perc)**2)

    
    bounds = [np.percentile(trials["n_inj"], [eps, 100.-eps])]
    
    # Brute Force for seed
    mu = np.linspace(*bounds[0], num=50)
    seed = mu[np.argmin([fun(i) for i in mu])]

    # minimize function
    x, f, info = fmin_l_bfgs_b(fun, [seed],bounds=bounds,
                               approx_grad=True)

    # calculate uncertainties on percentile
    beta, beta_err = poisson_percentile(x, trials["n_inj"], trials["logP"], TS_thres)

    # assert that fitted within tolerance
    if np.fabs(perc - beta) > 1.e-2 or beta_err > 2.5e-2:
        
        print(33*"-")
        print("ERROR", perc, beta, beta_err)
        print(bounds, seed, x)
        print(33*"-")
        raise RuntimeError("Minimizer did not work successful")
        
    return x[0], poisson_weight(trials["n_inj"], x), beta, beta_err
