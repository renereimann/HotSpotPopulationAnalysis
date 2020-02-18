#!/usr/bin/env python

import os, glob, re, argparse
import cPickle as pickle
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma, kstest
from statistics import llh2Sigma, sigma2pval, poisson_percentile
from utils import dec_range

class sensitivity_result(object):
    def __init__(self, TS_thres=None, alpha=None, beta=None, label=None):
        # define kind of CL level
        self.TS_thres = TS_thres
        self.alpha = alpha
        self.beta = beta
        self.label = label

        # tot injected at CL level
        self.tot_n_inj = None
        self.success = False

        # define tot_n_inj -> flux conversion factor
        self.flux_per_mu = None
        self.flux_units = "1/GeV cm^2 s"
        self.flux_pivot_point = "1 GeV"

        # source mode specific
        self.source_model = None

    def __str__(self):
        return "Sensitivity Result: "+str(self.__dict__)

    @property
    def disc_pot_like(self):
        return self.beta == 0.5

    @property
    def limit_like(self):
        return not self.disc_pot_like

    @property
    def CL(self):
        if self.disc_pot_like:
            return self.alpha
        return self.beta

def make_gamma_fit(trials, scan_likelihood=[16,16], verbose=False):
    r"""Fits a Gamma Distribution to the HPA-TS values. The Goodness of fit
    is calculated as well as the median (and its errors) are calculated.
    If scan_likelihood is given a likelihood scan is used and the 3 and 5 sigma
    quantiles of the distribution are extrapolated with error.

    Parameters:
    * trials: array_like
        List of HPA-TS values from the background distribution.
    * scan_likelihood: list
        list of two numbers, that give the dimensions of the scan. If None, the scan is skipped.
    * verbose: bool, default=False
        some verbose printing

    Returns:
        dict - The dict containse the calculated best fit parameters, GoF, Median and LLH scan
    """

    # do a fit with a gamma function
    params = gamma.fit(trials, floc=0)

    # check the goodness of fit
    ks_gamma = kstest(trials, gamma(*params).cdf, alternative="greater")[1]

    # calculate the median of the distribution
    # errors are estimated due to binomial statistics
    trials.sort()
    median = np.median(trials)
    median_lower = trials[int(len(trials)/2.-np.sqrt(len(trials))*0.5)]
    median_upper = trials[int(len(trials)/2.+np.sqrt(len(trials))*0.5)+1]

    result = {}
    result["params"] = params
    result["ks_pval"] = ks_gamma
    result["median"] = (median, (median_lower, median_upper))
    result["TS_vs_quantile"] = lambda quantile: gamma(*params).isf(quantile)

    if verbose:
        print "Fit parameters:", params
        print "KS-Test between fit and sample", ks_gamma
        print "Median at {:.2f} + {:.2f} - {:.2f}".format(median, median-median_lower, median_upper-median)

    if scan_likelihood is not None:
        # calculate LLH landscape
        x = np.linspace(params[0]-0.3, params[0]+0.3, scan_likelihood[0])
        y = np.linspace(params[2]-0.1, params[2]+0.1, scan_likelihood[1])
        xv, yv = np.meshgrid(x, y)
        llh = [-np.sum(gamma(x1, 0, x2).logpdf(trials)) for x1, x2 in zip(xv.flatten(), yv.flatten())]
        dllh = np.reshape(llh, np.shape(xv))-np.min(llh)
        xmin = xv.flatten()[np.argmin(dllh)]
        ymin = yv.flatten()[np.argmin(dllh)]
        sigma = llh2Sigma(dllh, dof=2, alreadyTimes2=False, oneSided=False)

        result["scan_best_fit_pixel"] = (xmin, ymin)
        result["scan_x"] = xv
        result["scan_y"] = yv
        result["scan_sigma"] = sigma

        # 1 sigma error on the extrapolation
        ma = sigma < 1
        TS_vs_quantile_lower = lambda quantile: np.min([gamma(xx, 0, yy).isf(quantile) for xx, yy in zip(xv[ma], yv[ma])])
        TS_vs_quantile_upper = lambda quantile: np.max([gamma(xx, 0, yy).isf(quantile) for xx, yy in zip(xv[ma], yv[ma])])
        result["TS_vs_quantile_lower"] = TS_vs_quantile_lower
        result["TS_vs_quantile_upper"] = TS_vs_quantile_upper
    return result

def get_mu_2_flux_factor(dec_range=dec_range, spectral_index=2.):
    """ Calculates the conversion factor from mu to flux for a given spectral index and sample.

    Parameters
    ----------
    season: string
        Not used. Remove any dependency on that. Just keep it for now so that we do not break everything.
    sinHem: float, default sin(5 deg)
        sin of the declination of the boarder of the hemisphere considered
    spectral_index: float, default 2
        spectral index of the power law flux that is considered.

    Returns
    -------
    float
        mu -> flux conversion factor
    """

    version_path = "/home/rene/Desktop/version-002-p06/"
    season_list = [("dataset_8yr_fit_IC59_MC_compressed.npy",        "GRL/dataset_8yr_fit_IC59_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC79_MC_compressed.npy",         "GRL/dataset_8yr_fit_IC79_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC86_2011_MC_compressed.npy",    "GRL/dataset_8yr_fit_IC86_2011_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC86_2012_16_MC_compressed.npy", "GRL/dataset_8yr_fit_IC86_2012_16_exp_compressed.npy"),]

    denominator = 0
    for mc_path, grl_path in season_list:
        mc = np.load(os.path.join(version_path, mc_path))
        grl = np.load(os.path.join(version_path, grl_path))
        livetime = np.sum(grl["livetime"])

        weight = mc["ow"]*livetime*mc["trueE"]**(-np.abs(spectral_index))
        mask = np.logical_and(min(dec_range) < mc["dec"], mc["dec"] < max(dec_range))
        denominator += np.sum(weight[mask])

    solAng = 2. * np.pi * (np.sin(max(dec_range)) - np.sin(min(dec_range)))
    return solAng/denominator

def sens_estimation(trials, TS_thres, perc, eps=2.5):
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
    res = minimize(fun, x0=[seed], method="L-BFGS-B", bounds=bounds)#, approx_grad=True)

    # calculate uncertainties on percentile
    beta, beta_err = poisson_percentile(res["x"], trials["n_inj"], trials["logP"], TS_thres)

    # assert that fitted within tolerance
    if np.fabs(perc - beta) > 1.e-2 or beta_err > 2.5e-2:

        print(33*"-")
        print("ERROR", perc, beta, beta_err)
        print(bounds, seed, res["x"])
        print(33*"-")
        raise RuntimeError("Minimizer did not work successful")

    return res["x"][0], poisson_weight(trials["n_inj"], res["x"]), beta, beta_err

parser = argparse.ArgumentParser()
parser.add_argument("--backgroung_HPA_trials",
                    type=str,
                    #default="test_data/HPA_TS_background_from_skylab.pickle",
                    default="test_data/max_local_pVal_min_ang_dist_1.00_min_thres_2.00.pickle",
                    help="Give inpath.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/sensitivity_test_nsrc_256.pickle",
                    help="Give inpath.")
parser.add_argument("--signal_files",
                    type=str,
                    nargs='+',
                    default=["test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000000.npy", "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000001.npy",
                             "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000002.npy", "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000003.npy",
                             "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000004.npy", "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000005.npy",
                             "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000006.npy", "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000007.npy",
                             "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000008.npy", "test_data/HPA_signal_trials_nsrc_00000256_nsrcIdx_00000009.npy",],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--unblinded_value",
                    type=float,
                    default=None, # 1.379
                    help="If you have unblinded give the TS value here and upper limits are calculated.")
args, source_model_args = parser.parse_known_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

# get background HPA trials
with open(args.backgroung_HPA_trials, "r") as open_file:
    backgroung_HPA_trials = pickle.load(open_file)
# fit the background distribution and extrapolate
#gamma_fit = make_gamma_fit(backgroung_HPA_trials["hpa_ts"])
gamma_fit = make_gamma_fit(backgroung_HPA_trials)

# Get mu -> flux factor
mu2flux = get_mu_2_flux_factor()
print "Flux per mu in 1/GeV cm^2 s @ 1 GeV:", mu2flux

# Settings
sig_trials = np.concatenate([np.load(f) for f in args.signal_files])
source_model = source_model_args
print "Source Model contains", source_model
print

# define the quantities to calculate
results = [sensitivity_result(alpha=0.5,                          beta=0.9, label="sens"),
           sensitivity_result(alpha=sigma2pval(3, oneSided=True), beta=0.5, label="disc pot 3sig"),
           sensitivity_result(alpha=sigma2pval(5, oneSided=True), beta=0.5, label="disc pot 5sig")]
if args.unblinded_value is not None:
    results.append(sensitivity_result(TS_thres=args.unblinded_value, beta=0.9, label="UL"))

# loop over sensitivity and discovery potential TS
for res in results:
    res.source_model = source_model
    res.flux_per_mu = mu2flux
    # if TS-threshold is not given, we get it from the background distribution
    if res.TS_thres is None:
        if res.alpha == 0.5:
            res.TS_thres = gamma_fit["median"][0]
        else:
            res.TS_thres = gamma_fit["TS_vs_quantile"](res.alpha)
    try:
        # SENSITIVITY estimation
        tot_n_inj, w, b, b_err = sens_estimation(sig_trials, res.TS_thres, res.beta)
        # needed for plotting
        # nsrc sig_trials, w, b, b_err
        res.tot_n_inj = tot_n_inj
        res.success = True
    except Exception as e:
        print e
        continue

print
for res in results:
    print res

with open(args.outfile, "w") as open_file:
    pickle.dump(results, open_file, protocol=2)
