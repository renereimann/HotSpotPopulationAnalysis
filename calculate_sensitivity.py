#!/usr/bin/env python

import os, glob, re, argparse
import cPickle as pickle
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma, kstest
from statistics import llh2Sigma, sigma2pval
from utils import dec_range

class sensitivity_result(object):
    def __init__(self, CL=None, alpha=None, beta=None, TS_thres=None, disc_pot_like=True, label=None, source_model=None):
        # define kind of CL level
        self.alpha = alpha
        self.beta = beta
        self.TS_thres = TS_thres
        self.CL = CL
        self.disc_pot_like = disc_pot_like
        self.limit_like = not disc_pot_like
        self.label = label

        # tot injected at CL level
        self.mu = None

        # define mu -> flux conversion factor
        self.flux_per_mu = None
        self.flux_units = "1/GeV cm^2 s"
        self.flux_pivot_point = "1 GeV"

        # source mode specific
        self.source_model = source_model

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

    TS_vs_quantile = lambda quantile: gamma(*params).isf(quantile)
    p_3sigma = sigma2pval(3, oneSided=True)
    p_5sigma = sigma2pval(5, oneSided=True)
    ext_3sig = gamma(*params).isf(p_5sigma)
    ext_5sig = gamma(*params).isf(p_5sigma)

    result = {}
    result["params"] = params
    result["ks_pval"] = ks_gamma
    result["median"] = (median, (median_lower, median_upper))
    result["TS_vs_quantile"] = TS_vs_quantile

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

    version_path = "/data/ana/analyses/northern_tracks/version-002-p06/"
    season_list = [("dataset_8yr_fit_IC59_MC_compressed.npy",        "GRL/dataset_8yr_fit_IC59_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC79_MC_compressed.npy",         "GRL/dataset_8yr_fit_IC79_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC86_2011_MC_compressed.npy",    "GRL/dataset_8yr_fit_IC86_2011_exp_compressed.npy"),
                  ("dataset_8yr_fit_IC86_2012_16_MC_compressed.npy", "GRL/dataset_8yr_fit_IC86_2012_16_exp_compressed.npy"),]

    denominator = 0
    for mc_path, grl_path in enumerate(season_list):
        mc = np.load(os.path.join(version_path, mc_path))
        grl = np.load(os.path.join(version_path, grl_path))
        livetime = np.sum(grl["livetime"])

        weight = mc["ow"]*livetime*mc["trueE"]**(-np.abs(spectral_index))
        mask = np.logical_and(min(dec_range) < mc["dec"], mc["dec"] < max(dec_range))
        denominator += np.sum(weight[mask])

    solAng = 2. * np.pi * (np.sin(max(dec_range)) - np.sin(min(dec_range)))
    return solAng/denominator

def get_files_for_nsrc(nsrc, glob_path):
    files = sorted(glob.glob(glob_path))

    # first get the nsrc number of each file
    nsrc_file_list = []
    for f in files:
        # get FIRST match in a string
        number = re.search('_'+8*'[0-9]', f)
        number = int(number.group(0)[1:])
        nsrc_file_list.append((number, f))

    # select just file pathes with nsrc in path
    selected_file_list = []
    for f in nsrc_file_list:
        if f[0] == nsrc:
            selected_file_list.append(f[1])

    return np.concatenate([np.load(f) for f in selected_file_list])

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
    res = minimize(fun, x0=[seed], method="L-BFGS-B", bounds=bounds, approx_grad=True)

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
                    required=True,
                    help="Give inpath.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

# Settings
TS_val_keys        = ["median", "3sig", "5sig", "UL"]
beta               = [0.9,      0.5,    0.5,    0.9]                    # quantiles of the signal TS distribution that should beat the correspont background values
alpha              = [0.5,      "3sig", "5sig", unblind_value]
sig_trial_path     = "/data/user/reimann/2017_10/HPA/sig_trials/mc_no_astro_bgd_2_19/HPA_signal_trials_nsrc_0000*_00*.npy"
nsrc_list          = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048]) # Number of Sources for which to calculate sensitivities
nsrc
unblind_value      = 1.37962118438


# get background HPA trials
with open(args.backgroung_HPA_trials, "r") as open_file:
    backgroung_HPA_trials = pickle.load(open_file)
# fit the background distribution and extrapolate
gamma_fit = make_gamma_fit(backgroung_HPA_trials)

# Get mu -> flux factor
mu2flux = get_mu_2_flux_factor()
print "Mu to flux conversion factor:", mu2flux

# get data
sig_trials = get_files_for_nsrc(nsrc, sig_trial_path)


results = []
# loop over sensitivity and discovery potential TS
bgd_TS_vals =[gamma_fit[k][0] if k!="UL" else unblind_value for k in TS_val_keys]

def __init__(self, CL=None, alpha=None, beta=None, TS_thres=None, disc_pot_like=True, label=None):
res = sensitivity_result(TS_thres=bgd_TS_vals, beta=, flux_per_mu=mu2flux, source_model={"nsrc": nsrc})

for res in results:
    try:
        # SENSITIVITY estimation
        mu, w, b, b_err = sens_estimation(sig_trials, res.TS_thres, res.beta)
        res.mu = mu
        res.success = True
    except:
        continue

    # needed for plotting
    # nsrc sig_trials, val_i, beta_i, eps, mu, w, b, b_err

save_path = "test_data/hpa_sensitivity_and_limit_bgd_wo_astro_2_19.pickle"
