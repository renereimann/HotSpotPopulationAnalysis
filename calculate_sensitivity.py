#!/usr/bin/env python

# This script uses background trials and signal trials to calculate sensitivity and discovery potential for the HPA.

import os, cPickle, glob, re, argparse 
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ps_analysis.plotting.hpa import ninj_vs_logP_plots, find_mu_plot, TS_hist_plot, histogram_observed_vs_expected, histogram_plocal_vs_ppost

        
def get_mu_2_flux_factor(season, sinHem=np.sin(np.radians(-5)), spectral_index=2.):
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
        mask = mc["dec"] > np.arcsin(sinHem)
        denominator += np.sum(weight[mask])

    solAng = 2. * np.pi * (1. - sinHem)
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
        with open(f, "r") as open_file:    
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
    x, f, info = fmin_l_bfgs_b(fun, [seed], bounds=bounds, approx_grad=True)

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


# Settings

gamma_fit_file     = "/data/user/reimann/2017_10/HPA/bgd_trials/mc_trials/gamma_fit_min_ang_dist_1.00_min_thres_2.00.pickle"
TS_val_keys        = ["median", "3sig", "5sig", "UL"]
beta               = [0.9,      0.5,    0.5,    0.9]                    # quantiles of the signal TS distribution that should beat the correspont background values
single_sens_folder = "/data/user/reimann/2017_10/sensitivity/mc_trials_fixed_negTS/E-2.0/sindec_bandwidth_1deg/"
sig_trial_path     = "/data/user/reimann/2017_10/HPA/sig_trials/mc_no_astro_bgd_2_19/HPA_signal_trials_nsrc_0000*_00*.npy"
nsrc_list          = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048])  # Number of Sources for which to calculate sensitivities
eps                = 2.5                                             # tolerance of fitter
save_path          = "/data/user/reimann/2017_10/HPA/hpa_sensitivity_and_limit_bgd_wo_astro_2_19.cPickle"
plotting           = False

# Get the TS values to beat from background fit

with open(gamma_fit_file, "r") as open_file:
    gamma_fit = cPickle.load(open_file)
bgd_TS_vals =[gamma_fit[k][0] if k!="UL" else 1.37962118438 for k in TS_val_keys]
print bgd_TS_vals

# Get mu -> flux factor 

mu2flux = get_mu2flux_from_sens_files(single_sens_folder)
print "Mu to flux conversion factor:", mu2flux



flux = np.zeros((len(nsrc_list), len(bgd_TS_vals)))
for i, nsrc in enumerate(nsrc_list):
    if plotting:
        # init plotting
        ninj_vs_logP_plot = ninj_vs_logP_plots(nsrc, sig_trials)
        find_mu = find_mu_plot(nsrc)
        TS_hist = TS_hist_plot(nsrc)
    
    # get data
    sig_trials = get_data_for_nsrc(nsrc, sig_trial_path)
    
    # loop over sensitivity and discovery potential TS
    for j, (val_i, beta_i) in enumerate(zip(bgd_TS_vals, beta)):
        print nsrc, val_i
        try:
            # SENSITIVITY estimation
            mu, w, b, b_err = sens_estimation(sig_trials, val_i, beta_i, eps)    
        except Exception as e:
            print nsrc, e
            continue
        
        flux[i,j] = mu * mu2flux
        
        if plotting:
            ninj_vs_logP_plot.add_threshold(val_i)
            find_mu.add_mu_profile(sig_trials, val_i, mu, eps, beta_i)
            TS_hist.add_hist_for_threshold(sig_trials, w, b, b_err, val_i, mu)
            histogram_observed_vs_expected(nsrc, sig_trials, w, mu).plot()
            histogram_plocal_vs_ppost(nsrc, sig_trials, w, mu).plot()
    
    if plotting:
        # plot plots
        ninj_vs_logP_plot.plot()
        find_mu.plot()
        TS_hist.plot()

# make the flux per source
for i in range(np.shape(flux)[1]):
    flux[:, i] /= nsrc_list

# convert flux from GeV in TeV
flux *= 1.e-3

print flux

# put everything together
my_result = {"nsrc": nsrc_list, "sens":flux[:, 0], "3sig":flux[:, 1], "5sig":flux[:,2], "UL": flux[:,3]}

with open(save_path, "w") as open_file:
    cPickle.dump(my_result, open_file)
