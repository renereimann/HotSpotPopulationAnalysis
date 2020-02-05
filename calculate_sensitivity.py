#!/usr/bin/env python

# This script uses background trials and signal trials to calculate sensitivity and discovery potential for the HPA.

from __future__ import print_function
import os, cPickle, glob, re, argparse 
import numpy as np

from ps_analysis.hpa.utils import get_mu2flux_from_sens_files, get_data_for_nsrc, sens_estimation
from ps_analysis.plotting.hpa import ninj_vs_logP_plots, find_mu_plot, TS_hist_plot, histogram_observed_vs_expected, histogram_plocal_vs_ppost

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
print(bgd_TS_vals)

# Get mu -> flux factor 

mu2flux = get_mu2flux_from_sens_files(single_sens_folder)
print("Mu to flux conversion factor:", mu2flux)



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
        print(nsrc, val_i)
        try:
            # SENSITIVITY estimation
            mu, w, b, b_err = sens_estimation(sig_trials, val_i, beta_i, eps)    
        except Exception as e:
            print(nsrc, e)
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

print(flux)

# put everything together
my_result = {"nsrc": nsrc_list, "sens":flux[:, 0], "3sig":flux[:, 1], "5sig":flux[:,2], "UL": flux[:,3]}

with open(save_path, "w") as open_file:
    cPickle.dump(my_result, open_file)
