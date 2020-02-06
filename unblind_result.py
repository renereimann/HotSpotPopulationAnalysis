#!/usr/bin/env python

from ps_analysis.hpa.utils import expectation, get_spots
from ps_analysis.scripts.calc_tools import deltaPsi
import cPickle
import numpy as np


with open("/data/user/reimann/2017_10/all_sky_trials/unblind/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_seed_unblind.pickle", "r") as open_file:
    job_args, scan = cPickle.load(open_file)
# clean skymap out of range
mask = np.logical_and( scan[0]["dec"] > np.radians(-3), scan[0]["dec"] <= np.radians(90))
scan[0]["pVal"][~mask] = 0

expect = expectation("/data/user/reimann/2017_10/HPA/check_poissonian/mc_trials/spline_expectation_cutoff_pVal_2.0_min_ang_dist_1.00.pickle")
with open("/data/user/reimann/2017_10/HPA/bgd_trials/mc_trials/max_local_pVal_min_ang_dist_1.00_min_thres_2.00.pickle") as open_file:
    hpa_trials_mc = cPickle.load(open_file)
with open("/data/user/reimann/2017_10/HPA/bgd_trials/exp_trials/max_local_pVal_min_ang_dist_1.00_min_thres_2.00.pickle") as open_file:
    hpa_trials_exp = cPickle.load(open_file)
    
# get the spots 
spots = get_spots(scan[0]["pVal"], cutoff_pval=2.0)

# clear close by sources
remove = []
for i in np.arange(0, len(spots)):
    temp_dist = np.degrees(deltaPsi(np.pi/2.-spots[i]["theta"], spots[i]["phi"], np.pi/2.-spots[i+1:]["theta"], spots[i+1:]["phi"]))
    m = np.where(temp_dist < 1.0)[0]
    if len(m) == 0: continue
    # we have at least 2 points closer than 1 deg
    if any(spots[m+i+1]["pVal"] >= spots[i]["pVal"]) or np.isnan(spots[i]["pVal"]): remove.append(i)
    else:
        print spots[m+i+1]["pVal"]
        print spots[i]["pVal"]
        raise ValueError("Should never happen because spots should be sorted by pValue")
spots = spots[~np.in1d(range(len(spots)), remove)]

# result
HPA_TS, hpa_pthres, hpa_n_observed, hpa_n_expected = expect.poisson_test(spots["pVal"])
 
hpa_ppost_exp = np.sum(hpa_trials_exp > HPA_TS, dtype=float) / len(hpa_trials_exp)
hpa_ppost_mc = np.sum(hpa_trials_mc > HPA_TS, dtype=float) / len(hpa_trials_mc)

print HPA_TS, hpa_pthres, hpa_n_observed, hpa_n_expected, hpa_ppost_exp, hpa_ppost_mc

result_dict = {"TS": HPA_TS, "p_thres": hpa_pthres, "#_expected": hpa_n_expected, "#_observed": hpa_n_observed, "pVal_post_exp": hpa_ppost_exp, "pVal_post_mc": hpa_ppost_mc, "spots": spots}
with open("/home/reimann/software/python-modules/ps_analysis/hpa/unblinding_result.cPickle", "w") as open_file:
    cPickle.dump(result_dict, open_file)
