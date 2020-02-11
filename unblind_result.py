#!/usr/bin/env python

from ps_analysis.hpa.utils import expectation
from skylab_data import SkylabAllSkyScan
import cPickle
import numpy as np

file_name = "test_data/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_seed_unblind.pickle"
scan = SkylabAllSkyScan(path=file_name)
scan.mask_hemisphere(dec_range=[np.radians(-3), np.radians(90)])
spots = scan.get_local_warm_spots(log10p_threshold=2.0, min_ang_dist=1.0)

expect = expectation("test_data/from_poisson_test/spline_expectation_cutoff_pVal_2.0_min_ang_dist_1.00.pickle")
with open("/data/user/reimann/2017_10/HPA/bgd_trials/mc_trials/max_local_pVal_min_ang_dist_1.00_min_thres_2.00.pickle") as open_file:
    hpa_trials_mc = cPickle.load(open_file)
with open("/data/user/reimann/2017_10/HPA/bgd_trials/exp_trials/max_local_pVal_min_ang_dist_1.00_min_thres_2.00.pickle") as open_file:
    hpa_trials_exp = cPickle.load(open_file)

# result
HPA_TS, hpa_pthres, hpa_n_observed, hpa_n_expected = expect.poisson_test(spots["pVal"])

hpa_ppost_exp = np.sum(hpa_trials_exp > HPA_TS, dtype=float) / len(hpa_trials_exp)
hpa_ppost_mc = np.sum(hpa_trials_mc > HPA_TS, dtype=float) / len(hpa_trials_mc)

print HPA_TS, hpa_pthres, hpa_n_observed, hpa_n_expected, hpa_ppost_exp, hpa_ppost_mc

result_dict = {"TS": HPA_TS, "p_thres": hpa_pthres, "#_expected": hpa_n_expected, "#_observed": hpa_n_observed, "pVal_post_exp": hpa_ppost_exp, "pVal_post_mc": hpa_ppost_mc, "spots": spots}
with open("/home/reimann/software/python-modules/ps_analysis/hpa/unblinding_result.cPickle", "w") as open_file:
    cPickle.dump(result_dict, open_file)
