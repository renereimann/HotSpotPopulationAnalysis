#!/usr/bin/env python

from data_types import LocalWarmSpotExpectation
from utils import HPA_analysis
from skylab_data import SkylabAllSkyScan
import cPickle
import numpy as np

all_sky_scan_path = "test_data/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_seed_unblind.pickle"
expectation_path = "test_data/from_poisson_test/HPA_nspot_expectation.pickle"
bgd_trials_path = "test_data/HPA_TS_background_from_skylab.pickle"
save_path = "test_data/unblinding_result.pickle"

scan = SkylabAllSkyScan(path=all_sky_scan_path)
scan.mask_hemisphere(dec_range=[np.radians(-3), np.radians(90)])
spots = scan.get_local_warm_spots(log10p_threshold=2.0, min_ang_dist=1.0)

expect = LocalWarmSpotExpectation(expectation_path)
analysis = HPA_analysis(expect)
result = analysis.best_fit(spots["pVal"])

with open(bgd_trials_path) as open_file:
    bgd_trials = cPickle.load(open_file)
hpa_ppost = np.sum(bgd_trials["hpa_ts"] > result["hpa_ts"], dtype=float) / len(bgd_trials)

result_dict = {"spots": spots, "best_fit_result": result, "pVal_post": hpa_ppost}
with open(, "w") as open_file:
    cPickle.dump(result_dict, open_file)
print result_dict
