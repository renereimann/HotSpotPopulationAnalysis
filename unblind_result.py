#!/usr/bin/env python

import argparse, os
import cPickle as pickle
import numpy as np
from data_types import LocalWarmSpotExpectation
from utils import HPA_analysis, dec_range
from skylab_data import SkylabAllSkyScan

parser = argparse.ArgumentParser()
parser.add_argument("--infile",
                        type=str,
                        default="test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3015_seed_3015.pickle",
                        help="Path of the skylab all sky scan that contains the unblinded sky map.")
parser.add_argument("--background_trials",
                        type=str,
                        default="test_data/HPA_TS_background_from_skylab.pickle",
                        help="Path of the background HPA TS trials file.")
parser.add_argument("--expectation",
                        type=str,
                        default="test_data/from_poisson_test/HPA_nspot_expectation.pickle",
                        help="Path of the LocalWarmSpotExpecation file.")
parser.add_argument("--outfile",
                        type=str,
                        default="test_data/unblinding_result.pickle",
                        help="Path where the unblinding results should be saved.")
parser.add_argument("--log10pVal_threshold",
                    type=float,
                    required=False,
                    default=2.0,
                    help="Give the -log10(p-value) above that spots should not be considerd. Default: 2.0.")
parser.add_argument("--min_ang_dist",
                    type=float,
                    required=False,
                    default=1.0,
                    help="Give the minimal angular distance allowed between two local warm spots. Units: degrees. Default: 1.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

scan = SkylabAllSkyScan(path=args.infile)
scan.mask_hemisphere(dec_range=dec_range)
spots = scan.get_local_warm_spots(log10p_threshold=args.log10pVal_threshold, min_ang_dist=args.min_ang_dist)

expect = LocalWarmSpotExpectation(load_path=args.expectation)
analysis = HPA_analysis(expect)
result = analysis.best_fit(spots["pVal"])

with open(args.background_trials) as open_file:
    bgd_trials = pickle.load(open_file)
hpa_ppost = np.sum(bgd_trials["hpa_ts"] > result["hpa_ts"], dtype=float) / len(bgd_trials)

result_dict = {"spots": spots, "best_fit_result": result, "pVal_post": hpa_ppost}
with open(args.outfile, "w") as open_file:
    pickle.dump(result_dict, open_file)
print result_dict
