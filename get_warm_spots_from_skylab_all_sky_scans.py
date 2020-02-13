#!/usr/bin/env python

import argparse
import os
import cPickle
import numpy as np
from skylab_data import SkylabAllSkyScan

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3010_seed_3010.pickle", "test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3011_seed_3011.pickle"],
                    help="List of input files. Input files should contain the healpy maps of a single all sky scan.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/extracted_background_populations/all_sky_population_bgd_trial_test.pickle",
                    help="Path of the output file.")
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

# here we store the output
spots_trials = []
for file_name in args.infiles:
    print "Processing", os.path.basename(file_name)
    scan = SkylabAllSkyScan(path=file_name)
    scan.mask_hemisphere(dec_range=[np.radians(-3), np.radians(90)])
    spots = scan.get_local_warm_spots(log10p_threshold=args.log10pVal_threshold,
                                      min_ang_dist=args.min_ang_dist)
    spots_trials.append(spots)

# write output
with open(args.outfile, "w") as open_file:
    cPickle.dump(spots_trials, open_file, protocol=2)
