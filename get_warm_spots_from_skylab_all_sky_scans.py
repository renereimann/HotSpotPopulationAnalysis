#!/usr/bin/env python

import argparse
import os
import cPickle
import numpy as np
from utils import get_spots

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3010_seed_3010.pickle", "test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3011_seed_3011.pickle"],
                    help="List of all the input files, that should be read and processed.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/extracted_background_populations/all_sky_population_bgd_trial_test.pickle",
                    help="Path of the output file.")
parser.add_argument("--cutoff",
                    type=float,
                    required=False,
                    default=3.0,
                    help="Give the -log10(p-value) above that spots should not be considerd. Default: 3.0.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

# here we store the output
spots_trials = []

for file_name in args.infiles:
    # read in the skymap
    # this is the way, I saved the sky map. That my be modified.
    print "Processing", os.path.basename(file_name)
    with open(file_name, "r") as open_file:
        job_args, scan = cPickle.load(open_file)
    log10p_map = scan[0]["pVal"]
    dec_map = scan[0]["dec"]
    
    # mask Southern hemisphere
    mask = np.logical_or( dec_map < np.radians(-3), dec_map > np.radians(90))
    log10p_map[mask] = 0

    # get the spots and append
    spots = get_spots(log10p_map, cutoff_pval=args.cutoff)
    spots_trials.append(spots)

# write output
with open(args.outfile, "w") as open_file:
    cPickle.dump(spots_trials, open_file, protocol=2)
