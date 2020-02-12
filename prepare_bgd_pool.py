#!/usr/bin/env python

import os, argparse
from utils import BackgroundLocalWarmSpotPool

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/extracted_background_populations/all_sky_population_bgd_trial_test.pickle"],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/background_pool.pickle",
                    help="Path of the output file.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

bgd_pool = BackgroundLocalWarmSpotPool(seed=1234)
bgd_pool.load_trials(args.infiles)
bgd_pool.save(args.outfile)
