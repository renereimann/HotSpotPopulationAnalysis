#!/usr/bin/env python

import numpy as np
import cPickle, os, argparse
from scipy.interpolate import UnivariateSpline
from data_types import LocalWarmSpotExpectation

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/extracted_background_populations/all_sky_population_bgd_trial_test.pickle"],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/from_poisson_test/HPA_nspot_expectation.pickle",
                    help="Path of the output folder.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

# read in stuff
trials = []
for file_name in args.infiles:
    with open(file_name, "r") as open_file:
        temp = cPickle.load(open_file)
    trials.extend(temp)
print "Read in %d trials"%len(trials)

expectation = LocalWarmSpotExpectation()
expectation.generate(trials, np.linspace(2,7, 5*10+1))
expectation.save(args.outfile)
