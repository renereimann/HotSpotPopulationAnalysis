#!/usr/bin/env python

import os, argparse
import cPickle as pickle
import numpy as np
from data_types import LocalWarmSpotExpectation
from utils import HPA_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/extracted_background_populations/all_sky_population_bgd_trial_test.pickle"],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/HPA_TS_background_from_skylab.pickle",
                    help="Path of the output file.")
parser.add_argument("--expectation",
                    type=str,
                    default="test_data/from_poisson_test/HPA_nspot_expectation.pickle",
                    help="Give spline_path.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

# read in all sky trials
trials = []
for file_name in args.infiles:
    with open(file_name, "r") as open_file:
        temp = pickle.load(open_file)
    trials.extend(temp)
print "Read in %d trials"%len(trials)

# read in local warm spot expectation
expect = LocalWarmSpotExpectation(load_path=args.expectation)
analysis = HPA_analysis(expect)

hpa_results = []
for data in trials:
    result =  analysis.best_fit(data["pVal"])
    hpa_results.append(result)
hpa_results = np.concatenate([hpa_results])

print hpa_results[:10]
# save HPA values
with open(args.outfile, "w") as open_file:
    pickle.dump(hpa_results, open_file)
