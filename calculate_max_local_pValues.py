#!/usr/bin/env python

import os, cPickle, argparse
import numpy as np
from ps_analysis.hpa.utils import expectation

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/extracted_background_populations/.pickle", "test_data/extracted_background_populations/.pickle"],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--outdir",
                    type=str,
                    required=True,
                    help="Give outpath.")
parser.add_argument("--expectation",
                    type=str,
                    required=True,
                    help="Give spline_path.")
parser.add_argument("--cutoff",
                    type=float,
                    required=False,
                    default=3.0,
                    help="Give the -log10(p-value) above that spots should not be considerd. Default: 3.0.")
parser.add_argument("--min_ang_dist",
                    type=float,
                    required=False,
                    default=1.0,
                    help="Give the digits except the last of seed numbers.")
args = parser.parse_args()

expect = expectation(args.expectation)

trials = []
for file_name in args.infiles:
    with open(file_name, "r") as open_file:
        temp = cPickle.load(open_file)
    trials.extend(temp)
print "Read in %d files"%len(trials)

for min_thres in np.linspace(args.cutoff, 4., int((4.-args.cutoff)*10)+1):

    trial_correction = np.zeros(len(trials))
    for i, t in enumerate(trials):
        data = t["pVal"][t["pVal"] >= min_thres]
        trial_correction[i] = expect.poisson_prob(data)

    save_path = os.path.join(args.outdir, "max_local_pVal_min_ang_dist_{args.min_ang_dist:.2f}_min_thres_{min_thres:.2f}.pickle".format( **locals() ) )
    with open(save_path, "w") as save_file:
        cPickle.dump(trial_correction, save_file)
