#!/usr/bin/env python

import os, cPickle, argparse
import numpy as np
from ps_analysis.hpa.utils import get_all_sky_trials, expectation 

parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    type=str,
                    required=True,
                    help="Give inpath.")
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
        
glob_path    = os.path.join(args.indir, "all_sky_population_bgd_trials_cutoff_pVal_{args.cutoff}_seed_*X.pickle".format(**locals()))
trials = get_all_sky_trials(glob_path, min_ang_dist=args.min_ang_dist) 

for min_thres in np.linspace(args.cutoff, 4., int((4.-args.cutoff)*10)+1):

    trial_correction = np.zeros(len(trials))
    for i, t in enumerate(trials):
        data = t["pVal"][t["pVal"] >= min_thres]
        trial_correction[i] = expect.poisson_prob(data)
                                              
    save_path = os.path.join(args.outdir, "max_local_pVal_min_ang_dist_{args.min_ang_dist:.2f}_min_thres_{min_thres:.2f}.pickle".format( **locals() ) )
    with open(save_path, "w") as save_file:
        cPickle.dump(trial_correction, save_file)
