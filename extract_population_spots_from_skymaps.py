#!/usr/bin/env python

import numpy as np
import cPickle, glob, os, argparse, healpy
from ps_analysis.hpa.utils import get_spots

parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    type=str,
                    required=True,
                    help="Give inpath.")
parser.add_argument("--outdir",
                    type=str,
                    required=True,
                    help="Give outpath.")
parser.add_argument("--number",
                    type=int,
                    required=True,
                    help="Give the digits except the last of seed numbers.")
parser.add_argument("--cutoff",
                    type=float,
                    required=False,
                    default=3.0,
                    help="Give the -log10(p-value) above that spots should not be considerd. Default: 3.0.")
args = parser.parse_args()

if args.number == 0:
    files = sorted(glob.glob(os.path.join(args.indir, "all_sky_scan_trial_iter2_*_seed_?.pickle")))
else:
    files = sorted(glob.glob(os.path.join(args.indir, "all_sky_scan_trial_iter2_*_seed_{}?.pickle".format(args.number))))
    
spots_trials = []
for fName in files:
    # get skymap
    print "Now do", os.path.basename(fName)
    with open(fName, "r") as open_file:
        job_args, scan = cPickle.load(open_file)
    pvalues = scan[0]["pVal"]
    
    # mask Southern hemisphere
    mask = np.logical_or( scan[0]["dec"] < np.radians(-3), scan[0]["dec"] > np.radians(90))
    pvalues[mask] = 0

    # get the spots and append
    spots = get_spots(pvalues, cutoff_pval=args.cutoff)
    spots_trials.append(spots)

# write output
with open(os.path.join(args.outdir, "all_sky_population_bgd_trials_cutoff_pVal_{args.cutoff}_seed_{args.number}X.pickle".format(**locals())), "w") as open_file:
    cPickle.dump(spots_trials, open_file, protocol=2)
