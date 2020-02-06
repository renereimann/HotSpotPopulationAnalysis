#!/usr/bin/env python

import numpy as np
import cPickle, os, argparse
from scipy.interpolate import UnivariateSpline
from utils import get_all_sky_trials, counts_above_pval

parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    type=str,
                    default="./test_data/extracted_background_populations/",
                    help="Give inpath.")
parser.add_argument("--outdir",
                    type=str,
                    default="test_data/from_poisson_test/",
                    help="Give outpath.")
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
parser.add_argument("--plotdir",
                    type=str,
                    required=False,
                    default = None,
                    help="Give plot path.")
args = parser.parse_args()

# read in stuff
glob_path    = os.path.join(args.indir, "all_sky_population_bgd_trials_cutoff_pVal_{args.cutoff}_seed_*X.pickle".format(**locals()))

trials = get_all_sky_trials(glob_path, min_ang_dist=args.min_ang_dist)    

# test poissonian distribution
parametrization = []
for p in np.linspace(2,7, 5*10+1):
    kwargs = {}
    if args.plotdir is not None:
        kwargs["plot_path"] = os.path.join(args.plotdir, "HSP_test_poissonian_neg_logP_{p:.2f}_min_ang_dist_{args.min_ang_dist:.2f}.png".format(**locals()))
        kwargs["plotting"] = True
    parametrization.append(counts_above_pval(trials, thres=p, **kwargs))
parametrization = np.array(parametrization)
spline = UnivariateSpline(parametrization[:,0], np.log10(parametrization[:,1]+1e-20), s=0, k=1)

# save stuff
if not os.path.exists(args.outdir): 
    raise ValueError("You try to save in a non existing directory. Create the folder : {args.outdir}".format(**locals()))
with open(os.path.join(args.outdir, "parametrization_expectiation_cutoff_pVal_{args.cutoff}_min_ang_dist_{args.min_ang_dist:.2f}.pickle".format(**locals())), "w") as open_file:
    cPickle.dump(parametrization, open_file)
with open(os.path.join(args.outdir, "spline_expectation_cutoff_pVal_{args.cutoff}_min_ang_dist_{args.min_ang_dist:.2f}.pickle".format(**locals())), "w") as open_file:
    cPickle.dump(spline, open_file)
