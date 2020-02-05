#!/usr/bin/env python

import matplotlib
matplotlib.use("agg")
import os, cPickle, argparse
from ps_analysis.hpa.utils import make_gamma_fit 

parser = argparse.ArgumentParser()
parser.add_argument("--infile",
                    type=str,
                    required=True,
                    help="Give inpath.")
parser.add_argument("--outdir",
                    type=str,
                    required=True,
                    help="Give outpath.")
parser.add_argument("--plotdir",
                    type=str,
                    required=False,
                    default=None,
                    help="Give plot dir.")
args = parser.parse_args()
 
post_fix = os.path.basename(args.infile).replace("max_local_pVal_", "").replace(".pickle", "") 
kwargs = {}
if args.plotdir is not None:
    kwargs["plot_hist"] = True
    kwargs["plot_path_hist"] = os.path.join(args.plotdir, "max_pVal_hist_{post_fix}.png".format(**locals()))
    kwargs["plot_contour"] = True
    kwargs["plot_path_contour"] = os.path.join(args.plotdir, "fit_llh_lands_{post_fix}.png".format(**locals()))
    kwargs["plot_survival"] = True
    kwargs["plot_path_survival"] = os.path.join(args.plotdir, "extrapolation_{post_fix}.png".format(**locals()))
 
fit_stuff = make_gamma_fit(args.infile, **kwargs)

# save stuff     
with open(os.path.join(args.outdir, "gamma_fit_{post_fix}.pickle".format(**locals())) , "w") as open_file:
    cPickle.dump(fit_stuff, open_file)
