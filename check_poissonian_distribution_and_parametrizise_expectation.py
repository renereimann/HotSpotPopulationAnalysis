#!/usr/bin/env python

import numpy as np
import cPickle, os, argparse
from scipy.interpolate import UnivariateSpline
from utils import get_all_sky_trials, counts_above_pval

parser = argparse.ArgumentParser()
parser.add_argument("--infiles",
                    type=str,
                    nargs='+',
                    default=["test_data/extracted_background_populations/.pickle", "test_data/extracted_background_populations/.pickle"],
                    help="List of input files. Input files should contain a list of extracted background populations.")
parser.add_argument("--outdir",
                    type=str,
                    default="test_data/from_poisson_test/",
                    help="Path of the output folder.")
parser.add_argument("--cutoff",
                    type=float,
                    required=False,
                    default=3.0,
                    help="Give the -log10(p-value) above that spots should not be considerd. Default: 3.0.")
parser.add_argument("--min_ang_dist",
                    type=float,
                    required=False,
                    default=1.0,
                    help="Give the minimal angular distance allowed between two local warm spots. Units: degrees. Default: 1.")
parser.add_argument("--plotdir",
                    type=str,
                    required=False,
                    default = None,
                    help="Give plot path.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

def counts_above_pval(trials, thres, plotting=False, verbose=False, plot_path=None, naive_expectation=None):
    """ Count the number of spots above a certain pValue threshold.
    You have to give the 'trials' and the threshold pValue as 'pVal'.
    Beside a fit of the poissonian also the ks-probability for a poissonian to data is performed.

    If plotting is True we will produce a histogram of number of trials together with fitted Poissonian.
    If plot_path is not None this plot is saved at that path and will not be shown otherwise it is send to display.
    If verbose is True you get a lot of print outs with different parameted.
    If naive_expectation is not None you have to give the number of effective trials. KS-Tests will be tested vs naive_expectations.
    Naive expectation is: N_expected = N_eff_trials*pValue
    """
    
    counts_above = [np.sum(t["pVal"] > thres) for t in trials]
    mean = np.mean(counts_above)
    if not naive_expectation is None:
        mean = 10**(-thres)*naive_expectation 
    ks_poisson = kstest(counts_above, poisson(mean).cdf, alternative="greater")[1]
    N_trials = mean/np.power(10, -thres)
    ks_binom   = kstest(counts_above,  binom(int(N_trials), np.power(10, -thres)).cdf, alternative="greater")[1]
    
    if plotting and (np.max(counts_above)-np.min(counts_above)) > 0 :
        curr_plot = counts_above_plot(counts_above, thres)
        curr_plot.plot(savepath=plot_path)
    
    if verbose:
        print "-log10(p):", thres
        print "Mean:", mean
        print "KS-Test (Poisson), p-val:", ks_poisson
        print "KS-Test (Binomial), p-val:", ks_binom
    
    return thres, mean, ks_poisson, ks_binom

# read in stuff
trials = get_all_sky_trials(args.infiles, min_ang_dist=args.min_ang_dist)    

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
