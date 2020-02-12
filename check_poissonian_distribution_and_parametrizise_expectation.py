#!/usr/bin/env python

import numpy as np
import cPickle, os, argparse
from scipy.interpolate import UnivariateSpline

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
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

def counts_above_pval(trials, thres, verbose=False, naive_expectation=None):
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

    if verbose:
        print "-log10(p):", thres
        print "Mean:", mean
        print "KS-Test (Poisson), p-val:", ks_poisson
        print "KS-Test (Binomial), p-val:", ks_binom

    return thres, mean, ks_poisson, ks_binom

# read in stuff
trials = []
for file_name in args.infiles:
    with open(file_name, "r") as open_file:
        temp = cPickle.load(open_file)
    trials.extend(temp)
print "Read in %d files"%len(trials)

# test poissonian distribution
parametrization = []
for p in np.linspace(2,7, 5*10+1):
    parametrization.append(counts_above_pval(trials, thres=p))
parametrization = np.array(parametrization)
spline = UnivariateSpline(parametrization[:,0], np.log10(parametrization[:,1]+1e-20), s=0, k=1)

# save stuff
if not os.path.exists(args.outdir):
    raise ValueError("You try to save in a non existing directory. Create the folder : {args.outdir}".format(**locals()))
with open(os.path.join(args.outdir, "parametrization_expectiation_cutoff_pVal_{args.cutoff}_min_ang_dist_{args.min_ang_dist:.2f}.pickle".format(**locals())), "w") as open_file:
    cPickle.dump(parametrization, open_file)
with open(os.path.join(args.outdir, "spline_expectation_cutoff_pVal_{args.cutoff}_min_ang_dist_{args.min_ang_dist:.2f}.pickle".format(**locals())), "w") as open_file:
    cPickle.dump(spline, open_file)
