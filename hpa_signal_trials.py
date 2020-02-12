#!/usr/bin/env python

import cPickle, os, argparse
import numpy as np
from ps_analysis.hpa.utils import expectation, BackgroundLocalWarmSpotPool, signal_pool, signal_trials

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_id",
                    type=int,
                    help="Job config_id to perform on")
parser.add_argument("--infile_signal",
                        type=str,
                        required=True,
                        help="Give indir for signal signal.")
parser.add_argument("--infile_background",
                        type=str,
                        required=True,
                        help="Give indir for background.")
parser.add_argument("--expectation",
                        type=str,
                        required=True,
                        help="Give path to expectation spline")
parser.add_argument("--outdir",
                        type=str,
                        required=True,
                        help="Give outpath.")
parser.add_argument("--n_iter",
                    type=int,
                    default=30000,
                    help="Number of iteration to perform for these settings.")
parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Number of iteration to perform for these settings.")
args = parser.parse_args()

for k, v in args._get_kwargs():
    print "%15s"%k, v

def job_num_2_nsrc_ninj_nsrcIdx(job_num):
    """ Convert a job number to number of sources
    and number of injected events"""

    k = 2
    j, i = divmod(job_num // k, 50)
    nsrc = 2**j
    n_inj = float((i + 1)**2) / nsrc

    return nsrc, n_inj, i

nsrc, n_inj, nsrcIdx = job_num_2_nsrc_ninj_nsrcIdx(args.config_id)
print("Job number: {0:d}".format(args.config_id))
print("Number of sources for estimation: {0:d}".format(nsrc))
print("Mean number of injected events per source: {0:.3f}".format(n_inj))
print(33*"-")

RNG = np.random.RandomState(args.seed)
seed_bgd_pool, seed_sig_pool = RNG.randint(0, np.iinfo(np.uint32).max, size=2)

# read expectation spline
print("Load expectation ...")
expect = expectation(args.expectation)

# get background pool
print("Load background pool ...")
bgd_pool = BackgroundLocalWarmSpotPool()
bgd_pool.load( args.infile_background, seed=seed_bgd_pool)

# get signal pool
print("Load signal pool ...")
sig_pool = signal_pool()
sig_pool.load(args.infile_signal, seed=seed_sig_pool)
sig_pool.get_signal_weights(n_inj)

#### start generating stuff
print("Start generating trials ...")
out = signal_trials(args.n_iter)
while out.need_more_trials():
    bgd            = bgd_pool.get_pseudo_experiment()
    sig, n_tot_inj = sig_pool.get_pseudo_experiment(nsrc)

    # combine and remove lowest nsrc trials
    # Threshold cut, no pValues below  min_thres
    data = np.sort(np.append(bgd, sig))[nsrc:]
    data = data[data >= bgd_pool.cutoff]

    # local pValue calculation and give back maximum significant pValue
    pseudo_result = expect.poisson_test(data)

    out.add_trial(n_tot_inj, pseudo_result)
print("... done.")
out.clean_nans()

# save stuff
# srcIdx can be converted to n_inj but n_inj is a float and not so nice for file name
print("Save results")
with os(os.path.join(args.outdir,"HPA_signal_trials_nsrc_{nsrc:08d}_nsrcIdx_{nsrcIdx:08d}.npy".format(**locals())), "w") as open_file:
    np.save(open_file, out.trials)
with os(os.path.join(args.outdir,"HPA_signal_trials_nsrc_{nsrc:08d}_nsrcIdx_{nsrcIdx:08d}.args".format(**locals())), "w") as open_file:
    cPickle.dump(args, open_file)
