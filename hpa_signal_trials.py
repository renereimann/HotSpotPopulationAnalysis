#!/usr/bin/env python

import cPickle, os, argparse, time
import numpy as np
from utils import HPA_analysis, BackgroundLocalWarmSpotPool, SingleSpotTrialPool, SignalSimulation, signal_trials
from data_types import LocalWarmSpotExpectation
from source_count_dist import SourceCountDistEqualFluxAtEarth, SourceCountDistFIRESONG

# get arguments
parser = argparse.ArgumentParser()
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
parser.add_argument("--nsrc",
                    type=int,
                    help="Job config_id to perform on")
parser.add_argument("--n_inj",
                    type=int,
                    help="Job config_id to perform on")
parser.add_argument("--infile_firesong",
                    type=str,
                    required=True,
                    help="Input file with that was generated by firesong")
parser.add_argument("--infile_source_universe",
                    type=str,
                    required=True,
                    help="Input file with that was generated by source_universe")
parser.add_argument("--density",
                    type=float,
                    default=None,
                    help="Density of sources.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print



RNG = np.random.RandomState(args.seed)
seed_bgd, seed_sig, seed_source_count = RNG.randint(0, np.iinfo(np.uint32).max, size=3)

print("Load expectation ...")
expect = LocalWarmSpotExpectation(args.expectation)

print("Load background pool ...")
bgd_pool = BackgroundLocalWarmSpotPool()
bgd_pool.load(args.infile_background, seed=seed_bgd)

print("Load signal pool ...")
sig_pool = SingleSpotTrialPool()
sig_pool.load(args.infile_signal, seed=seed_sig_pool)

print("Setup source count distribution ...")
config = "nsrc_{nsrc:08d}_n_inj_{n_inj:08d}"
firesong_config = ".".join(os.path.basename(args.infile_firesong).split(".")[:-1])
config = "firesong_{firesong_config}_seed_{args.seed}"
source_universe_config = ".".join(os.path.basename(args.infile_source_universe).split(".")[:-1])
config = "source_universe_signal_trials_{source_universe_config}_density_{args.density}_seed_{args.seed}"

source_count = None
if args.n_inj is not None and args.nsrc is not None:
    source_count = SourceCountDistEqualFluxAtEarth(phi_inj=args.n_inj, n_sources=args.nsrc)
else:
    source_count = SourceCountDistFIRESONG(infile=args.infile_source_universe, density=None)

print("Setup simulation ...")
sim = SignalSimulation( seed=None,
                        background_pool=bgd_pool,
                        single_spot_pool=sig_pool,
                        source_count_dist=source_count,
                        min_ang_dist=1.,
                        dec_range=[-3,90],
                        log10pVal_threshold=2.)

print("Setup analysis ...")
analysis = HPA_analysis(expect)

#### start generating stuff
print("Start generating trials ...")

t0 = time.time()
out = []
hottest_source = []
n_above = []
while len(out) < args.n_iter:
    data = sim.get_pseudo_experiment(**kwargs)
    hottest_source.append(np.max(data))
    n_above.append(data >= 6.542) # 5 sigma p-values
    pseudo_result = analysis.best_fit(data)
    out.append(np.array((n_tot_inj, ) + pseudo_result, dtype=[("n_inj", np.int),
                                               ("logP", np.float),
                                               ("pVal", np.float),
                                               ("count", np.int),
                                               ("exp", np.float)]))
print("... done.")
out = np.concatenate(out)
# double check that there are no nan pvalues
m = np.isfinite(out["logP"])
if np.any(~m):
    print("Found infinite logP: {0:d} of {1:d}".format(np.count_nonzero(~m), len(m)))
    for n in out.dtype.names:
        print(out[~m][n])
out = out[m]
print(time.time()-t0, "sec")

# save stuff
print("Save results")
config = str(source_count)
with open(os.path.join(args.outdir,"HPA_signal_trials_{config}.npy".format(**locals())), "w") as open_file:
    np.save(open_file, out.trials)
with open(os.path.join(args.outdir,"HPA_signal_trials_{config}.args".format(**locals())), "w") as open_file:
    cPickle.dump(args, open_file, protocol=2)
with open(os.path.join(args.outdir,"HPA_signal_trials_{config}_hottest_source.cPickle".format(**locals())), "w") as open_file:
    cPickle.dump(hottest_source, open_file, protocol=2)
with open(os.path.join(args.outdir,"HPA_signal_trials_{config}_n_above.cPickle".format(**locals())), "w") as open_file:
    cPickle.dump(n_above, open_file, protocol=2)
