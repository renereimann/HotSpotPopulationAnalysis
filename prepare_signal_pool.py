#!/usr/bin/env python

import os, argparse, glob
from SingleSpotTS2pValueParametrization import SingleSpotTS2pValueParametrization
from utils import SingleSpotTrialPool

parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    type=str,
                    default="test_data/single_spot_trials/",
                    help="Folder containing single spot trial files.")
parser.add_argument("--outfile",
                    type=str,
                    default="test_data/signal_pool.pickle",
                    help="Path of the output file.")
parser.add_argument("--parametrization",
                    type=str,
                    default="test_data/parametrization_TS_to_p_value_exponential_fit_above_threshold_5_from_mc.param",
                    help="Path of the TS parametrization.")
args = parser.parse_args()

print "Run", os.path.realpath(__file__)
print "Use arguments:", args
print

pVal = SingleSpotTS2pValueParametrization(path=args.parametrization)

infiles = sorted(glob.glob(os.path.join(args.indir,  "sens_*.pickle")))
sig_pool = SingleSpotTrialPool(seed=None)
sig_pool.load_trials(infiles, pValue_calculator=pVal)
sig_pool.save(args.outfile)
