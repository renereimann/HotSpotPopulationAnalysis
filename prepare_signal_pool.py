#!/usr/bin/env python

import glob, os
from SingleSpotTS2pValueParametrization import SingleSpotTS2pValueParametrization
from ps_analysis.hpa.utils import signal_pool

indir_sens  = "/data/user/reimann/2017_10/sensitivity/mc_trials_fixed_negTS/E-2.0/sindec_bandwidth_1deg/"
indir_sens2 = "/data/user/reimann/2017_10/sensitivity/mc_trials_fixed_negTS/E-2.0/sindec_bandwidth_1deg_TSVal/"

sens_files      = sorted(glob.glob(os.path.join(indir_sens,  "sens_*.pickle")))
additional_file = sorted(glob.glob(os.path.join(indir_sens2, "sens_*.pickle")))

pVal = SingleSpotTS2pValueParametrization(path="/data/user/reimann/2017_10/parametrization/parametrization_expon_mc_thres_5.param")
sig_pool = signal_pool(seed=None)
sig_pool.load_trials(sens_files, additional_file, pValue_calculator=pVal)

sig_pool.save("/data/user/reimann/2017_10/HPA/signal_pool.cPickle")
