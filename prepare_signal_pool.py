#!/usr/bin/env python

import glob, os
from SingleSpotTS2pValueParametrization import SingleSpotTS2pValueParametrization
from utils import SingleSpotTrialPool

indir_sens  = "test_data/single_spot_trials/"
parametrization_path = "test_data/parametrization_TS_to_p_value_exponential_fit_above_threshold_5_from_mc.param"

sens_files = sorted(glob.glob(os.path.join(indir_sens,  "sens_*.pickle")))

pVal = SingleSpotTS2pValueParametrization(path=parametrization_path)

sig_pool = SingleSpotTrialPool(seed=None)
sig_pool.load_trials(sens_files, pValue_calculator=pVal)
sig_pool.save("test_data/signal_pool.cPickle")
