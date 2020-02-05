#!/usr/bin/env python

from ps_analysis.hpa.utils import background_pool

bgd_pool = background_pool(seed=1234, min_ang_dist=1., cutoff=2., cut_on_nspot=None) # 250
bgd_pool.load_trials( "/data/user/reimann/2017_10/HPA/local_spots/mc_trials/all_sky_population_bgd_trials_cutoff_pVal_2.0_seed_*X.pickle" )
bgd_pool.save("/data/user/reimann/2017_10/HPA/bgd_pool_cutoff_2_min_ang_dist_1_cut_on_nspot_None.pickle")
