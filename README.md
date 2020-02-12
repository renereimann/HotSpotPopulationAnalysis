# Hot Spot Population Analysis (HSPA)

This projects contains scripts to run a hotpot population analysis. By this statistical method it is tested if a population of small p-values exists in a set of p-value results which are non significant on their own.
As a use case we implement the test for p-values testing neutrino fluxes from different directions in the sky.


This will explain how to run the scripts to get a HPA result.

We need a set of all sky scans, as TS parametrization (instance of parameetrization_fit) and single PS signal trials (in the optimal case also some with very high TS values).

The procedure will start as following:

1. get_warm_spots_from_skylab_all_sky_scans.py

    This script extracts a list of local warm spots from a skylab all sky scan.
    It computs a list of (theta, phi, p-value) tuples from a skylab p-value map.

    Parameters:
    * Infiles: List of skylab all_sky_scan file paths
    * Outfile: File path of the output file, where the list with local warm spots
        should be written.
    * log10pVal_threshold: Threshold in -log10(p-value) above which local warm spots are
        considered.
    * min_ang_dist: Minimal angular distance allowed between two local warm spots.

2. generate_expectation.py

    Will check for different thresholds if spots are poissonian distributed.
    Will produce a parametrization / spline of the #spots expectation vs p-value threshold.
    The parametrization and the spline of this parametization are saved in pickle files.

    Parameters:
    * Infiles: List of inputfile pathes. Inputfile pathes should contain lists of LocalWarmSpotLists.
    * Outputfile: Location of the file where the LocalWarmSpotExpecation is saved.

3. calculate_max_local_pValues.py

    Will generate background trials and calculate the HPA TS value for these background trials. A list of TS values will be the output.

    Parameters:
    * infiles: List of inputfile pathes. Inputfile pathes should contain lists of LocalWarmSpotLists
    * outfile: File location where the list of HPA_analysis_result are saved.
    * expectation: Location of the file where the LocalWarmSpotExpecation is saved.

4. prepare_bgd_pool.py
    Generates a background pool object and saves it. Needed for signal trial generation.

    Parameters:
    * infiles: List of input files. Input files should contain a list of extracted background populations.
    * outfile: Path of the output file.

5. prepare_signal_pool.py

4. Run make_extrapolation.py
    You have to give the output from 3.

    Will read in the background HPA TS and make a gamma fit to it. Will produce plots.
    The fitted gamma-distribution parameters are the output in gamma_fit_min_ang_dist_*_min_thres_*.pickle

5. Run hpa_signal_trials.py
    You have to give the output from 1., 2. and 4. as well as the TS parametrization and the single PS sens trials and an index.

    Will generate signal trials for the HPA analysis for a fixed mu and a fixed N_source. Mu and N_source depend on the given index.
    Result will be a list of signal trials, where for each signal trial the HPA TS (-log10p-val_HPA_min) is given, the count the expectation, ...
    Result will be stored in Stefans_HPA_scripts_output_*_*.npy

6. Run IpythonNotebook to calculate sensitivity. (ToDo: Make it a python script.)

7. Run other IpythonNotebook to plot the stuff. (ToDo: Make it a python class and script.)


The script condor_chain should set up the complet work-flow. It uses a config file as input and creates directories, checks for basic inputs and produces dag files. You just run it once with the config file as arguments and then you just have to run the dag file in the correct order.


The files utils.py and sensitivity_plots.py contain functions and classes that are used in several places. These are collections of functions.


Notes:
    * We wondered why the expectation in the paper was cutting of. The difference is that the median was shown while we thought it would be the mean. The mean (also the expectation value) is still decreasing linearly.


Import dependencies
-------------------

import cPickle
import os
import glob
import argparse
import re
import time
import collections
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.interpolate import UnivariateSpline
import scipy.integrate
from scipy.special import gamma, gammaincc
from scipy.stats import poisson, norm, binom, gamma, chi2, expon, kstest
from scipy.optimize import curve_fit, minimize
from scipy.optimize import fmin_l_bfgs_b

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm

import healpy
import cosmolopy                                                                                                            # Kowalsky.py
from FIRESONG.Evolution import Evolution, RedshiftDistribution, StandardCandleSources, cosmology, Ntot                      # Kowalsky.py
from SourceUniverse.SourceUniverse import SourceCountDistribution                                                           # utils.py

Files
-----

drwxrwxr-x 2 bootcamp bootcamp   4096 Feb  6 14:27 cluster/                                                                 #
drwxrwxr-x 2 bootcamp bootcamp   4096 Feb  7 10:13 external_data/                                                           # Digitized Data for reference
drwxrwxr-x 2 bootcamp bootcamp   4096 Feb  5 16:08 plotting/                                                                #
drwxrwxr-x 7 bootcamp bootcamp 184320 Feb  6 20:51 test_data/                                                               # Test data for testing
-rw-rw-r-- 1 bootcamp bootcamp   4978 Feb  7 18:53 README.md                                                                # README
-rw-rw-r-- 1 bootcamp bootcamp  35149 Feb  5 15:08 LICENSE                                                                  # LICENSE
-rw-rw-r-- 1 bootcamp bootcamp      0 Feb  5 15:10 __init__.py                                                              # Make this a package
-rwxrwxr-x 1 bootcamp bootcamp   1943 Feb  6 15:00 get_warm_spots_from_skylab_all_sky_scans.py                              # 1. get warm spots from all sky scans
-rwxrwxr-x 1 bootcamp bootcamp   2582 Feb  6 16:35 generate_expectation.py                                                  # 2. generate expectation spline for background populations
-rwxrwxr-x 1 bootcamp bootcamp   1892 Feb  5 15:10 calculate_max_local_pValues.py                                           # 3. Generate / calculate background HPA TS
(-rw-rw-r-- 1 bootcamp bootcamp   1408 Feb  6 10:20 make_extrapolation.py                                                    # 4. Fit / extrapolate background HPA TS)
-rw-rw-r-- 1 bootcamp bootcamp    411 Feb  5 15:10 prepare_bgd_pool.py                                                      # 5. generate a background pool
-rw-rw-r-- 1 bootcamp bootcamp    807 Feb  5 15:10 prepare_signal_pool.py                                                   # 5. generate a signal pool
-rw-rw-r-- 1 bootcamp bootcamp   3750 Feb  7 18:43 hpa_firesong_signal_trials.py                                            # 6. Generate signal HPA TS trials, FIRESONG source count distributions
-rwxrwxr-x 1 bootcamp bootcamp   3487 Feb  6 10:20 hpa_signal_trials.py                                                     # 6. Generate signal HPA TS trials, equal flux source count distributions
-rwxrwxr-x 1 bootcamp bootcamp   4733 Feb  6 09:58 hpa_SourceUniverse_signal_trials.py                                      # 6. Generate signal HPA TS trials, SourceUnivers source count distr
-rw-rw-r-- 1 bootcamp bootcamp   3366 Feb  7 18:38 calculate_sensitivity.py                                                 # 7. Calculate sensitivity
-rw-rw-r-- 1 bootcamp bootcamp   2539 Feb  6 10:21 unblind_result.py                                                        # 8. The final unblinded value
-rw-rw-r-- 1 bootcamp bootcamp   5547 Feb  6 09:48 statistics.py                                                            # functions, related to statistics
-rw-rw-r-- 1 bootcamp bootcamp  39600 Feb  7 18:50 utils.py                                                                 # functions, classes
                                                   skylab_data.py                                                           # classes interacting with skylab
                                                   data_types.py                                                            # classes with data types
-rw-rw-r-- 1 bootcamp bootcamp  23233 Feb  6 09:53 SingleSpotTS2pValueParametrization.py                                    # class for TS -> pValue conversion (only used in generate signal pool)

# add plotting for
* poisson test (has been removed for now), counts above plot
* BackgroundLocalWarmSpotPool, plot_pool, histogram of self.bgd_pool

