# Hot Spot Population Analysis (HSPA)

This projects contains scripts to run a hotpot population analysis. By this statistical method it is tested if a population of small p-values exists in a set of p-value results which are non significant on their own.
As a use case we implement the test for p-values testing neutrino fluxes from different directions in the sky.


This will explain how to run the scripts to get a HPA result.

We need a set of all sky scans, as TS parametrization (instance of parameetrization_fit) and single PS signal trials (in the optimal case also some with very high TS values).

The procedure will start as following:

1. Run extract_population_spots_from_skylab.py
    You have to give the input and output directory, the minimal -log10 p-value threshold, and the beginning of the seed that should be read in.

    The result will be a file that contains a list of p-value, theta and phi tuples for each local spot with a p-value smaller then threshold.
    Usually one file contains lists for several all sky scans.

2. Run check_poissonian_distribution_and_parametrizise_expectation.py
    You have to give the input and output directory, a minimal angular distance, the minimal -log10 p-value threshold and you can give a plot dir.

    Will check for different thresholds if spots are poissonian distributed. Will produce plots and put them in a plot dir
    Will produce a parametrization / spline of the #spots expectation vs p-value threshold.
    The parametrization and the spline of this parametization are saved in pickle files.
    parametrization_min_ang_dist_?.pickle and spline_expectation_min_ang_?dist_min_thres_?.py

3. Run calculate_max_local_pValues.py
    You have to give the input and output directory, the path to the expectation spline, a minimal angular distance and the minimal -log10 p-value threshold.

    Will generate background trials and calculate the HPA TS value for these background trials. A list of TS values will be the output in
    max_local_pVal_min_ang_dist*_min_thres_*.pickle

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
