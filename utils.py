#!/usr/bin/env python

#import matplotlib.pyplot as plt
import cPickle as pickle
import glob, os, re, argparse
import numpy as np
from numpy.lib.recfunctions import append_fields

from scipy.stats import poisson
from scipy.interpolate import UnivariateSpline
from skylab_data import SkylabSingleSpotTrial
from statistics import poisson_percentile, poisson_weight

class HPA_analysis(object):
    dtype = [("hpa_ts", float), ("log10p_thres", float), ("n_observed", int), ("n_expected", float)]

    def __init__(self, expectation):
        self.expectation = expectation

    def scan(self, data):
        r"""Calculate population p-values at all thresholds """
        # data contains the sorted -log10(p-value) list

        if len(data) < 1:
            return 0., 0., 0, 0.

        # n_observed in decending order, because p-value list is sorted
        n_observed = np.arange(len(data), 0, -1)
        # evaluate expectation at -log10(p-value) of data
        n_expected = self.expectation(data)
        # the poisson p-value
        poisson_pValues = -poisson(n_expected).logsf(n_observed - 1) / np.log(10)

        return poisson_pValues, data, n_observed, n_expected

    def best_fit(self, data):
        poisson_pValues, data, n_observed, n_expected = self.scan(data)
        idx = np.argmax(poisson_pValues)
        return np.array((poisson_pValues[idx], data[idx], n_observed[idx], n_expected[idx]), dtype=self.dtype)

class BackgroundLocalWarmSpotPool(object):
    def __init__(self, **kwargs):
        seed = kwargs.pop("seed", None)
        self.random = np.random.RandomState(seed)

    def load_trials(self, infiles):
        # read in all sky trials
        trials = []
        for file_name in infiles:
            with open(file_name, "r") as open_file:
                temp = pickle.load(open_file)
            trials.extend(temp)
        print "Read in %d trials"%len(trials)

        # in this mode we use all spots and draw a poisson number base on expecation
        # the mean expected number of events above p-value threshold
        self.n_expected = np.mean([len(t) for t in trials])
        # the pool of background p-values
        bgd_pool = np.concatenate([t["pVal"] for t in trials])
        # clean if there are nans or infs
        self.bgd_pool = bgd_pool[np.isfinite(bgd_pool)]

    def get_pseudo_experiment(self):
        nspots = self.random.poisson(self.n_expected)
        return self.random.choice(self.bgd_pool, nspots, replace=False)

    def load(self, load_path, seed=None):
        with open(load_path, "r") as open_file:
            state = pickle.load(open_file)
        self.bgd_pool = state["bgd_pool"]
        self.n_expected = state["n_expected"]
        if seed is not None:
            self.random = np.random.RandomState(seed)

    def save(self, save_path):
        state = {"bgd_pool": self.bgd_pool, "n_expected": self.n_expected}
        with open(save_path, "w") as open_file:
            pickle.dump(state, open_file)

class SingleSpotTrialPool(object):
    def __init__(self, **kwargs):
        self.set_seed( kwargs.pop("seed", None))

    def set_seed(self, seed):
        self.random = np.random.RandomState(seed)

    def load_trials(self, infiles, pValue_calculator):
        """ Read in sensitivity trials from given file path
        You get back a list of injection trials, detector weights and declination positions.
        In addition you have to give a pValue_calculator, thus the local pValue can be calculated."""

        trials = {}
        for file_name in infiles:
            sig_trial_file = SkylabSingleSpotTrial(file_name)
            dec = sig_trial_file.declination
            inj = sig_trial_file.trials
            mu_per_flux = sig_trial_file.mu_per_flux

            # local p-value
            pVal = -np.log10(pValue_calculator(inj["TS"], dec))
            # weights are 1/N for the moment, uniform distribution
            w = np.ones(len(inj), dtype=np.float)/ len(inj)
            inj = append_fields(inj, ["pVal", "w"], [pVal, w])
            # handle NaNs and infs
            inj = inj[np.isfinite(inj["pVal"])]

            if dec not in trials.keys():
                trials[dec] = {"mu_per_flux": [], "inj": []}
            trials[dec]["mu_per_flux"].append(mu_per_flux)
            trials[dec]["inj"].append(inj)
        for dec in trials.keys():
            trials[dec]["mu_per_flux"] = np.mean(trials[dec]["mu_per_flux"])
            trials[dec]["inj"] = np.concatenate(trials[dec]["inj"])
        self.trials = trials

    def get_random_trial(self, fluxes, decs):
        r"""Picks random trials for sources at declination `decs` with fluxes `fluxes`.
        Internal: We pick the closes declination for which trials exist and convert the
        flux into mu taking into account the detector acceptance. Trials are picked randomly
        using a poisson expectation on mu.

        Parameters:
        * fluxes: array_like
            List of neutrino flux per source. Units: ???
        * decs: array_like
            List of source declinations

        """
        sig = []
        for flux, dec in zip(fluxes, decs):
            # can not detect sources outside the simulated range
            if dec < min(self.trials.keys()) or dec > max(self.trials.keys()):
                continue
            # we take the trials from the closest declination
            nearest_dec = self.trials.keys()[np.argmin(np.abs(self.trials.keys() - dec))]
            # by converting from flux to mu we take into account the detector efficiency
            mu = self.trials[nearest_dec]["mu_per_flux"]*flux
            # we assume that mu is poisson distributed, so we chose the weights accordingly
            w = poisson_weight(self.trials[nearest_dec]["inj"]["n_inj"], mu)
            # get a random trial following the weights
            sig.append(self.random.choice(self.trials[nearest_dec]["inj"], weights=w))
        return np.concatenate(sig)

    def save(self, save_path):
        with open(save_path, "w") as open_file:
            pickle.dump(self.trials, open_file, protocol=2)

    def load(self, load_path, **kwargs):
        with open(load_path, "r") as open_file:
            self.trials = pickle.load(open_file)
        self.set_seed( kwargs.pop("seed", None))

class SignalSimulation(object):
    def __init__(self, **kwargs):
        self.random = np.random.RandomState(kwargs.pop("seed", None))
        self.background_pool = kwargs.pop("background_pool", None)
        self.single_spot_pool = kwargs.pop("single_spot_pool", None)
        self.source_count_dist = kwargs.pop("source_count_dist", None)
        self.min_ang_dist = np.radians(kwargs.pop("min_ang_dist", 1.))
        self.sinDec_range = np.sin(np.radians(kwargs.pop("dec_range", [-3,90])))
        self.log10pVal_threshold = kwargs.pop("log10pVal_threshold", 2.)
        self.solid_angle_hemisphere = 2*np.pi*(max(self.sinDec_range)-min(self.sinDec_range))
        self.solid_angle_per_source = np.pi*self.min_ang_dist**2

    def get_signal(self, **kwargs):
        r"""Generate a single Pseud Experiment by sampling fluxes from
        the source count distribution and converting it to local warm spot
        p-values."""

        fluxes = self.source_count_dist.get_fluxes(**kwargs)
        # sources are uniformly distributed on the sky
        decs = np.arcsin(np.random.uniform(-1., 1., len(fluxes)))
        injs = self.single_spot_pool.get_random_trial(fluxes, decs)
        return sig["pVal"], sig["n_inj"].sum()

    def get_pseudo_experiment(self, **kwargs):
        r"""Generates a pseudo experiment including signal and background.
        kwargs are passed to the get_signal function which further passes the
        kwargs to the source count distribution."""
        sig, n_tot_inj = self.get_signal(**kwargs)
        data = self.background_pool.get_pseudo_experiment()

        # mearge signal into background
        for i, s in enumerate(sig):
            prob = len(data)*self.solid_angle_per_source/self.solid_angle_hemisphere
            # check is two sources are to close
            if self.random.uniform(0., 1.) < prob:
                compare_idx = self.random.randint(len(data))
                # take the larger -log10(p-value)
                if data[compare_idx] < s:
                    data[compare_idx] = s
            else:
                data = np.concatenate([data, [s]])

        # Threshold cut, no pValues below  min_thres
        data = data[data >= self.log10pVal_threshold]
        # sort the data
        data = np.sort(data)
        return data
