#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from scipy.interpolate import UnivariateSpline

import cPickle, os
import numpy as np
import external_data as stefans_results
from ps_analysis.plotting.hpa import HPA_sens_plot

# reproduce stefans paper plot
nsrc=2**(np.arange(12))
plot = HPA_sens_plot()
plot.add_line(nsrc, stefans_results.diffuse_flux_example.spline()(nsrc), label=r"$2\pi\times 10^{-11} / N_\mathrm{sources} \,[\mathrm{TeV}/(\mathrm{cm}^2\,\mathrm{s})]$",color="r", mode="theo")
plot.add_line(nsrc, stefans_results.sensitivity.spline()(nsrc), label="Sensitivity", color="r", mode="sens")
plot.add_line(nsrc, stefans_results.discovery_potential.spline()(nsrc), label="Discovery Potential", color="r", mode="disc")
plot.add_line(nsrc, stefans_results.upper_limit_90_per_cent.spline()(nsrc), label=r"90\% Upper Limit", color="r", mode="UL")

plot.plot()
plt.xlim(1e0, 2e3)
plt.ylim(1e-15, 1e-11)

# stefans bug fixed version
st_new = np.genfromtxt("external_data/flux_north.txt")
st_new_spline_sens = UnivariateSpline(st_new[:,0], st_new[:,1], s=0, k=1) 
st_new_spline_disc = UnivariateSpline(st_new[:,0], st_new[:,2], s=0, k=1)
st_new_spline_UL = UnivariateSpline(st_new[:,0], st_new[:,3], s=0, k=1) 

# my sensitivity
with open("/data/user/reimann/2017_10/HPA/hpa_sensitivity.cPickle", "r") as open_file:
    my_result = cPickle.load(open_file)
    
nsrc = my_result["nsrc"]
plot = HPA_sens_plot()

# my sensitivity
plot.add_line(my_result["nsrc"], my_result["sens"], label="Sensitivity 8yr diffuse samp.", mode="sens", color="r")
plot.add_line(my_result["nsrc"], my_result["5sig"], label="$5\sigma$ Disc. Pot. 8yr diffuse samp.", mode="disc", color="r")

# Stefans paper sensitivity
plot.add_spline(stefans_results.sensitivity.spline(), label="Sensitivity 7yr-PS-paper", mode="sens", color="lightgray")
plot.add_spline(stefans_results.discovery_potential.spline(), label="$5\sigma$ Disc. Pot. 7yr-PS-Paper", mode="disc", color="lightgray")

# Stefans bug fixed sensitivity
# note that this is integrated and in GeV. We have to divide by nsrc and multiply byn 1e-3 to get same format as for otheres
plot.add_line(nsrc, st_new_spline_sens(nsrc)/nsrc*1e-3, label="Sensitivity 7yr-PS (bug fixed)", mode="sens", color="k")
plot.add_line(nsrc, st_new_spline_disc(nsrc)/nsrc*1e-3, label="$5\sigma$ Disc. Pot. 7yr-PS (bug fixed)", mode="disc", color="k")

# Natural scaling
#plot.add_spline( lambda x: my_result["sens"][0]/x, label="Natural Scaling $\phi\,/\/N_\mathrm{Sources}$", mode="sens", color="gray")

# single PS sensitivity
# plot.add_single_ps_flux(ps_disc_min, ps_disc_max, label="Rene's single PS\n5 sigma disc. pot.", color="lightgray")
# plot.add_single_ps_flux(ps_sens_min, ps_sens_max, label="Rene's single PS\nSensitivity", color="darkgray")

_ = plot.plot()

plt.plot(nsrc, my_result["sens"] / (st_new_spline_sens(nsrc)/nsrc*1e-3), label="Sensitivity", ls="--", color="r")
plt.plot(nsrc, my_result["5sig"] / (st_new_spline_disc(nsrc)/nsrc*1e-3), label="$5\sigma$ Disc. Pot.", ls="-", color="r")
plt.xscale("log")
plt.xlabel("Number of sources")
plt.ylabel("Ratio of Flux per souce\n(8yr diff / 7yr PS)")
