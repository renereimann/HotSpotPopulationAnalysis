#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import cPickle as pickle

import external_data as stefans_results
class HPA_sens_plot(object):
    def __init__(self, xmin=1, xmax=1e3):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = None
        self.ymax = None

        self.lines = []
        self.regions = []

    def _update_ylim(self, y):
        ymin = 10**np.floor(np.log10(np.min(y)))
        ymax = 10**np.ceil( np.log10(np.max(y)))
        if self.ymin == None or self.ymin > ymin: self.ymin=ymin
        if self.ymax == None or self.ymax < ymax: self.ymax=ymax

    def add_line(self, x, y, label, mode, color, no_marker=False):
        line = {}
        line["x"] = x
        line["y"] = y
        line["label"] = label
        line["ls"] = "--" if mode=="sens" else ":" if mode=="theo" else "-"
        line["marker"] = "o" if mode != "theo" and no_marker==False else ""
        line["color"] = color
        line["lw"] = 2 if mode != "UL" else 4
        self.lines.append(line)
        self._update_ylim(line["y"])

    def add_spline(self, spline, label, mode, color):
        x = np.logspace(np.log10(self.xmin), np.log10(self.xmax), 1000)
        self.add_line(x, spline(x), label, mode, color, no_marker=True)

    def add_single_ps_flux(self, min_flux, max_flux, label, color="lightgray"):
        region = {}
        region["min"] = min_flux
        region["max"] = max_flux
        region["label"] = label
        region["color"] = color
        self.regions.append(region)

    def plot(self, fig=None, savepath=None, preliminary=True):
        if fig is None:
            fig = plt.figure(figsize=(8,6))
        ax = fig.gca()

        for line in self.lines:
            ax.loglog(line["x"], line["y"], label=line["label"], marker=line["marker"], linestyle=line["ls"], color=line["color"], lw=line["lw"])

        for region in self.regions:
            plt.axhspan(region["min"], region["max"], facecolor=region["color"], alpha=0.3, lw=0)
            plt.text(9e2, 0.9*region["max"], region["label"], horizontalalignment="right", verticalalignment="top")

        # labeling
        ax.set_xlabel("Number of sources")
        ax.set_ylabel(r"$E^2\frac{\partial\phi_\mathrm{Source}}{\partial E}"
                      r"\,/\,"
                      r"\left(\mathrm{TeV}\,/\,\mathrm{cm}^2\,\mathrm{s}\right)$")
        ax.set_ylim(self.ymin, self.ymax)
        ax.minorticks_on()
        ax.grid(b="off", which="minor")

        l = ax.legend(markerscale=0, loc="best")
        if preliminary:
            l.set_title("IceCube Preliminary")
        plt.setp(l.get_title(), color="r")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig


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
    my_result = pickle.load(open_file)

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
