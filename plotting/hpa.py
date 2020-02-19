#!/usr/bin/env python

from plot_utils import *

import os
import cPickle
import numpy as np
from scipy.stats import poisson, gamma
from statistics import poisson_percentile

# add plotting for
#* BackgroundLocalWarmSpotPool, plot_pool, histogram of self.bgd_pool
#* signal pool_ plot_sinDec_distribution, plot_mu_2_flux
#* signal_trials, plot_ninj

#Notes:
#    * We wondered why the expectation in the paper was cutting of. The difference is that the median was shown while we thought it would be the mean. The mean (also the expectation value) is still decreasing linearly.


# generate parametrization

class ks_test_poisson_parametrization_plot(base_plot):
    """ Plots the ks-test probability for poissonian fit to data.
    You have to give the parametrization, that is a list of tuples, where tuples is (thres, mean, ks_poisson, ks_binom).
    If hold_fig is True will be plotted in previous figure.
    If plot_path is given plot will be saved and closed."""

    def __init__(self, parametrization, **kwargs):
        """ Plots the ks-test probability for poissonian fit to data.
        You have to give the parametrization, that is a list of tuples, where tuples is (thres, mean, ks_poisson, ks_binom).
        If hold_fig is True will be plotted in previous figure.
        If plot_path is given plot will be saved and closed."""

        self.parametrization = parametrization

    def plot(self, fig=None, savepath=None):
        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        ax.plot(self.parametrization[:,0], self.parametrization[:,2])
        ax.plot(self.parametrization[:,0], self.parametrization[:,3], "--")
        plt.xlabel(r"$-\log_{10}(p_{\mathrm{local}})$")
        plt.ylabel("KS-Test probability")
        plt.yscale("log")
        plt.minorticks_on()
        plt.grid(b="off", which="minor")
        plt.ylim(ymin=1e-4)

class counts_above_plot(object):
    """ These are the plots from poisson test """

    def __init__(self, counts_above, thres, **kwargs):
        """ bla """
        self.thres  = thres
        self.pre_calc(counts_above)

    def pre_calc(self, counts_above):
        nbins = np.max(counts_above)-np.min(counts_above)
        self.hi, edg = np.histogram(counts_above, bins=nbins)
        xs = np.linspace(edg[0], edg[-1], edg[-1]-edg[0]+1)
        mean = np.mean(counts_above)
        self.bin_centers = (edg[1:]+edg[:-1])/2. -0.5

        self.xs = xs -0.5
        self.ys = (edg[-1]-edg[0])/float(nbins)*len(counts_above)*poisson(mean).pmf(xs)

    def plot(self, fig=None, savepath=None):
        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        ax.errorbar(self.bin_centers , self.hi, yerr=np.sqrt(self.hi), xerr=.5, ls="", label="Trials")
        ax.plot(self.xs, self.ys, drawstyle="steps-post", label="Poisson")

        plt.xlabel(r"\# Hotspots above $-\log_{10}(p_{local})$")
        plt.ylabel("counts")
        plt.legend(loc="best")
        plt.title(r"$-\log_{10}(p_{local}) = "+"%.2f"%self.thres+ "$")
        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class pseudo_exp_plot(object):
    """ """

    def __init__(self, expectation):
        """ expectation object and list of p-values"""

        self.expectation = expectation

        self.xs = np.linspace(2, 7, 1000)
        self.ys = self.expectation(self.xs)
        self.cl = []
        for cl in [0.6827, 0.9545, 0.9973]:
            upper = poisson(self.expectation(self.xs)).ppf(1.-(1.-cl)/2.)
            lower = poisson(self.expectation(self.xs)).ppf((1.-cl)/2.)
            self.cl.append((lower, upper))
        self.plot_obj = []
        self.plot_args = []

    def add(self, obj, **kwargs):
        self.plot_obj.append(obj)
        self.plot_args.append(kwargs)

    def plot_data(self, ax1, ax2, data=None, label=None, color="k"):
        # pseudo-exp. result
        line_obs = ax1.plot(self.xs, np.array([ np.sum(data > p) for p in self.xs ]), drawstyle="steps-post", color=color, lw=2)
        ax2.plot(np.concatenate([data, [data[-1]]]), np.concatenate([np.power(10, -self.expectation.poisson_test_all_pVals(data)), [1]]),
                 color=color, lw=2)

        #if local_opt is not None:
        #    line_opt = ax1.axvline(local_opt, ls="--", color="gray", lw=2, label="bla")
        #    ax2.axvline(local_opt, ls="--", color="gray", lw=2)
        return line_obs[0], label

    def plot_local_opt(self, ax1, ax2, local_opt, label=None, color="k"):
        line_opt = ax1.axvline(local_opt, ls="--", color=color, lw=2)
        ax2.axvline(local_opt, ls="--", color=color, lw=2)
        if label is None:
            return None, None
        return line_opt, label

    def plot(self, savepath=None, colors=([0.5,0.5, 1.0], [0.25,0.25,0.75], [0,0,0.5]), pLocal_min=None, pThres_max=None):
        fig = plt.figure()

        ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2], sharex=ax1)

        # expectation & CL bands
        ax1.fill_between(self.xs, self.cl[2][1], self.cl[2][0], alpha=1.0, color=colors[0])
        ax1.fill_between(self.xs, self.cl[1][1], self.cl[1][0], alpha=1.0, color=colors[1])
        ax1.fill_between(self.xs, self.cl[0][1], self.cl[0][0], alpha=1.0, color=colors[2])
        ax1.plot(self.xs, self.ys, label="Expectation", lw=2, ls="--", color="k", dashes=(3,2))

        handles = [AnyObject(colors, line=True, color="k", ls="--", lw=2, dashes=(3,2))]
        labels = ["Expectation"]
        for obj, kwargs in zip(self.plot_obj, self.plot_args):
            handle, label = obj(ax1, ax2, **kwargs)
            if handle is not None:
                handles.append(handle)
                labels.append(label)

        # labeling
        for xlabel_i in ax1.get_xticklabels():
            xlabel_i.set_visible(False)
        #ax1.set_xticklabels(ax1.get_xticklabels(), visible=False)# [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        #ax2.set_xticklabels(ax2.get_xticklabels(), visible=True) #[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ax2.set_xlabel(r"$-\log_{10}(p_\mathrm{thres})$")
        if pThres_max is not None:
            ax2.set_xlim([2,pThres_max])

        ax1.set_ylabel("$N(\leq p_\mathrm{thres})$")
        ax1.set_yscale("log", nonposy="clip")
        ax1.set_ylim(ymin=0.02)
        ax1.minorticks_on()
        ax1.grid(b="off", which="minor")

        #if local_opt is not None:
        #    handles.append(line_opt)
        #    labels.append("Largest Excess")
        ax1.legend(handles, labels, loc="best", handler_map={AnyObject: AnyObjectHandler()})

        ax2.set_ylabel("$p_\mathrm{local}$")
        ax2.set_yscale("log", nonposy="clip")
        if pLocal_min is not None:
            ax2.set_ylim([pLocal_min, 1e0])
        ax2.minorticks_on()
        ax2.grid(b="off", which="minor")

        self.save(savepath)
        return fig

    def save(self, savepath):
        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

# background fit

class gamma_fit_to_histogram(object):
    """ fit to histogram """

    def __init__(self, params, hi, edg, **kwargs):
        """ fit to histogram """
        self.hi  =     hi
        self.edg =     edg
        self.params =  params

        self.pre_calc()

    def pre_calc(self):
        self.label = "Trials (MC)"
        self.xss = np.linspace(0,7,200)
        self.yss = gamma(*self.params).pdf(self.xss)

    def plot(self, fig=None, savepath=None):
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()

        p1 = ax.plot(self.edg, np.concatenate([self.hi,[0]]), drawstyle="steps-post", label=self.label, lw=2)
        ax.plot(self.xss, self.yss, ls="--", color=p1[0].get_color(), lw=2, label="$\Gamma$-fit")

        # labeling
        plt.xlabel(r"$-\log_{10}(\min( p_{\mathrm{local}}^{\mathrm{HSP}}))$")
        plt.ylabel(r"rel. counts")
        plt.yscale("log", nonposy="clip")
        plt.legend(loc="best", framealpha=1, title="IceCube Preliminary")
        plt.minorticks_on()
        plt.grid(b="off", which="minor")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class gamma_fit_contour(object):
    """ fit to histogram """

    def __init__(self, xv, yv, sigma, xmin, ymin, **kwargs):
        """ fit to histogram """
        self.xv  =    xv
        self.yv =     yv
        self.sigma =  sigma
        self.xmin =   xmin
        self.ymin =   ymin

        self.pre_calc()

    def pre_calc(self):
        pass

    def plot(self, fig=None, savepath=None):
        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        cont = ax.contourf(self.xv, self.yv, self.sigma, range(11))
        ax.plot([self.xmin], [self.ymin], marker="o")

        # labeling
        plt.colorbar(cont).set_label("$\sigma$")
        plt.xlabel("shape")
        plt.ylabel("scale")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class gamma_fit_survival_plot(object):
    """ gamma_fit_survival_plot """

    def __init__(self, params, trials, range_median, range_3sig, range_5sig, **kwargs):
        """ fit to histogram """
        self.params       = params
        self.range_median = range_median
        self.range_3sig   = range_3sig ## [np.min(TS_3sig), np.max(TS_3sig)]
        self.range_5sig   = range_5sig ##[np.min(TS_5sig), np.max(TS_5sig)]

        self.pre_calc(trials)

    def pre_calc(self, trials):
        self.xs = np.linspace(0,7,200)
        self.y_fit = gamma(*self.params).sf(self.xs)
        self.y_trials = np.array([np.sum(trials > x, dtype=float) for x in self.xs])/len(trials)


    def plot(self, fig=None, savepath=None):
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()

        p1 = plt.plot(self.xs, self.y_trials, drawstyle="steps-post", label="Trials (MC)", lw=2)
        plt.plot(self.xs, self.y_fit , ls="--", color=p1[0].get_color(), lw=2, label="$\Gamma$-fit")

        #plt.axvspan(*self.range_median, alpha=0.5)
        plt.axhline(0.5, ls="--", color="gray", lw=2)
        plt.text(0.1, 0.5*0.5, r"median")
        #plt.axvspan(*self.range_3sig, alpha=0.5)
        plt.axhline(0.00135, ls="--", color="gray", lw=2)
        plt.text(0.1, 0.5*0.00135, r"$3\sigma$")
        #plt.axvspan(*self.range_5sig, alpha=0.5)
        #plt.axhline(2.867e-7, ls="--", color="gray", lw=2)

        # labeling
        plt.xlabel(r"$-\log_{10}(\min( p_{\mathrm{local}}^{\mathrm{HSP}}))$")
        # plt.ylabel(r"survival fraction")
        plt.ylabel(r"$p_\mathrm{post}$")
        plt.yscale("log", nonposy="clip")
        plt.legend(loc="best", title="IceCube Preliminary")
        plt.minorticks_on()
        plt.grid(b="off", which="minor")
        plt.ylim(ymin=1e-4)

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

# from sens calculation

class ninj_vs_logP_plots(object):
    def __init__(self, nsrc, trials, **kwargs):
        """ Set up plot for n_ing vs logP which is the TS for HPA.

        Parameters:
            - nsrc: int Number of Sources
            - trials: structured array, signal trials containing n_inj and TS at least
        """

        self.title = "{0:d} sources".format(nsrc)
        self.nsrc = nsrc
        self.thres = []
        self.pre_calc(trials)

    def pre_calc(self, trials):

        bins = (np.arange(trials["n_inj"].min() - 0.5, trials["n_inj"].max() + 1.5, 1.) / self.nsrc,
                np.linspace(0., trials["logP"].max(), 1000))

        h, xb, yb = np.histogram2d(trials["n_inj"].astype(np.float) / self.nsrc,
                                   trials["logP"], bins=bins)
        h = np.cumsum(h, axis=1)
        m = h[:, -1] > 0
        norm = h[m, -1]
        h = h.T
        h[:, m] = h[:, m] / norm

        self.xb = (xb[1:] + xb[:-1])/2.
        self.yb = (yb[1:] + yb[:-1])/2.
        self.hist = h

    def add_threshold(self, thres):
        """ Add a threshold value to the plot. Threshold value that should be beaten by signal trials."""
        self.thres.append(thres)

    def plot(self, N=10, fig=None, savepath=None):
        """ Plots n_ing vs logP which is the TS for HPA.

        Parameters:
            - N: int (10) Number of contourlines in the plot.
        """
        np.set_printoptions(precision=2)

        # make figure with subplots
        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        p = ax.contourf(self.xb, self.yb, self.hist,
                        cmap=plt.cm.get_cmap("Spectral_r", 10),
                        levels=[i for i in np.linspace(0., 1., N + 1)[1:-1]], vmin=0, vmax=1)
        plt.colorbar(mappable=p, ax=ax)

        # add threshold lines
        for v in self.thres:
            ax.axhline(v, color="black", linestyle="dotted")

        # labeling
        ax.set_title(self.title)
        ax.set_xlabel("Events / source")
        ax.set_ylabel("logP")
        ax.set_ylim(0., 25.)

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class find_mu_plot(object):
    def __init__(self, nsrc, **kwargs):
        """ Initialize plot to check if correct mu was calculated.

        Parameters:
            - nsrc: int Number of Sources
        """
        self.title = "{0:d} sources".format(nsrc)
        self.nsrc = nsrc
        self.threholds = []

    def add_mu_profile(self, trials, threshold, mu, eps, beta_i):
        """ Add another line for new threshold.

        Parameters:
            - trials, structured array, Signal trials at least containing n_inj and logP
            - threshold, float threshold value of TS that should be beaten by signal trials
            - mu, float Fitted number of events per source to get signal trials to threshold.
            - eps, float Tolerance of the fitter used.
            - beta_i, float Signal TS quantile.
        """

        bounds = np.percentile(trials["n_inj"], [eps, 100.-eps])
        x = np.linspace(*bounds, num=250)
        # calculate log10((quantile_line-CL)^2)
        y = [np.log10( (poisson_percentile(i, trials["n_inj"], trials["logP"],
                                              threshold)[0] - beta_i)**2 )
                         for i in x]
        self.threholds.append({"mu": mu, "x": x, "y": y})

    def plot(self, fig=None, savepath=None):
        """Plots log10( (TS-quantile -CL)^2 )."""

        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        # one line for each threshold
        for thres in self.threholds:
            l = ax.plot(thres["x"], thres["y"])
            ax.axvline(thres["mu"], color=l[0].get_color(), linestyle="dashed")

        # labeling
        ax.set_title(self.title)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$-\log_{10}( (quantile(TS(\mu)) - CL)^2 )$")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class TS_hist_plot(object):
    def __init__(self, nsrc, **kwargs):
        """ Initialize plot to histogram TS at sensitivity and discovery potential level.
        Parameters:
            - nsrc: int Number of Sources
        """

        self.title = "{0:d} sources".format(nsrc)
        self.nsrc = nsrc
        self.thresholds = []

    def add_hist_for_threshold(self, trials, w, b, b_err, threshold, mu):
        """ Add a new histogram for certain threshold.

        Parameters:
            - trials, structured array, Signal trials with at least logP
            - w, array, Poisson weights for signal trials
            - b, float, Confidence level
            - b_err, float, uncertainty on Confidence Level
            - threshold, float, Threhold that should be beaten by signal trials
            - mu ,  float , Fitted number of events per source
        """
        assert len(w) == len(trials)

        sor = np.argsort(trials["logP"])
        W = np.cumsum(w[sor]) / np.sum(w)
        up = trials["logP"][sor][W > 1. - 1.e-3][0]

        h, bins = np.histogram(trials["logP"], bins=100, weights=w, range=[0., up])

        label = ("${0:7.2%}\pm{1:7.2%}$\nabove logP = {2:.1f},"
                            " {3:.1f} events / source").format( b, b_err, threshold,
                                                mu / self.nsrc).replace("%", "\%")
        self.thresholds.append({"threshold": threshold, "h": h, "bins": bins, "label": label })

    def plot(self, fig=None, savepath=None):
        """ Plot TS histogram at sensitivity and discovery potential level."""

        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        # for each threshold one histogram
        for v in self.thresholds:
            p = ax.plot(v["bins"], np.concatenate([v["h"],[0]]), drawstyle="steps-post", label=v["label"])
            ax.axvline(v["threshold"], linestyle="dashed", color=p[0].get_color())

        ax.set_title(self.title)
        ax.set_xlabel("logP")
        ax.set_ylabel("Probability")
        ax.legend()

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class histogram_observed_vs_expected(object):
    def __init__(self, nsrc, trials, w, mu):
        """ """
        bins = [np.arange(-0.5, trials["count"].max() + 1.5, 2.)] * 2
        h, xb, yb = np.histogram2d(trials["count"], trials["exp"], weights=w, bins=bins)

        self.title = "{0:d} sources, $\mu={1:.1f}$".format(nsrc, mu)
        self.exp =trials["exp"]
        self.count = trials["count"]
        self.w = w
        self.h = h
        self.xb = xb
        self.yb = yb

    def plot(self, fig=None, savepath=None):
        """ # scatter plot, expectation vs. count """
        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        h, xb, yb, p = ax.hist2d(self.exp, self.count,
                                    weights=self.w,
                                    bins=[self.xb, self.yb], norm=LogNorm(),
                                    cmap=plt.cm.Spectral_r,
                                    vmin=1.e-3*self.h.max(), vmax=self.h.max())

        ymin, ymax = gamma(xb[1:]).interval(0.68)
        ax.fill_between(xb[1:], ymin, ymax, color="black", alpha=0.33)
        ax.plot([xb[0], xb[-1]], [yb[0], yb[-1]], color="black", linestyle="dashed")
        plt.colorbar(mappable=p, ax=ax)

        ax.set_title(self.title)
        ax.set_xlabel("Expected Count")
        ax.set_ylabel("Event Count")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig

class histogram_plocal_vs_ppost(object):
    def __init__(self, nsrc, trials, w, mu):
        """ """

        self.title = "{0:d} sources, $\mu={1:.1f}$".format(nsrc, mu)

        bins = [np.arange(-0.5, trials["count"].max() + 1.5, 2.)] * 2

        h, xb, yb = np.histogram2d(trials["pVal"], trials["logP"],
                                               weights=w, bins=100,
                                               range=[[0., 10.]] * 2)

        self.pVal =trials["pVal"]
        self.logP = trials["logP"]
        self.w = w
        self.h = h
        self.xb = xb
        self.yb = yb

    def plot(self, fig=None, savepath=None):
        """ # scatter plot, expectation vs. count """

        if fig is None:
            fig = plt.figure(figsize=(16,9))
        ax = fig.gca()

        h, xb, yb, p = ax.hist2d(self.pVal, self.logP,
                                    weights=self.w,
                                    bins=[self.xb, self.yb], norm=LogNorm(),
                                    cmap=plt.cm.Spectral_r,
                                    vmin=1.e-3*self.h.max(), vmax=self.h.max())

        plt.colorbar(mappable=p, ax=ax)

        ax.set_title(self.title)
        ax.set_xlabel("pre trial pValue")
        ax.set_ylabel("poisson pValue")

        plt.tight_layout()

        if not savepath is None:
            if not os.path.exists(os.path.dirname(savepath)):
                raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
            plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
            plt.savefig(savepath, bbox_inches='tight', dpi=600)
            plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
            plt.close()

        return fig


if __name__ == "__main__":

    plot_path = "./plots/"

    parametrization_plots     = True

    if parametrization_plots:
        min_ang_dist = 1.0
        cutoff       = 2.0
        indir        = "/data/user/reimann/2017_10/HPA/"
        with open(os.path.join(indir, "parametrization_expectiation_cutoff_pVal_{cutoff}_min_ang_dist_{min_ang_dist:.2f}.pickle".format(**locals())), "r") as open_file:
            parametrization = cPickle.load(open_file)

        current_plot = ks_test_poisson_parametrization_plot(parametrization)
        current_plot.plot(savepath=os.path.join(plot_path, "HSP_test_poissonian_ks_test_cutoff_pVal_{cutoff}_min_ang_dist_{min_ang_dist:.2f}.png".format(**locals())))
