
import argparse
import glob
import os
import cPickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib
import numpy as np
import scipy.integrate
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma, gammaincc
try:
    from FIRESONG.Evolution import Evolution, RedshiftDistribution, StandardCandleSources, cosmology, Ntot
except:
    pass

def get_luminosity_from_diffuse(fluxnorm = 0.9e-8,
                                density = 1e-9,
                                index=2.19,
                                evol="HB2006SFR",
                                zmax=10.,
                                emin=1e4,
                                emax=1e7):

    options = argparse.Namespace( density    = density,
                                  Evolution  = evol,
                                  Transient  = False,
                                  zmax       = zmax,
                                  fluxnorm   = fluxnorm,
                                  index      = index,     )

    N_sample, candleflux = StandardCandleSources(options)
    luminosity = candleflux * (1.e-5) * scipy.integrate.quad(lambda E: 2.**(index-2)*(E/1.e5)**(-index+1), emin, emax)[0] * \
                 4*np.pi * (cosmolopy.distance.luminosity_distance(1., **cosmology)*3.086e24)**2. *50526

    return luminosity

def get_Ntotal_from_density(density = 1e-9,
                            evol="HB2006SFR",
                            zmax=10.):

    options = argparse.Namespace( density    = density,
                                  Evolution  = evol,
                                  Transient  = False,
                                  zmax       = zmax)
    return Ntot(options)

########################################################################

class AnyObject(object):
    def __init__(self, colors=["r", "y", "g"], line=False, **kwargs):
        self.colors = colors
        self.line = line
        self.kwargs = kwargs

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        colors = orig_handle.colors
        step = 0.5/len(colors)
        patches = []
        for i, c in enumerate(colors):
            j = len(colors)-i
            patch = mpatches.Rectangle([x0, y0+height*(0.5-j*step)], width, height*2*j*step,
                                       facecolor=c, lw=0, transform=handlebox.get_transform())
            patches.append(patch)
        if orig_handle.line:
            print("will draw the line")
            print(orig_handle.kwargs)
            patches.append( mlines.Line2D([x0, x0+width], [y0+0.5*height, y0+0.5*height], **(orig_handle.kwargs)) )

        for patch in patches:
            handlebox.add_artist(patch)

        return patches

def add_arrow(pos_x, pos_y, color="gray", marker="s", minor_per_major = 10, tick_size=0.1, size=50, fontsize=16, ax_length=1e-2):
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ratio = bbox.width/bbox.height
    ax_len = np.log10(ax_length)
    ax.scatter(pos_x, pos_y, marker=marker, s=size, color=color)
    ax.annotate("", xy=(pos_x*10**(1.1*ax_len), pos_y*10**(1.1*ax_len)), xytext=(pos_x, pos_y),arrowprops=dict(arrowstyle="->", color=color))
    c = tick_size
    for fact in np.logspace(ax_len, 0, np.abs(ax_len)+1):
        ax.plot([fact*pos_x*(1.-c/ratio), fact*pos_x*(1.+c/ratio)], [fact*pos_y*(1.+c*ratio), fact*pos_y*(1.-c*ratio)], color=color)
        label = "$10^{%.1f}$"%np.log10(fact) if fact != 1 else "$1.0$"
        ax.text(fact*pos_x*(1.+1.2*c/ratio), fact*pos_y*(1.-1.2*c*ratio), label, fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
    c = 0.5*tick_size
    for fact in np.logspace(ax_len, 0, minor_per_major*np.abs(ax_len)+1):
        ax.plot([fact*pos_x*(1.-c/ratio), fact*pos_x*(1.+c/ratio)], [fact*pos_y*(1.+c*ratio), fact*pos_y*(1.-c*ratio)], color=color)

    ax.text(10**(1.1*ax_len)*pos_x*(1.+1.2*c/ratio), 10**(1.1*ax_len)*pos_y*(1.-1.2*c*ratio), r"$\epsilon_{\gamma/\nu}$", fontsize=fontsize, horizontalalignment='center',
            verticalalignment='top')

class kowalsky_plot(object):

    def __init__(self):
        self.plot_functions = []

    def add(self, func, **kwargs):
        self.plot_functions.append([func, kwargs])

    def plot_sources_Mertsch(self, correct_lumi=1., color="k", **kwargs):
        sources = [(9.777149525529321e+38, 0.000990530308188473, "o", "LL-AGN"),
                   (1.3884820512535637e+40, 0.000024400385861460354, "*", "SBG"),
                   (2.986734303589964e+41, 0.000006753757371189952, "h", "GC"),
                   (1.9943999858067636e+40, 3.340106622862209e-8, "v", "FR-I"),
                   (2.8213045478595147e+41, 1.851663148816149e-7, "^", "FR-II"),
                   (3.5935433862318535e+44, 1.4806572771858651e-8, "D", "BL Lac"),
                   (3.31809115493172e+45, 5.54372467076941e-11, "s", "FSRQ"),
                  ]
        sec_to_year = 3600*24*365.

        for Lnu, dens, mark, lab in sources:
            #add_arrow(Lnu*sec_to_year, dens, color="gray", marker=mark, minor_per_major = 10, tick_size=0.1, size=50, fontsize=16, ax_length=1e-2)
            plt.scatter(correct_lumi*Lnu*sec_to_year, dens, marker=mark, s=50, color=color)
            plt.text(correct_lumi*Lnu*sec_to_year, dens*2, lab)
        return None, None

    def plot_diffuse_band_Mertsch(self, correct_lumi=1., **kwargs):
        sec_to_year = 3600*24*365.
        fill_patch = mpatches.Patch(color="gray", alpha=0.2, label="diffuse flux (1% - 100%)")
        plt.fill_between([correct_lumi*9.863562924308358e+37*sec_to_year*10e-3, correct_lumi*9.863562924308358e+37*sec_to_year*10e10],
                         [0.000019280751929803258/10e-3, 0.000019280751929803258/10e10],
                         [0.0020076405107283513/10e-3, 0.0020076405107283513/10e10],
                         color="gray", alpha=0.2)
        return mpatches.Patch(color="gray", alpha=0.2), "diffuse flux (1% - 100%)"

    def plot_ps_limit_Mertsch(self, correct_lumi=1, **kwargs):
        sec_to_year = 3600*24*365.
        dm1 = lambda L: 180 * np.sqrt(L/1e42)           # Mpc, give L in erg / s
        dOmega = 4*np.pi                                # sr
        ps_limit_mertsch = lambda L, mu_ps: 1/(dOmega*dm1(L)**3)*3*gamma(mu_ps)/gamma(mu_ps-3./2)

        handle, = plt.plot([correct_lumi*1.0e+39*sec_to_year, correct_lumi*1.0e+47*sec_to_year], [ps_limit_mertsch(1e+39, 23), ps_limit_mertsch(1.0e+47, 23)], color="k", ls="--")
        return handle, "arXiv:1612.07311 Fig.4"

    def plot_ps_limit_Murase(self, correct_lumi=1., **kwargs):
        x = np.array([38.27001569858713, 45.84301412872841])
        y = np.array([-1.0266021765417181, -11.973397823458283])
        s2yr = 365*24*3600
        handle, = plt.plot(correct_lumi*10**x*s2yr, 10**y, ls="--", lw=2)
        return handle, "arXiv:1607.01601 Fig.3"

    def plot_sources_Murase(self, correct_lumi=1., color="k", **kwargs):
        s2yr = 365*24*3600
        for Lnueff, Lpheff, n0eff, n0tot, name, marker in [(3e46, 5e47, 2e-12, 1e-9, "FSRQ", "s"),
                                                           (2e44, 5e45, 5e-9, 1e-7, "BL Lac", "D"),
                                                           (2e40, 1e41, 1e-5, 3e-5, "SBG", "*"),
                                                           (1e42, 8e44, 1e-6, 2e-6, "GC-acc", "H"),
                                                           (2e40, 6e43, 1e-5, 5e-5, "GC/GG-int", "h"),
                                                           (2e42, 1e43, 1e-7, 1e-4, "RL AGN", "o"),
                                                           (7e40, 1e44, 3e-6, 1e-4, "RQ AGN", "o"),
                                                           (1e39, 1e40, 1e-3, 1e-2, "LL AGN", "o"),
                                                          ]:
            plt.plot([correct_lumi*Lnueff*s2yr, correct_lumi*Lpheff*s2yr], [n0eff, n0tot], marker="", color=color, ls="-")
            plt.scatter([correct_lumi*Lnueff*s2yr], [n0eff], color=color, marker=marker, s=50)
            plt.text(correct_lumi*0.5*Lnueff*s2yr, 0.3*n0eff, name)
        return None, None

    def diffuse_flux(self, **kwargs):
        xs = 1e+52/np.logspace(-3, 7, 100)
        ys = lambda norm: norm / xs * 1e-9
        #colors = ("red", "yellow", "green")
        #colors = ([0.5,0.5, 1.0],
        #          [0.25,0.25,0.75],
        #          [0,0,0.5])
        colors = ([0.9, 0.9, 0.9],
                  #[0.7,0.7,0.7],
                  [0.5,0.5,0.5])
        if "colors" in kwargs.keys():
            colors=kwargs["colors"]
        fix_range = lambda y: np.minimum( 0.99e-2, np.maximum( 1.01e-12, y))
        plt.fill_between(xs, fix_range( ys(1.52963340556e+52) ), fix_range( ys(8.12765595387e+52) ), color=colors[0])
        #plt.fill_between(xs, fix_range( ys(1.79718768466e+52) ), fix_range( ys(7.21042509655e+52) ), color=colors[1])
        plt.fill_between(xs, fix_range( ys(2.54180910743e+52) ), fix_range( ys(5.87124996072e+52) ), color=colors[1])
        plt.plot(xs, ys(4.12939075494e+52), color="k", ls="")

        label = "Diffuse Flux $\pm1\sigma,\,\pm3\sigma$"
        if "label" in kwargs.keys():
            label = kwargs["label"]

        return AnyObject(colors), label # PoS(ICRC2017)1005 \pm2\sigma,\,

    def label_extrapolation(self, x=1e51, y=1e-5, **kwargs):
        kw = {}
        if "fontsize" in kwargs.keys() :
            kw["fontsize"] = kwargs["fontsize"]
        plt.text(x, y, r"Extrapolation $\propto {L_{\nu_\mu+\bar{\nu}_\mu}^{\mathrm{eff}}}^{-3/2}$", **kw)
        return None, None

    def plot_2MRS_sens_Mertsch(self, correct_lumi=1, **kwargs):
        sec_to_year = 3600*24*365.
        Nbgd = 30                                       # events
        mu = 2.9                                          # events
        zc=0.02                                         # w/o units
        dOmega = 4*np.pi                                # sr
        sigma = np.sqrt(2e-2)                           # sr
        c_H0 = 299792.458/(0.71 * 100)                  # Mpc
        dm1 = lambda L: 180 * np.sqrt(L/1e42)           # Mpc, give L in erg / s
        lambda_c = lambda L: (dm1(L) / (zc*c_H0))**2    # w/o units
        h = lambda L: dOmega * dm1(L)**3. * gamma(mu-3./2.) / (3.*gamma(mu))
        nominator = lambda L: gamma(mu) + lambda_c(L)**(3./2.)*gammaincc(mu-3./2., lambda_c(L))*gamma(mu-3./2)-gammaincc(mu, lambda_c(L))*gamma(mu)
        f = lambda L: dOmega * dm1(L)**3. * nominator(L)/(3*lambda_c(L)**(3./2.)*gamma(mu))
        sqrt_term = lambda L, TSp: np.sqrt(1 + (4*Nbgd*f(L)**2 * (4*np.pi*sigma)**2/(TSp * h(L)**2)))
        rho = lambda L, TSp: TSp* h(L) / (2*f(L)**2 * (4*np.pi*sigma)**2) * (1+sqrt_term(L, TSp))

        lumis = np.logspace(40,44, 101)
        handle, = plt.loglog(lumis*sec_to_year, rho(lumis, 0.45), color="k", ls=":")  #  Eq. 2.17, assuming TS(p)=0.45
        return handle, "arXiv:1612.07311\ncorrelation with 2MRS"

    def plot_diffuse_band_evol(self, **kwargs):
        diffuse_flux_max = kwargs.pop("norm_max", 1.01e-8)
        diffuse_flux_min = kwargs.pop("norm_min", 1.01e-10)
        index        = kwargs.pop("index", 2.19)
        evol         = kwargs.pop("evol", "YMKBH2008SFR")
        zmax         = kwargs.pop("zmax", 10.)
        emin         = kwargs.pop("emin", 1e4)
        emax         = kwargs.pop("emax", 1e7)
        color        = kwargs.pop("color", "gray")
        label        = kwargs.pop("label", None)
        alpha        = kwargs.pop("alpha", 0.2)

        density_range = np.logspace(-12, -1, 3)
        lumi_100 = [get_luminosity_from_diffuse(fluxnorm=diffuse_flux_max, density=d, index=index, evol=evol, zmax=zmax, emin=emin, emax=emax) for d in density_range]
        plt.fill_between(lumi_100[::-1], density_range[::-1], density_range[::-1]*diffuse_flux_min/diffuse_flux_max,  color=color, alpha=alpha)
        if label is None:
            return  None, None
        return mpatches.Patch(color=color, alpha=alpha), label

    def plot_limit(self, correct_density=1., correct_lumi=1., **kwargs):
        lf_set = kwargs.pop("lf_set", None)
        quantile = kwargs.pop("quantile", 10)
        key = kwargs.pop("key", "logP")
        cutoff1 = kwargs.pop("min", 1e40)
        cutoff2 = kwargs.pop("max", 1e60)
        color = kwargs.pop("color", "r")
        label = kwargs.pop("label", None)
        ls = kwargs.pop("ls", "-")
        lw = kwargs.pop("lw", 1)
        TS_obs = kwargs.pop("TS_obs", None)
        extrapolate = kwargs.pop("extrapolate", False)

        lumis, ULs = lumi_UL(get_quantile(lf_set, quantile=quantile, key=key), TS_obs, cutoff=cutoff1)
        m = lumis <= cutoff2
        handle, = plt.plot(correct_lumi*lumis[m], correct_density*ULs[m], color=color, ls=ls, lw=1, **kwargs)

        if extrapolate:
            xs = np.logspace(45, np.log10(cutoff1), 10)
            ys = correct_density*np.mean(ULs[m]*lumis[m]**1.5)*xs**(-1.5)
            plt.plot(correct_lumi*xs, ys, ls=":", color=color)

        if label is None:
            return None, None
        return handle, label

    def plot_hpa(self, lf_set, quantile=10, key="logP", TS_obs=None, mode="accept", vmin = 0., vmax = 10., log=False, label=None, correct_density=1., **kwargs):
        lumi_dens_perc = get_quantile(lf_set, quantile=quantile, key=key)

        mnan = np.isfinite(lumi_dens_perc[:,2])
        mask = lumi_dens_perc[:,2][mnan] > TS_obs
        s = 10

        if mode == "perc":
            kwargs = {}
            if log:
                kwargs["norm"] = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                kwargs["vmin"] = vmin
                kwargs["vmax"] = vmax

            temp = lumi_dens_perc[:,2]
            if log:
                temp[temp <= 0] = 1e-5
            mapable = plt.scatter(lumi_dens_perc[:,0][mnan][mask], correct_density*lumi_dens_perc[:,1][mnan][mask], marker="s", s=s, c=temp[mnan][mask], lw=0, cmap="autumn", **kwargs)
            handle = plt.scatter(lumi_dens_perc[:,0][mnan][~mask], correct_density*lumi_dens_perc[:,1][mnan][~mask], marker="o", s=s, c=temp[mnan][~mask], lw=0, cmap="autumn", **kwargs)

            cb = plt.colorbar(mapable, pad=0.125)
            cb.set_label(label)
            if TS_obs is not None:
                cb.ax.plot([0,1], [TS_obs/vmax]*2, color='w', ls="-", lw=3)

            return None, None

        handle1 = plt.scatter(lumi_dens_perc[:,0][mnan][mask], correct_density*lumi_dens_perc[:,1][mnan][mask], marker="s", s=s, color="r")
        handle2 = plt.scatter(lumi_dens_perc[:,0][mnan][~mask], correct_density*lumi_dens_perc[:,1][mnan][~mask], marker="o", s=s, color="g")
        if label is None:
            return None, None
        return [handle1, handle2], label

    def plot_fails(self, lumi_dens, **kwargs):
        handle, = plt.scatter(lumi_dens[:,0], lumi_dens[:,1], marker="x", s=30, color="k")
        return handle, "failed"

    def plot(self, title=None, on_plot=True, right_axis=True, figsize=(6.4, 4.8), fontsize=None, fig=None):
        if fontsize is not None:
            matplotlib.rcParams["font.size"] = fontsize
        if fig is None:
            plt.figure(figsize=figsize)
        if not on_plot:
            plt.subplot2grid((1,5), (0,0), colspan=3)
        #ax = plt.gca()
        ax = fig.add_subplot(111)

        handles = []
        labels = []
        for func, kwargs in self.plot_functions:
            handle, label = func(**kwargs)
            if handle is not None:
                if type( handle ) == list:
                    for h, l in zip(handle, label):
                        handles.append(h)
                        labels.append(l)
                else:
                    handles.append(handle)
                    labels.append(label)

        ax.set_ylim([1e-12, 1e-2])
        ax.set_yscale("log")
        ax.set_ylabel(r"$\rho_0^\mathrm{eff}\,[\mathrm{Mpc}^{-3}]$")
        ax.set_xlim([1e45, 1e55])
        ax.set_xscale("log")
        ax.set_xlabel(r"$L_{\nu_\mu+\bar{\nu}_\mu}^\mathrm{eff}\,\left[\frac{\mathrm{erg}}{\mathrm{yr}}\right]$")

        if right_axis:
            ax2 = ax.twinx()
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim([get_Ntotal_from_density(density=ymin), get_Ntotal_from_density(density=ymax)])
            ax2.set_yscale("log")
            ax2.set_ylabel(r'$N_\mathrm{sources}$ assuming SFR')

        ax3 = ax.twiny()
        xmin, xmax = ax.get_xlim()
        ax3.set_xlim([xmin/(365.*24.*3600.), xmax/(365.*24.*3600.)])
        ax3.set_xscale("log")
        ax3.xaxis.labelpad = 10
        ax3.set_xlabel(r"$L_{\nu_\mu+\bar{\nu}_\mu}^\mathrm{eff}\,\left[\frac{\mathrm{erg}}{\mathrm{s}}\right]$")

        if on_plot:
            plt.legend(handles, labels,  loc=3, title=title, handler_map={AnyObject: AnyObjectHandler()}) #fontsize='medium',
        else:
            plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05 if len(plt.gcf().axes)==1 else 1.35, 1.), fontsize=14, title=title, handler_map={AnyObject: AnyObjectHandler()})

    def save(self, savepath):
        plt.savefig(savepath, bbox_inches='tight', dpi=600)
        plt.savefig(savepath.replace(".png", "_low_reso.png"), bbox_inches='tight', dpi=72)
        plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')

########################################################################

class TS_dist(object):
    def __init__(self, lumi, density, np=True):
        self.lumi = lumi
        self.density = density
        self.TS = None
        self.np = np

    def add(self, path):
        if not float(os.path.basename(path).split("_")[10]) == self.density:
            return
        try:
            if self.np:
                temp = np.load(path)
            else:
                with open(path, "r") as open_file:
                    temp2 = cPickle.load(open_file)
                if type(temp2[0]) == np.ndarray:
                    temp2 = [np.sum(t) for t in temp2]
                temp = np.zeros(len(temp2), dtype=[("TS", np.float32)])
                temp["TS"] = temp2
            if self.TS is None:
                self.TS = np.atleast_1d(temp)
            else:
                self.TS = np.concatenate([self.TS, np.atleast_1d(temp)])
        except Exception as e:
            print e
            # raise

class lumi_set(object):
    def __init__(self, path, np=True, prefix=""):
        print os.path.basename(path)
        self.lumi_path = path
        extension = "*.npy" if np else "*.cPickle"
        self.files = glob.glob(os.path.join(path,prefix+extension))

        self.lumi = os.path.basename(path).split("_")[1]
        self.densities = set([float(os.path.basename(f).split("_")[10]) for f in self.files])

        self.TSs = [TS_dist(self.lumi, d, np=np) for d in self.densities]
        for f in self.files:
            for ts in self.TSs:
                ts.add(f)

def get_quantile(lf_sets, quantile=10, key="logP"):
    coords = []
    for lf in lf_sets:
        for TS in lf.TSs:
            if len(TS.TS) < 5:
                continue
            if key=="ntrials":
                quant = len(TS.TS)
            else:
                quant =  np.percentile( TS.TS[key], quantile)
            coords.append((float(TS.lumi), float(TS.density), quant))
    coords = np.array(coords)
    return coords

def lumi_UL(coords, TS_obs, cutoff=3e47):

    lumis = []
    ULs = []

    # loop over luminosities
    for l in set(coords[:,0]):
        # just select one luminosity slice
        mask = coords[:,0] == l
        # sort by density otherwise spline gives nan
        idx = np.argsort(coords[:,1][mask])

        # spline TS vs. density
        dens = coords[:,1][mask][idx]
        TS = coords[:,2][mask][idx]

        if len(dens) <= 1: continue

        spl = UnivariateSpline(dens, TS, k=1, s=0)

        # Find UL point by two iteration scan
        UL = lambda density: (spl(density) - TS_obs)**2

        dens_scan_iter1 = np.logspace(np.log10(dens.min()), np.log10(dens.max()), 100)
        min_iter1 = dens_scan_iter1[np.argmin(UL(dens_scan_iter1))]

        dens_scan_iter2 = np.logspace(np.log10(min_iter1)-0.1, np.log10(min_iter1)+0.1, 100)
        min_iter2 = dens_scan_iter2[np.argmin(UL(dens_scan_iter2))]

        lumis.append(l)
        ULs.append(min_iter2)

    idx = np.argsort(lumis)
    lumis = np.array(lumis)[idx]
    ULs = np.array(ULs)[idx]
    mask = lumis > cutoff
    lumis = lumis[mask]
    ULs = ULs[mask]

    return lumis, ULs
