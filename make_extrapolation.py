#!/usr/bin/env python

import os, cPickle, argparse
from scipy.stats import gamma, kstest
from scipy.optimize import minimize
import numpy as np
from statistics import llh2Sigma

def make_gamma_fit(read_path, verbose=False, 
                   grid_size=[16,16], hold_fig=False,  
                   plot_hist=True, plot_path_hist=None, 
                   plot_contour=True, plot_path_contour=None, 
                   plot_survival=True, plot_path_survival=None, label=None):
    with open(read_path, "r") as read_file:
        trials = cPickle.load(read_file)
    
    params = gamma.fit(trials, floc=0)
    
    ks_gamma = kstest(trials, gamma(*params).cdf, alternative="greater")[1]
        
    # histogram
    hi, edg = np.histogram(trials, bins=np.linspace(0, 5, 21), density=True)
    
    # calculate LLH landscape
    x = np.linspace(params[0]-0.3, params[0]+0.3, grid_size[0])
    y = np.linspace(params[2]-0.1, params[2]+0.1, grid_size[1])
    xv, yv = np.meshgrid(x, y)
    llh = [-np.sum(gamma(x1, 0, x2).logpdf(trials)) for x1, x2 in zip(xv.flatten(), yv.flatten())]
    dllh = np.reshape(llh, np.shape(xv))-np.min(llh)
    xmin = xv.flatten()[np.argmin(dllh)]
    ymin = yv.flatten()[np.argmin(dllh)]
    sigma = llh2Sigma(dllh, dof=2, alreadyTimes2=False, oneSided=False)
    
    # median
    trials.sort()    
    median = np.median(trials)
    median_lower = trials[int(len(trials)/2.-np.sqrt(len(trials))*0.5)]
    median_upper = trials[int(len(trials)/2.+np.sqrt(len(trials))*0.5)+1]
    
    # extrapolation
    ma = sigma < 1
    TS_3sig = []
    TS_5sig = []
    for xx, yy in zip(xv[ma], yv[ma]):    
        func_3sig = lambda p: np.log((gamma(xx, 0, yy).sf(p)-0.00135)**2)
        func_5sig = lambda p: np.log((gamma(xx, 0, yy).sf(p)-2.867e-7)**2)
        res_3sig = minimize(func_3sig, x0=[5],  method="L-BFGS-B")
        res_5sig = minimize(func_5sig, x0=[10], method="L-BFGS-B")
        TS_3sig.append(res_3sig["x"][0])
        TS_5sig.append(res_5sig["x"][0])
    func_3sig = lambda p: np.log((gamma(*params).sf(p)-0.00135)**2)
    func_5sig = lambda p: np.log((gamma(*params).sf(p)-2.867e-7)**2)
    res_3sig = minimize(func_3sig, x0=[5],  method="L-BFGS-B")
    res_5sig = minimize(func_5sig, x0=[10], method="L-BFGS-B")
    ext_3sig = res_3sig["x"][0]
    ext_5sig = res_5sig["x"][0] 
    lower_3sig = np.min(TS_3sig)
    upper_3sig = np.max(TS_3sig)
    lower_5sig = np.min(TS_5sig)
    upper_5sig = np.max(TS_5sig)
        
    if verbose: 
        print "Fit parameters:", params
        print "KS-Test between fit and sample", ks_gamma
        print "Median at {:.2f} + {:.2f} - {:.2f}".format(median, median-median_lower, median_upper-median)
        print "3 sigma at {:.2f} + {:.2f} - {:.2f}".format(ext_3sig, ext_3sig-lower_3sig, upper_3sig-ext_3sig)
        print "5 sigma at {:.2f} + {:.2f} - {:.2f}".format(ext_5sig, ext_5sig-lower_5sig, upper_5sig-ext_5sig)    
    
    if plot_hist:
        curr_plot = gamma_fit_to_histogram(params, hi, edg)
        curr_plot.plot(savepath=plot_path_hist)
            
    if plot_contour:
        curr_plot = gamma_fit_contour(xv, yv, sigma, xmin, ymin)
        curr_plot.plot(savepath=plot_path_contour)
        
    if plot_survival:
        curr_plot = gamma_fit_survival_plot( params, trials, 
                                             (median_lower, median_upper), 
                                             (lower_3sig, upper_3sig), 
                                             (lower_5sig, upper_5sig))
        curr_plot.plot(savepath=plot_path_survival)
            
    return {"params": params, "ks_pval": ks_gamma, "median": (median, (median_lower, median_upper)), "3sig": (ext_3sig, (lower_3sig, upper_3sig)), "5sig": (ext_5sig, (lower_5sig, upper_5sig))}
    

parser = argparse.ArgumentParser()
parser.add_argument("--infile",
                    type=str,
                    required=True,
                    help="Give inpath.")
parser.add_argument("--outdir",
                    type=str,
                    required=True,
                    help="Give outpath.")
parser.add_argument("--plotdir",
                    type=str,
                    required=False,
                    default=None,
                    help="Give plot dir.")
args = parser.parse_args()
 
post_fix = os.path.basename(args.infile).replace("max_local_pVal_", "").replace(".pickle", "") 
kwargs = {}
if args.plotdir is not None:
    kwargs["plot_hist"] = True
    kwargs["plot_path_hist"] = os.path.join(args.plotdir, "max_pVal_hist_{post_fix}.png".format(**locals()))
    kwargs["plot_contour"] = True
    kwargs["plot_path_contour"] = os.path.join(args.plotdir, "fit_llh_lands_{post_fix}.png".format(**locals()))
    kwargs["plot_survival"] = True
    kwargs["plot_path_survival"] = os.path.join(args.plotdir, "extrapolation_{post_fix}.png".format(**locals()))
 
fit_stuff = make_gamma_fit(args.infile, **kwargs)

# save stuff     
with open(os.path.join(args.outdir, "gamma_fit_{post_fix}.pickle".format(**locals())) , "w") as open_file:
    cPickle.dump(fit_stuff, open_file)
