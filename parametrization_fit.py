import numpy as np
from scipy.stats import genpareto, chi2, norm, cauchy, t, gamma, genextreme, f, poisson, kstest, percentileofscore, expon
from scipy.stats._constants import _XMAX
from scipy.special import erf
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import glob, os, cPickle, collections

from ps_analysis.scripts.stager import FileStager
import ps_analysis

class templ_fit:
    
    def __init__(self, seed=[1], **kwargs):
        self._bounds = [(0, None)]
        
        self._n_params = len(self._bounds)
        self._x = [None]*self._n_params
        self._fixed = [False]*self._n_params
        self._fixed[0] = kwargs.pop("fix_Norm", False)
        self.thres = None
        
        self.set_seed(seed)
        
    def set_seed(self, seed):
        assert len(seed) == self._n_params
        self._seed = seed
        for i in range(self._n_params):
            if self._fixed[i]: self._bounds[i] = (seed[i], seed[i])
        
    def __str__(self):
        return "NotImplemented-Function\n\tNorm={}".format(*self._x)
    
    def set_thres(self, thres):
        self.thres = thres
        
    def get_thres(self):
        return self.thres
    
    def expectation(self, xmin, xmax):
        return self._x[0]/(1.-self.cdf(self.thres))*np.abs(( self.cdf(xmax) - self.cdf(xmin) ))
    
    def pdf(self, x):
        return self._function().pdf(x)
    
    def cdf(self, x):
        return self._function().cdf(x)
    
    def _function(self):
        raise NotImplementedError()
    
    def set_params(self, x):
        self._x = x
        
    def get_seed(self):
        return self._seed
    
    def get_bounds(self):
        return self._bounds
    
    def get_norm(self):
        return self._x[0]/(1.-self.cdf(self.thres))
        
    def get_nparams(self):
        return self._n_params
        
    def get_best_fit(self):
        return self._x
        
class exponential(templ_fit):
    
    def __init__(self, seed=[1, 1], **kwargs):
        self._bounds = [(0, None), (0.1, None)]
        
        self._n_params = len(self._bounds)
        self._fixed = [False]*self._n_params
        self._fixed[0] = kwargs.pop("fix_Norm", False)
        self._fixed[1] = kwargs.pop("fix_slope", False)
        self._x = [None]*self._n_params
        self.set_seed(seed)
        self.thres = None
        
    def __str__(self):
        return "Exp-Function\n\tNorm={}\n\tSlope={}".format(*self._x)
    
    def _function(self):
        return expon(loc=0., scale=self._x[1])

class chi2_fit(templ_fit):
    
    def __init__(self, seed=[1, 1, 1], **kwargs):
        self._bounds = [(0, None), (0.5, None), (0.1, None)]
        
        self._n_params = len(self._bounds)
        self._fixed = [False]*self._n_params
        self._fixed[0] = kwargs.pop("fix_Norm", False)
        self._fixed[1] = kwargs.pop("fix_NDoF", False)
        self._fixed[2] = kwargs.pop("fix_slope", False)
        self._x = [None]*self._n_params
        self.set_seed(seed)
        self.thres = None
    
    def __str__(self):
        return "Chi2-Function\n\tNorm={}\n\tNDoF={}\n\tscale={}".format(*self._x)
    
    def _function(self):
        return chi2(self._x[1], loc=0, scale=self._x[2])
    
class test_statistic(object):    
    
    def __init__(self, infile=None):
        self._declinations = None
        
        if not infile is None: self.load(infile)
        
    def load(self, infile):
        # loads data from file
        raise NotImplementedError("The load function is not yet implemented. Sorry :( ")
        
    def save(self, outfile):
        # save stuff to file
        raise NotImplementedError("The save function is not yet implemented. Sorry :( ")
        
    def _kstest_thres(self, data, cdf, thres):
        """ This function gives the KS test probability for a cdf if just tested above above a threshold """
        cdf_thres = lambda x: (cdf(x)-cdf(thres))/(1-cdf(thres))
        return kstest(data[data>thres], cdf_thres)

    def _hist_TS(self, values, bins, density=False):
        """ This function histograms values and returns the histogram, 
        its edges, the bin centers, the bin width, and the errors on 
        the histogram entries assuming poisson errors."""
        
        hi, edg = np.histogram(values, bins=bins)
        errors = np.sqrt(hi)
        
        bin_center = (edg[1:]+edg[:-1])/2.
        dBin = (edg[1:]-edg[:-1])
        
        return hi, edg, bin_center, dBin, errors
        
    def _fit_func(self, funct, data, hi_in, edg_in, err_in, thres, verbose=False, err_est=False, mode="poisson"):
        """ fit a funct to the histogram given 
        """
        
        # if threshold given we have to mask our histogram to the region above the threshold
        # this is needed if poisson method is used
        ma = edg_in[:-1] >= thres
        hi, edg, errors = hi_in[ma], edg_in[edg_in >= thres], err_in[ma]
        dBin = (edg[1:]-edg[:-1])
        
        # set threshold in fitted function 
        funct.set_thres(thres)
        thres = funct.get_thres()
            
        def fit_call_ML(x, verbose=False):
            """ returns llh for a set of parameters x """
            funct.set_params(x)
            renorm = 1.-funct.cdf(thres)
            
            logpdf = np.log(funct.pdf(data[data >= thres])/renorm)
            finit_logpdf = np.isfinite(logpdf)
            n_bad = np.sum(~finit_logpdf)
            #if n_bad != 0: print "WARN: n_bad !=0"
            
            llh = -np.sum(logpdf[finit_logpdf]) + n_bad*np.log(_XMAX)*100
            return llh 
        
        def fit_call_poisson(x, verbose=False):
            funct.set_params(x)
            expect = funct.expectation(edg[:-1], edg[1:])
            if verbose: print expect, "\n", hi
            # poissonian LLH is given by: -sum_i (n_i*log(mu_i) - mu_i -log(n_i!))
            # we can neglect -log(n_i!) because its a constant w.r.t. mu_i
            ma = expect!=0
            return  -np.sum(hi[ma]*np.log(expect[ma]) - expect[ma]) 

        # choose fit method
        if mode=="ml":
            fit_function_call = fit_call_ML  
        elif mode=="poisson":
            fit_function_call = fit_call_poisson
        else:
            raise ValueError("Mode not known. Choose one of 'ml' or 'poisson'. Default: 'poisson'.")
        
        # okay now do the fit
        res = minimize(fit_function_call, funct.get_seed(), method="L-BFGS-B", bounds=funct.get_bounds())
           
        if not res["success"]: print "Fit failed!"
        
        # we can estimate uncertainties by doing a profile llh scan
        if err_est:
            deltaLLH   = np.sqrt(1)/2.
            profile    = []
            confidence = []
            
            # loop over all parameters
            for i in range(len(res["x"])):
                bounds = funct.get_bounds()
                seed   = funct.get_seed()
                if bounds[i][0] == bounds[i][1]:
                    """if parameter is fixed do not estimate uncertainties""" 
                    profile.append({"pars": [], "scan": []})
                    confidence.append(bounds[i])
                    continue
                    
                # now scan the range 0.9*bestfit - 1.1*bestfit
                pars = np.linspace(0.9*res["x"][i], 1.1*res["x"][i], 50)
                scan = []
                for p in pars:
                    # reset the parameter to be fixed
                    bounds[i] = (p,p)
                    seed[i]   = p
                    scan.append( minimize(fit_function_call, seed, method="L-BFGS-B", bounds=bounds)["fun"] )
                    
                # make the scan relative to the best fit
                scan = np.array(scan)-res["fun"]
                profile.append({"pars": pars, "scan": scan})

                # now get the parameter uncertainties
                if len(pars[scan < deltaLLH]) == 0: confidence.append( (np.nan, np.nan) )
                else: confidence.append( ( np.min(pars[scan < deltaLLH]), np.max(pars[scan < deltaLLH]) ) )

        # get pull from fit
        funct.set_params(res["x"])    
        expects = funct.expectation(edg_in[:-1], edg_in[1:])
        pull = np.nan*np.zeros(len(hi_in))
        ma = err_in != 0
        pull[ma] = (hi_in[ma]-expects[ma])/err_in[ma]
                
        if verbose: print funct
        
        if err_est: return funct, pull, confidence, profile
        return funct, pull

    def do_fit_single_dec(self, TS, funct=None, confidence=False, ml=False, thres=0, bins=None):
        
        # we split the sample in two sub samples to test goodness of fit
        first_half_of_trials  = TS[:len(TS)/2]
        second_half_of_trials = TS[len(TS)/2:]
        
        # get histogram
        if bins is None: bins = np.linspace(0, 40, 1000) 
        hi, edg, bin_center, dBin, errors = self._hist_TS(first_half_of_trials, bins, density=False)

        if funct is None:
            events_above_threshold = np.sum(first_half_of_trials > thres)
            
            funct = exponential(fix_Norm=True, 
                                fix_slope=False)

            seed = [1]*funct.get_nparams()
            seed[0] = events_above_threshold
            funct.set_seed(seed)

        # do fit on first half of events
        result = self._fit_func( funct, 
                                 first_half_of_trials, 
                                 hi, bins, errors,
                                 thres=thres, 
                                 err_est=confidence,
                                 mode="ml" if ml else "poisson")
        
        # get result
        funct, pull = result[0], result[1] 
        if confidence: 
            conf = result[2]
            profile = result[3]
        plot_dict = {"edg": edg, "hi": hi, "errors": errors, "pull": pull}
        
        # KS test
        ks_same = self._kstest_thres(first_half_of_trials, funct.cdf, thres=thres)
        ks_ind = self._kstest_thres(second_half_of_trials, funct.cdf, thres=thres)
        
        # save things
        ma = np.isfinite(pull)
        result_dict = { "ks":       ks_ind,
                        "ks_same":  ks_same,
                        "params":   funct.get_best_fit(),
                        "Ntrials":  len(first_half_of_trials), 
                        "Nthres":   np.sum(first_half_of_trials > thres),
                        "chi2_all": np.sum(pull[ma]**2), 
                        "chi2_one": np.sum(pull[ma][1:]**2),
                        "thres":    thres,
                        "NDoF_all": len(pull[ma]),
                        "plot":     plot_dict,
                      } 
                      
        if confidence: 
            result_dict["confidence"] = conf
            result_dict["profile"]    = profile

        return result_dict

    def do_low_TS_param(self, trials, thres):
        """ Parametrizises the low part of TS just by trials. End at threshold.
        Returns spline giving the P-value. """

        TSs = np.linspace(-10, thres, int((thres+10)*100))
        pVals = np.array([ np.sum(trials>TSmax) for TSmax in TSs], dtype=float)/len(trials)
    
        spline = UnivariateSpline(TSs, pVals, k=1, s=0)
        return spline

    def do_low_and_high_TS_param(self, dec, trials, funct=None, confidence=False, ml=False, thres=0, bins=None):
        # first high TS parametrization
        result_dict = self.do_fit_single_dec(trials, confidence=confidence, ml=ml, thres=thres, bins=bins)
        
        # add low TS parametrization
        result_dict["lowTS_param"] = self.do_low_TS_param(trials, thres=thres)
        
        # some additional book keeping
        result_dict["dec"] = dec
        result_dict["version"] = 2
        
        return result_dict

    def do_fit(self, d, files, save_dict = None, confidence=False, ml=False, kde=False, fix_scale=False, thres=0, bins=None):
        dfiles = sorted([f for f in files if "_{d}_".format(**locals()) in  f])
        
        # get trials
        TS = []
        for f in dfiles:
            temp = cPickle.load(open(f, "r"))
            TS.append(temp["TS"])    
        TS = np.concatenate(TS)
        
        result_dict = self.do_low_and_high_TS_param(d, TS, confidence=confidence, ml=ml, thres=thres, bins=bins)
        
        # save things
        if not save_dict is None:
            save_dict[float(d)] = result_dict



    def do_fit_all(self, glob_pathes):
        
        files = []
        for p in glob_pathes:
            files.extend(glob.glob(p))
        declinations = [os.path.basename(f).split("_declination_")[-1].split("_")[0] for f in files]
        declinations = sorted(list(set(declinations)))
        self._dec = declinations
        self._ks = {}
        for d in declinations[::]:
            self.do_fit(d, files, save_dict=self._ks, ml=True, kde=True, confidence=True, fix_scale=True)

class pVal_calc():
    _fit_dict = None
    
    def __init__(self, fit_dict):
        if type(fit_dict) is str and FileStager.exists(fit_dict):
            with FileStager(fit_dict, "r") as open_file:
                fit_dict = cPickle.load(open_file)
        if not type(fit_dict) is dict:
            raise IOError("We need a dict for the pVal_calc class. Either you give the dict directly or you give the path from which to load.")
            
        if fit_dict[fit_dict.keys()[0]].has_key("version") and fit_dict[fit_dict.keys()[0]]["version"] == 2:
            self._init_v2(fit_dict)
        else:
            self._init_v1(fit_dict)
        
    def _init_v1(self, fit_dict):
        self._version   = 1
        self._fit_dict  = fit_dict
        self._decs      = np.array(sorted(self._fit_dict.keys()))
        self._thres     = np.array([self._fit_dict[d]["thres"] for d in self._decs])
        self._fract     = np.array([float(self._fit_dict[d]["Nthres"])/float(self._fit_dict[d]["Ntrials"]) for d in self._decs])
        self._spline    = np.array([UnivariateSpline(self._fit_dict[d]["kde"][0], self._fit_dict[d]["kde"][1], k=1, s=0) for d in self._decs])
        self._norm_spl  = (1.-self._fract)/np.array([self._spline[i].integral(-np.inf, thr) for i, thr in enumerate(self._thres)])
        self._chi2      = np.array([chi2(*self._fit_dict[d]["params"][1:]) for d in self._decs])
        self._norm_chi2 = np.array([1-self._chi2[i].cdf(thr) for i, thr in enumerate(self._thres)])
    
    def _init_v2(self, fit_dict):
        self._version   = 2
        self._fit_dict  = fit_dict
        self._decs      = np.array(sorted(self._fit_dict.keys()))
        self._thres     = np.array([self._fit_dict[d]["thres"] for d in self._decs])
        self._fract     = np.array([float(self._fit_dict[d]["Nthres"])/float(self._fit_dict[d]["Ntrials"]) for d in self._decs])
        self._low_TS    = np.array([self._fit_dict[d]["lowTS_param"] for d in self._decs])
        self._expon      = np.array([expon(loc=0, scale=self._fit_dict[d]["params"][1]) for d in self._decs])
        self._norm_expon = np.array([1-self._expon[i].cdf(thr) for i, thr in enumerate(self._thres)])
        
    def __str__(self):
        return "This class can calculate p-values based on a fit_dict. This fit_dict contains {} different declinations.".format(len(self._decs))

    def __call__(self, TS, dec):
        return self.pVal(TS, dec)
        
    def _pVal_from_round_dec(self, TS, dec):
        if self._version == 1: 
            return self._pVal_from_round_dec_V1(TS, dec)
        elif self._version == 2:
            return self._pVal_from_round_dec_V2(TS, dec)    
        else:
            raise NotImplementedError("Just up to version 2 is implemented!")
        
    def _pVal_from_round_dec_V1(self, TS, dec):
        dec = np.atleast_1d(dec)
        TS = np.atleast_1d(TS)
        
        idx = np.searchsorted(self._decs, dec)
        thres                  = self._thres[idx]
        fract_above_thres      = self._fract[idx]
        spline                 = self._spline[idx]
        spline_norm            = self._norm_spl[idx]
        chi2_funct             = self._chi2[idx]
        chi2_norm              = self._norm_chi2[idx]

        ma = [TS > thr for thr in thres]

        pval     = np.array([1.-np.array([spline[i].integral(-np.inf,t) for t in TS])*corr for i, corr in enumerate(spline_norm)])
        for i, (d, func) in enumerate(zip(dec, chi2_funct)):
            pval[i][ma[i]] = func.sf(TS[ma[i]])/chi2_norm[i]*fract_above_thres[i]
            
        return pval
    
    def _pVal_from_round_dec_V2(self, TS, dec):
        dec = np.atleast_1d(dec)
        TS = np.atleast_1d(TS)
        
        idx = np.searchsorted(self._decs, dec)
        thres                  = self._thres[idx]
        fract_above_thres      = self._fract[idx]
        low_TS                 = self._low_TS[idx]
        expon_funct            = self._expon[idx]
        expon_norm             = self._norm_expon[idx]

        # split in low TS region and high TS region
        ma = [TS > thr for thr in thres]

        # use pValue spline for low TS values
        pval     = np.array([np.array([low_TS[i](t) for t in TS]) for i in xrange(len(low_TS))])
        
        # use fit function for high TS region
        for i, (d, func) in enumerate(zip(dec, expon_funct)):
            pval[i][ma[i]] = func.sf(TS[ma[i]])/expon_norm[i]*fract_above_thres[i]
            
        return pval
    
    def pValFunc(self, TS, dec):
        return self.pVal(TS, dec)   
        
    def pVal(self, TS, dec):
        TS_single = not isinstance(TS, (collections.Sequence, np.ndarray, np.recarray))
        dec_single = not isinstance(dec, (collections.Sequence, np.ndarray, np.recarray))
        
        if (not dec_single) and (not TS_single) and len(TS) == len(dec):
            # if TS and dec match to each other, return a single array
            dec = np.array(dec, dtype=float)
            TS = np.array(TS, dtype=float)
            pVal = np.zeros_like(dec)
            for d in sorted(set(dec)):
                m = dec==d
                pVal[m] = self._pVal_grid(TS[m], d)[0,:]
            return pVal
        
        pVal = self._pVal_grid(TS, dec)
        if dec_single and TS_single:
            return pVal[0,0]
        if dec_single:
            return pVal[0,:]
        if TS_single:
            return pVal[:,0]
        if len(TS) != len(dec):
            return pVal 
        else:
            raise NotImplementedError("This is a case that never should happen.")
        
    def _pVal_grid(self, TS, dec):
        idx_up = np.searchsorted(self._decs, dec)
        idx_low = idx_up-1

        p_low = self._pVal_from_round_dec(TS, self._decs[idx_low])
        p_up  = self._pVal_from_round_dec(TS, self._decs[idx_up])

        ones = np.ones_like(TS)
        _, dec_low = np.meshgrid(ones, self._decs[idx_low])
        _, dec_up  = np.meshgrid(ones, self._decs[idx_up])
        _, d       = np.meshgrid(ones, dec)

        return (p_low-p_up)/(dec_low-dec_up)*(d-dec_low)+p_low

    def neglogpVal(self, TS, dec):
        return -np.log10(self.pVal(TS, dec))
    
    def neglogpValFunc(self, TS, dec):
        return self.neglogpVal(TS, dec)
        
class pVal_calc_trials(pVal_calc):
    _trials = None
    
    def __init__(self, files=None, load_file=None, load_for_cat=None, skip_bad=False):
        if load_file is None and files is None and load_for_cat is None:
            raise ValueError("You have to give either files or load_file or load_for_cat")
        if not load_file is None:
            self._load(load_file)
        elif not files is None:
            self._load_trials(files)
        elif not load_for_cat is None:
            self._load_for_cat(load_for_cat, skip_bad)
        else:
            raise NotImplementedError("There are no other options implemented!")
    
    def _load(self, load_file):
        with FileStager(load_file, "r") as open_file:
            self._trials = cPickle.load(open_file)
        self._decs = np.array(sorted(self._trials.keys()))
        
    def save(self, save_file):
        with FileStager(save_file, "w") as open_file:
            cPickle.dump(self._trials, open_file, protocol=2)
    
    def _load_trials(self, files):
        """ will read in trials from basedir

        Parameter:
            - basedir, string: Path where the background trials are located

        Returns:
            - dict, keys will be declinations of catalog, for each key a numpy array with background TS trials is stored
        """

        trials = {}

        # read in trials
        for f in files:
            dec = float(os.path.basename(f).split("_declination_")[-1].split("_")[0])
            
            if not dec in trials.keys(): trials[dec] = []
            
            with FileStager(f, "r") as open_file:
                tmp = cPickle.load(open_file)
            trials[dec].append( tmp["TS"][:] )
            del tmp
            
        # convert arrays and checks
        for d in trials.keys():
            trials[d] = np.concatenate(trials[d])
            assert len(trials[dec]) > 0, "There are no trials, for declination {} that can be used to calculate the pValues.".format(d)

        self._trials = trials
        self._decs = np.array(sorted(trials.keys()))
        
    def _load_for_cat(self, load_for_cat, skip_bad):
        
        if not FileStager.exists(load_for_cat):
            raise IOError("load_for_cat directory does not exist. {load_for_cat}".format(**locals()))
        
        with open(os.path.join(os.path.dirname(ps_analysis.__file__), "catalog/source_list_dummy.pickle"), "r") as open_file:
            catalog = cPickle.load(open_file)
    
        trials = {}
    
        for dec in catalog["dec"]:
            ts_val_path = os.path.join(load_for_cat, "dec_{dec}/combined_TSValues.pickle".format(**locals()))
            if not FileStager.exists(ts_val_path):
                decs = np.array([float(os.path.basename(f)[4:]) for f in sorted(glob.glob(os.path.join(load_for_cat, "dec_*")))])
                if any(decs - dec < 1e-7):
                    print "Warn did not match exactly, numeric difference < 1e-7"
                    idx = np.argmin( decs - dec )
                    ts_val_path = os.path.join(sorted(glob.glob(os.path.join(load_for_cat, "dec_*")))[idx], "combined_TSValues.pickle".format(**locals()))
                else:
                    if skip_bad: continue
                    raise IOError("combined_TSValues not found for declination {dec}".format(**locals()))
            
            with FileStager(ts_val_path, "r") as open_file:
                trials[dec] = cPickle.load(open_file)
                
        self._trials = trials
        self._decs = np.array(sorted(trials.keys()))
    
    def __str__(self):
        return "This class can calculate p-values based on trials. Trials are read in for {} different declinations.".format(len(self._decs))
        
    def pVal(self, TS, dec):
        TS = np.atleast_1d(TS)
        if not any(np.abs(self._decs - dec) < 1e-7): raise RuntimeError("No trials for declination {dec}. We have {self._decs}".format(**locals()))
        dec = self._decs[np.argmin(np.abs(self._decs - dec))]
        
        return np.array([(100.-percentileofscore(self._trials[dec], t))/100. for t in TS])
