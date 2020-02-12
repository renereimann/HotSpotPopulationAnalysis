try:
    from SourceUniverse.SourceUniverse import SourceCountDistribution
except:
    pass

from numpy.lib.recfunctions import append_fields

class signal_pool(object):
    def __init__(self, **kwargs):
        self.set_seed( kwargs.pop("seed", None))

    def set_seed(self, seed):
        self.random            = np.random.RandomState(seed)

    def __str__(self):
        doc = ""
        print "Different declinations:", np.shape(self.injects)
        print "Typical shape of inject per declination:", np.shape(self.injects[0])
        print "Declination range:", np.degrees(np.min(self.declinations)), np.degrees(np.max(self.declinations))
        print "Det weights shape, min and max:", np.shape(self.det_weight), np.min(self.det_weight), np.max(self.det_weight)
        print "Singal pool shape:", np.shape(self.signal_pool)
        if hasattr(self, "flux_weights"):
            print self._weight_str
        return doc

    def load_trials(self, pathes, pathes_add, pValue_calculator, pathes_add2=[]):
        """ Read in sensitivity trials from given file path
        You get back a list of injection trials, detector weights and declination positions.
        You should previde two lists of file names. In addition you have to give a pValue_calculator,
        thus the local pValue can be calculated."""

        # we want to get uniform distributed trials in sindec, thus just take equidist trials in sinDec.
        # This is how we generate the dag files
        equidist_sindec = 100
        equidist_dec = np.arcsin(np.linspace(np.sin(np.radians(-5)), 1, equidist_sindec+1))
        pathes_equidist = []
        for f in pathes:
            dec_str = re.findall("_declination_([+-]?\d*\.\d+|[+-]?\d+)", f)[0]
            if np.min(np.abs(equidist_dec - float(dec_str))) < 1e-10: pathes_equidist.append(f)
        assert len(pathes_equidist) == equidist_sindec, "you did not get all declinations"

        # det_weight should be used to correct for zenith acceptance
        # incects will be a list of signal trials
        declinations, det_weight, injects, mus, fluxes = list(), list(), list(), list(), list()
        for f1 in pathes_equidist:
                # read single source trials from sensitivity calculation
                dec_key, mu, flux, inj = self.read_in_sens_file(f1)

                # add more files with same declination
                float_re = "([+-]?\d*\.\d+|[+-]?\d+)"
                dec_str = re.findall("_declination_{float_re}".format(**locals()), f1)[0]
                for f_add in pathes_add:
                    if not dec_str in f_add: continue
                    dec_key_add, _, _,  inj_add = self.read_in_sens_file(f_add)
                    assert(np.fabs(dec_key   - dec_key_add)   < 2. * np.finfo(type(dec_key)).eps), "{dec_key}, {dec_key_add}".format(**locals())
                    inj = np.append(inj, inj_add)

                for f_add in pathes_add2:
                    if not dec_str in f_add: continue
                    inj_add = self.read_in_trial_file(f_add)
                    inj = np.append(inj, inj_add)

                # add pValue and relative weighting
                inj = append_fields(inj, ["pVal", "w"],
                                 [np.ones(len(inj), dtype=np.float),
                                  np.ones(len(inj), dtype=np.float)])
                inj["pVal"] = -np.log10(pValue_calculator(inj["TS"], dec_key))
                # weights are 1/N for the moment, uniform distribution
                inj["w"]    = np.ones(len(inj), dtype=np.float) / len(inj)
                inj         = inj[np.isfinite(inj["pVal"])]

                declinations.append(dec_key)
                det_weight.append(mu/flux)
                mus.append(mu)
                fluxes.append(flux)
                injects.append(inj)

        # the detector weights is normed relatively to the mean of the hemisphere
        det_weight        = np.asarray(det_weight)
        self.flux_2_mu    = det_weight
        self.flux         = np.array(fluxes)
        self.mu           = np.array(mus)
        det_weight       /= np.mean(det_weight)
        self.mean_flux_2_mu = np.mean(self.mu/self.flux)

        self.injects      = injects
        self.det_weight   = det_weight
        self.declinations = np.array(declinations)
        self.signal_pool  = np.concatenate(self.injects)

    def read_in_sens_file(self, path):
        """ Read in file and return declinationa flux2mu factor and trial-array with "n_inj", "TS"
        Raises if file does not exist.
        """
        if not os.path.exists(path):
            raise ValueError("Could not load signal trials, because file does not exist.\nYou specified {path}".format(**locals()))
        with open(path) as open_file:
            job_args, trials = cPickle.load(open_file)
        dec_key    = trials.keys()[0]
        mu         = np.mean(trials[dec_key][0]["mu"])
        flux       = np.mean(trials[dec_key][0]["flux"])

        return dec_key, mu, flux, trials[dec_key][1][["n_inj", "TS"]]

    def read_in_trial_file(self, path):
        """ Read in file and return declinationa flux2mu factor and trial-array with "n_inj", "TS"
        Raises if file does not exist.
        """
        if not os.path.exists(path):
            raise ValueError("Could not load signal trials, because file does not exist.\nYou specified {path}".format(**locals()))
        with open(path) as open_file:
            job_args, trials = cPickle.load(open_file)

        return trials[["n_inj", "TS"]]

    def plot_sinDec_distribution(self):
        plt.figure()
        plt.hist(np.sin(self.declinations))
        plt.xlabel("$\sin(\delta)$")
        plt.ylabel("counts")

    def plot_mu_2_flux(self):
        idxs = np.argsort(self.declinations)
        plt.plot(np.array(self.declinations)[idxs], np.array(self.flux)[idxs] / np.array(self.mu)[idxs])
        plt.xlabel("$\sin(\delta)$")
        plt.ylabel("flux/mu")

    def _weight_str(self):
        return "N_inj: {}".format(self.n_inj)

    def get_signal_weights(self, n_inj):
        """Calculates weights for each trials of injects. The mean expectation is n_inj.
        Take detector acceptance into account and poissoinan spread.
        Return signal pool with corresponding weights"""

        # scale the number of injected events for all trials according to the relative weight
        # w = Poisson(lambda=1/N*det_w*n_inj_expectation).pdf(n_inj_simulated)

        w = np.concatenate([poisson_weight(inject["n_inj"], n_inj * det_w) for inject, det_w in zip(self.injects, self.det_weight)])

        # sum over w has to be equal to len of injects because sum over w for each declination has to be 1.
        # the sum of weighted trials is 10% less then what it should be, missing edges
        # if not fullfilled essential regions of the poisson distributions are not covered

        if np.fabs(1. - w.sum() / len(self.injects)) > 0.1:
            print("{}, {}".format(w.sum(), len(self.injects)))
            raise ValueError("Poissonian weighting not possible, too few trials.\nGenerate events within ns-range from {} to {}".format(np.min(n_inj*self.det_weight), np.max(n_inj*self.det_weight)) )

        # draw equal distributed between sinDec
        w /= w.sum()
        assert np.all(w <= 0.1), "Too few trials in this regime, exit.\nSome events would have a probability larger then 10%.\nMax weights: {}".format( np.sort(w)[-10:] )

        self.flux_weights   = w
        self.n_inj             = n_inj

    def get_pseudo_experiment(self, nsrc):
        """ Draws n src p-values from signal pool and returns the local p-values as well as the n_injected summed over all nsrc."""
        if not hasattr(self, "random"):       raise ValueError("Random not yet set. Please set a seed with set_seed(seed)")
        if not hasattr(self, "flux_weights"): raise ValueError("Weights not yet initialized. Please initialize them with get_signal_weights(n_inj)")

        sig       = self.random.choice(self.signal_pool, nsrc, p=self.flux_weights)
        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj

    def save(self, save_path):
        state = {}
        state["flux_2_mu"]    = self.flux_2_mu
        state["injects"]      = self.injects
        state["declinations"] = self.declinations
        state["flux"]         = self.flux
        state["mu"]           = self.mu
        with open(save_path, "w") as open_file:
            cPickle.dump(state, open_file, protocol=2)

    def load(self, load_path, **kwargs):
        """ Loads signal pool from file,
        you can set in addition the seed"""
        if not os.path.exists(load_path): raise IOError("Inpath does not exist. {load_path}".format(**locals()))
        with open(load_path, "r") as open_file:
            state = cPickle.load(open_file)
        self.flux_2_mu    = state["flux_2_mu"]
        self.injects      = state["injects"]
        self.det_weight   = self.flux_2_mu/np.mean(self.flux_2_mu)
        self.declinations = state["declinations"]
        self.flux         = state["flux"]
        self.mu           = state["mu"]
        self.signal_pool  = np.concatenate(self.injects)
        self.set_seed( kwargs.pop("seed", None))
        self.mean_flux_2_mu = np.mean(self.mu/self.flux)

class signal_pool_FIRESONG(signal_pool):

    def load_firesong_representation(self, infile, density=None):
        fluxes = []
        with open(infile, "r") as open_file:
            for line in open_file:
                if line.startswith("#"): continue
                if len(line.split()) != 3: continue
                fluxes.append( line.split()[2] )
        self.firesong_fluxes = np.array(fluxes, dtype=float) # in GeV cm^-2 s^-1
        # self.mu   # ranges from about 6-12
        # self.flux # in 5e-10 - 4e-9 in GeV cm^-2 s^-1
        # fluxes in FIRESONG are Phi0(at 100TeV) * (100TeV)^2
        # this is the same as Phi0(at 1 GeV) *1GeV^2
        # as GeV is the basic unit this is the same as Phi0(at 1GeV) with change of units from 1/GeV cm^2 s to GeV/cm^2 s
        if density is not None:
            self.default_density = density

    def get_pseudo_experiment(self, density=None):
        """ Draws poisson mus from firesong expectations and get corresponding p-values from signal pool and returns the local p-values as well as the n_injected summed over all sources."""
        # firesong mues are expectation values and have declination dependence already integrated
        # the final mu however is a poisson random number

        if (density is not None) and (hasattr(self, "default_density")):
            if density > self.default_density:
                raise ValueError("The given density is above the default density. That is not possible.")
            fluxes = self.random.choice(self.firesong_fluxes, int(density/self.default_density*len(self.firesong_fluxes)))
        else:
            fluxes = self.firesong_fluxes

        dec_idx = self.random.choice(range(len(self.declinations)), len(fluxes))
        mus = self.random.poisson( fluxes * self.mu[dec_idx] / self.flux[dec_idx] )
        # get rid of all the null events we have to save computing

        dec_idx = dec_idx[mus!=0]
        mus = mus[mus != 0]
        sig = np.zeros(len(mus), dtype=self.signal_pool.dtype)
        for i, (mu, idx) in enumerate(zip(mus, dec_idx)):
            mask_ninj = self.injects[idx]["n_inj"] == mu
            sig[i] = self.random.choice(self.injects[idx][mask_ninj])

        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj

class signal_pool_SourceUnivers(signal_pool):

    def load_SourceUnivers_representation(self, infile, seed=None):
        self.flux_dist = SourceCountDistribution(file=infile, seed=seed)

    def get_pseudo_experiment(self, density):
        """ Draws poisson mus from firesong expectations and get corresponding p-values from signal pool and returns the local p-values as well as the n_injected summed over all sources."""
        # firesong mues are expectation values and have declination dependence already integrated
        # the final mu however is a poisson random number

        dec_idx = []
        mus = []
        n_sources = self.flux_dist.calc_N(density)

        count = 0
        while count < n_sources:
            n_loop = np.min([n_sources-count, 10000000])
            fluxes = 10**self.flux_dist.random(n_loop) # in GeV cm^-2 s^-1
            dec_idx_loop = self.random.choice(range(len(self.declinations)), len(fluxes))
            mus_loop = self.random.poisson( fluxes * self.mu[dec_idx_loop] / self.flux[dec_idx_loop] )
            count += len(mus_loop)

            # get rid of all the null events we have to save computing
            dec_idx.append(dec_idx_loop[mus_loop!=0])
            mus.append(mus_loop[mus_loop != 0])

        dec_idx = np.concatenate(dec_idx)
        mus = np.concatenate(mus)

        sig = np.zeros(len(mus), dtype=self.signal_pool.dtype)
        for i, (mu, idx) in enumerate(zip(mus, dec_idx)):
            try:
                mask_ninj = self.injects[idx]["n_inj"] == mu
                sig[i] = self.random.choice(self.injects[idx][mask_ninj])
            except:
                pass

        n_tot_inj = sig["n_inj"].sum()
        return sig["pVal"], n_tot_inj
