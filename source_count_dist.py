import numpy as np

class SourceCountDistribution(object):
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError()

    def get_fluxes(self, **kwargs):
        raise NotImplementedError()

class SourceCountDistEqualFluxAtEarth(SourceCountDistribution):
    def __init__(self, **kwargs):
        self.source_flux = kwargs.pop("phi_inj", None)
        self.n_sources = kwargs.pop("n_sources", None)

    def __str__(self):
        return "phi_inj_%s_nsrc_%s"%(self.source_flux, self.n_sources)

    def get_fluxes(self, **kwargs):
        return self.source_flux*np.ones(self.n_sources)

class SourceCountDistFIRESONG(SourceCountDistribution):
    def load_firesong_representation(self, infile, density=None):
        self.infile = infile
        fluxes = []
        with open(infile, "r") as open_file:
            for line in open_file:
                if line.startswith("#"): continue
                if len(line.split()) != 3: continue
                fluxes.append( line.split()[2] )
        self.firesong_fluxes = np.array(fluxes, dtype=float) # in GeV cm^-2 s^-1
        # fluxes in FIRESONG are Phi0(at 100TeV) * (100TeV)^2
        # this is the same as Phi0(at 1 GeV) *1GeV^2
        # as GeV is the basic unit this is the same as Phi0(at 1GeV) with change of units from 1/GeV cm^2 s to GeV/cm^2 s
        if density is not None:
            self.default_density = density

    def __str__(self):
        return "infile_%s_density_%s"%(self.infile, self.density)

    def get_fluxes(self, density=None):
        r"""Draws fluxes from a firesong generated source count distribution.
        Parameters:
        * density

        Returns:
            list of neutrino fluxes
        """
        if (density is not None) and (hasattr(self, "default_density")):
            if density > self.default_density:
                raise ValueError("The given density is above the default density. That is not possible.")
            n_sources = int(density/self.default_density*len(self.firesong_fluxes))
            fluxes = self.random.choice(self.firesong_fluxes, n_sources)
        else:
            fluxes = self.firesong_fluxes
        return fluxes
