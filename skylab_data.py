import cPickle as pickle
import numpy as np
import healpy
from data_types import LocalWarmSpotList

def deltaPsi(dec1, ra1, dec2, ra2):
    """Calculate angular distance between two directions.

    Parameters
    ----------
    dec1: float, array_like
        Declination of first direction. Units: radian
    ra1: float, array_like
        Right ascension of first direction. Units: radian
    dec2: float, array_like
        Declination of second direction. Units: radian
    ra2: float, array_like
        Right ascension of second direction. Units: radian

    Returns
    -------
    ndarray
        Angular distance. Units: radian
    """

    cDec1 = np.cos(dec1)
    cDec2 = np.cos(dec2)
    cosTheta = cDec1*np.cos(ra1)*cDec2*np.cos(ra2) + cDec1*np.sin(ra1)*cDec2*np.sin(ra2) + np.sin(dec1)*np.sin(dec2)
    cosTheta[cosTheta>1.] = 1.
    cosTheta[cosTheta<-1.] = -1.
    return np.arccos(cosTheta)

class SkylabAllSkyScan(object):
    r"""This class represents a single all sky scan in skylab.
    The class can load the skymap of the all sky scan from disc and allows to
    interface with the sky map.

    The key function for the Hot Spot Population Analysis is the
    get_local_warm_spots function.

    The init takes a single optional parameter which is the path of the input
    file from which the skymap can be loaded.
    """
    def __init__(self, **kwargs):
        self.log10p_map = None
        self.dec_map = None
        self.ra_map = None
        if "path" in kwargs.keys():
            self.load_from_file(kwargs["path"])

    @property
    def log10p_map(self):
        return self._log10p_map
    @log10p_map.setter
    def log10p_map(self, value):
        self._log10p_map = value

    @property
    def ra_map(self):
        return self._ra_map
    @ra_map.setter
    def ra_map(self, value):
        self._ra_map = value

    @property
    def dec_map(self):
        return self._dec_map
    @dec_map.setter
    def dec_map(self, value):
        self._dec_map = value

    def load_from_file(self, fileName):
        """Reads a skylab all sky scan from file.

        Parameters
        ----------
        fileName: str
            The filepath from which we can read the all sky scan
        """
        with open(fileName, "r") as open_file:
            job_args, scan = pickle.load(open_file)
        self.log10p_map = scan[0]["pVal"]
        self.dec_map = scan[0]["dec"]
        self.ra_map = scan[0]["ra"]

    def mask_hemisphere(self, dec_range):
        """Set all p-values on the wrong hemisphere to 1.

        Parameters
        ----------
        dec_range: tuple
            Range that is used for analysis. Everything outside gets masked.
        """
        mask = np.logical_or(self.dec_map < min(dec_range), self.dec_map > max(dec_range))
        self.log10p_map[mask] = 0

    @staticmethod
    def apply_seperation(spots, min_ang_dist):
        remove = []
        for i in np.arange(0, len(spots)):
            ang_dist = deltaPsi(spots.list["dec"][i], spots.list["ra"][i], spots.list["dec"][i+1:], spots.list["ra"][i+1:])
            mask = np.where(ang_dist < np.radians(min_ang_dist))[0]
            if len(mask) == 0: continue
            if any(spots.list["pVal"][mask+i+1] >= spots.list["pVal"][i]):
                remove.append(i)
        mask = np.logical_not(np.in1d(range(len(spots)), remove))
        return spots.list[mask]


    def get_local_warm_spots(self, log10p_threshold=2, min_ang_dist=1):
        r"""Extract local warm spots from a p-value skymap.

        Parameters
        ----------
        log10p_threshold: float, default=2
            Threshold on log10(p-value), above local warm spots should be considered.
        min_ang_dist: float, units: degree, default: 1
            Minimal distance between two local warm spots.

        Returns
        -------
        ndarry ("dec":float, "ra": float, "pVal": float)
            List of local warm spots. Each warm spot is described by a tuple (dec, ra, p-value)

        """
        log10p = self.log10p_map

        # get npix and nside
        npix = len(log10p)
        nside = healpy.npix2nside(npix)

        # mask large p-values and infs
        mask = np.logical_and(log10p > log10p_threshold, np.isfinite(log10p))
        warm_spots_idx = []
        for pix in np.arange(npix)[mask]:
            theta, phi = healpy.pix2ang(nside, pix)

            # if no larger neighbour, we are at a spot
            neighbours = healpy.get_all_neighbours(nside, theta, phi)
            if not any(log10p[neighbours] > log10p[pix]):
                warm_spots_idx.append(pix)

        # get pVal and direction of spots and sort them
        p_spots = log10p[warm_spots_idx]
        theta_spots, phi_spots = healpy.pix2ang(nside, warm_spots_idx)

        # fill into record-array
        spots = LocalWarmSpotList()
        spots.add(theta=theta_spots, phi=phi_spots, pVal=p_spots)

        spots = SkylabAllSkyScan.apply_seperation(spots, min_ang_dist)

        return spots

class SkylabSingleSpotTrial(object):
    def __init__(self, path, **kwargs):
        self.load(path)

    def load(self, path):
        with open(path, "r") as open_file:
            job_args, data = pickle.load(open_file)
        self.declination = job_args.declination
        sens = data[self.declination][0]
        trials = data[self.declination][1]
        self.mu_per_flux = np.mean(sens["mu"]/sens["flux"])
        self.trials = trials[["n_inj", "TS"]]
