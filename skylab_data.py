import cPickle as pickle
import numpy as np
import healpy
from utils import deltaPsi


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
        idx = np.argsort(p_spots)
        p_spots = p_spots[idx]
        theta_spots, ra_spots = healpy.pix2ang(nside, np.array(warm_spots_idx)[idx])
        dec_spots = np.pi/2-theta_spots

        # require min dist > x deg
        remove = []
        for i in np.arange(0, len(p_spots)):
            ang_dist = np.degrees(deltaPsi(dec_spots[i], ra_spots[i], dec_spots[i+1:], ra_spots[i+1:]))
            mask = np.where(ang_dist < min_ang_dist)[0]
            if len(mask) == 0: continue
            if any(p_spots[mask+i+1] >= p_spots[i]):
                # we have at least 2 points closer than 1 deg
                remove.append(i)

        mask = np.logical_not(np.in1d(range(len(p_spots)), remove))

        # fill into record-array
        spots = np.recarray((len(p_spots[mask]),), dtype=[("dec", float), ("ra", float), ("pVal", float)])
        spots["dec"] = dec_spots[mask]
        spots["ra"]   = ra_spots[mask]
        spots["pVal"]  = p_spots[mask]

        return spots

class SkylabSingleSpotTrial(object):
    def __init__(self, **kwargs):
        pass

if __name__ == "__main__":
    test = SkylabAllSkyScan(path="test_data/all_sky_scans_background/all_sky_scan_trial_iter2_skylab_sens_model_MCLLH3_season_IC_8yr_bestfit_spline_bin_mod_dec_2_spline_bin_mod_ener_2_prior_2.19_0.1_negTS_V2_inject_2.0_nside_256_followup_1_pseudo_experiment_3010_seed_3010.pickle")
    test.mask_hemisphere(dec_range=[0,1.5])
    lws = test.get_local_warm_spots()
    print(lws)
    print(len(lws))
