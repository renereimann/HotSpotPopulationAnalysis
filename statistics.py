import numpy as np
import scipy

### Poisson distribution ###

def poisson_percentile(mu, x, y, yval):
    r"""Calculate upper percentile using a Poisson distribution.

    Parameters
    ----------
    mu : float
        Mean value of Poisson distribution
    x : array_like,
        Trials of variable that is expected to be Poisson distributed
    y : array_like
        Observed variable connected to `x`
    yval : float
        Value to calculate the percentile at

    Returns
    -------
    score : float
        Value at percentile *alpha*
    err : float
        Uncertainty on `score`

    """
    x = np.asarray(x, dtype=np.int)
    y = np.asarray(y, dtype=np.float)

    w = poisson_weight(x, mu)

    # Get percentile at yval.
    m = y > yval
    u = np.sum(w[m], dtype=np.float)

    if u == 0.:
        return 1., 1.

    err = np.sqrt(np.sum(w[m]**2)) / np.sum(w)

    return u / np.sum(w, dtype=np.float), err

def poisson_weight(vals, mean, weights=None):
    r"""Calculate weights for a sample that it resembles a Poisson.

    Parameters
    ----------
    vals : array_like
        Random integers to be weighted
    mean : float
        Poisson mean
    weights : array_like, optional
        Weights for each event

    Returns
    -------
    ndarray
        Weights for each event

    """
    mean = float(mean)
    vals = np.asarray(vals, dtype=np.int)

    if weights is None:
        weights = np.ones_like(vals, dtype=np.float)

    # Get occurences of integers.
    bincount = np.bincount(vals, weights=weights)

    n_max = len(bincount)

    # Get poisson probability.
    if mean > 0:
        p = scipy.stats.poisson(mean).pmf(range(n_max))
    else:
        p = np.zeros(n_max, dtype=np.float)
        p[0] = 1.

    # Weights for each integer
    w = np.zeros_like(bincount, dtype=np.float)
    m = bincount > 0
    w[m] = p[m] / bincount[m]

    w = w[np.searchsorted(np.arange(n_max), vals)]

    return w * weights

### Likelihood difference <-> Chi2 <-> p-values <-> sigmas ###

def pval2Sigma(pval, oneSided=False):
    r"""Converts p-values into Gaussian sigmas.

    Parameters
    ----------
    pval : float or array_like
        p-values to be converted in Gaussian sigmas
    oneSided: bool, optional, default=False
        If the sigma should be calculated one-sided or two-sided.

    Returns
    -------
    ndarray
        Gaussian sigma values

    """
    # usually not done one-sided
    if oneSided: pval *= 2.0
    sigma = scipy.special.erfinv(1.0 - pval)*np.sqrt(2)
    return sigma

def sigma2pval(sigma, oneSided=False):
    r"""Converts gGaussian sigmas into p-values.

    Parameters
    ----------
    sigma : float or array_like
        Gaussian sigmas to be converted in p-values
    oneSided: bool, optional, default=False
        If the sigma should be considered one-sided or two-sided.

    Returns
    -------
    ndarray
        p-values

    """
    pval = 1-scipy.special.erf(sigma/np.sqrt(2))
    if oneSided: pval /= 2.0
    return pval

def chiSquaredVal2pVal(chi2, dof):
    r"""Convers chi2 values into p-values assuming a chi2-distribution with
    'dof' Degrees of Freedoms.

    Parameters
    ----------
    chi2: float or array_like
        Chi2 values for which the p-value should be calculated.
    dof : float
        Degrees of Freedom used in Chi2-distribution.

    Returns
    -------
    ndarray
        p-values

    """
    pVal = scipy.stats.chi2.sf(chi2, dof)
    return pVal

def pVal2ChiSquaredVal(pval, dof):
    r"""Convers p-values into chi2 values assuming a chi2-distribution with
    'dof' Degrees of Freedoms.

    Parameters
    ----------
    pval: float or array_like
        p-values for which the chi2 values should be calculated.
    dof : float
        Degrees of Freedom used in Chi2-distribution.

    Returns
    -------
    ndarray
        chi2 values

    """
    pVal = scipy.stats.chi2.isf(pval, dof, loc=0, scale=1)
    return pVal

def llh2Sigma(llh, dof, alreadyTimes2=False, oneSided=False):
    r"""Converts a likelihood difference into gaussian sigmas assuming Wilks' theorem.

    Parameters
    ----------
    llh : float or array_like
        Likelihood difference for which the sigma should be calculated.
    dof : float
        Degrees of Freedom used in Wilks' theorem.
    alreadyTimes2 : bool, optional, default=False
        Is pure likelihood diffrence or 2*likelihood diffrence (Test statistic)
    oneSided: bool, optional, default=False
        If the sigma should be calculated one-sided or two-sided.

    Returns
    -------
    ndarray
        Gaussian sigma values

    """
    llh = np.atleast_1d(llh)
    if np.any(llh < 0): raise ValueError(  "Can not calculate the significance for a negative value of llh!"  )
    if not alreadyTimes2:   dLlhTimes2 = llh*2.0
    else:                   dLlhTimes2 = llh
    return pval2Sigma(  chiSquaredVal2pVal(dLlhTimes2, dof), oneSided  )

def sigma2ChiSquared(sigma, dof): # for contours: give sigma, want level in chi2 plot that corresponds! #
    r"""Convers Gaissian sigmas into chi2 values assuming Wilks' theorem with
    'dof' Degrees of Freedoms.

    Parameters
    ----------
    sigma: float or array_like
        Gaussian sigmas for which the chi2 values should be calculated.
    dof : float
        Degrees of Freedom used in Wilks' theorem.

    Returns
    -------
    ndarray
        chi2 values

    """
    pval  = sigma2Pval(sigma, one_sided=False)
    chi2  = pVal2ChiSquaredVal(pval, dof)
    return chi2
